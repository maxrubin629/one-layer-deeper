from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch, count_model_state_elements
from benchmark.validation import validate_optimizer, validate_submission
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
CANDIDATES = (
    "v3_factored_orbit_workspace",
    "v3.1_damped_correction",
    "v3.2_local_state_refiner",
    "v3.3_gabor_corrector",
    "v4_attractor_picard_corrector",
    "v4.1_attractor_anderson_corrector",
    "v5_sequence_smoothmax_loss",
    "v5.1_prompt_reconstruction",
    "v5.2_composition_stability",
)
RESEARCH = "v3.4_factored_exact_t_diagnostic"


def load_source(experiment_id: str, filename: str = "submission.py"):
    path = EXPERIMENTS / experiment_id / filename
    spec = importlib.util.spec_from_file_location(
        "test_" + experiment_id.replace(".", "_"), path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StructuredExperimentTests(unittest.TestCase):
    model_spec = ModelSpec(17, 12, 500_000_000)
    optimizer_spec = OptimizerSpec(1.0, "cpu")

    def test_layout_metadata_and_standalone_source_policy(self) -> None:
        for experiment_id in CANDIDATES:
            with self.subTest(experiment_id=experiment_id):
                directory = EXPERIMENTS / experiment_id
                metadata = json.loads((directory / "experiment.json").read_text())
                self.assertEqual(metadata["id"], experiment_id)
                self.assertEqual(metadata["eligibility"], "candidate-safe")
                self.assertEqual(metadata["entrypoint"], "submission.py")
                source = (directory / "submission.py").read_text(encoding="utf-8")
                self.assertEqual(
                    validate_submission_source("submission.py", source, 256 * 1024),
                    "submission.py",
                )
                self.assertLessEqual(len(source.encode("utf-8")), 256 * 1024)

        research_dir = EXPERIMENTS / RESEARCH
        metadata = json.loads((research_dir / "experiment.json").read_text())
        self.assertEqual(metadata["eligibility"], "research-only")
        self.assertEqual(metadata["entrypoint"], "research.py")
        self.assertTrue((research_dir / "BLOCKED_FROM_SUBMISSION").is_file())
        self.assertFalse((research_dir / "submission.py").exists())

    def test_candidates_load_train_and_cover_every_parameter(self) -> None:
        input_ids = torch.randint(0, self.model_spec.vocab_size, (2, 8))
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        for experiment_id in CANDIDATES:
            with self.subTest(experiment_id=experiment_id):
                module = load_source(experiment_id)
                validate_submission(module.SUBMISSION)
                model = module.SUBMISSION.build_model(self.model_spec)
                bundle = module.SUBMISSION.build_optimizer(model, self.optimizer_spec)
                validate_optimizer(bundle, model, torch.device("cpu"))
                logits, auxiliary = model(input_ids, mask)
                self.assertEqual(tuple(logits.shape), (2, 8, 17))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertLessEqual(
                    count_model_state_elements(model),
                    self.model_spec.maximum_model_state_elements,
                )
                loss = logits.float().square().mean()
                if module.SUBMISSION.token_training_loss is not None:
                    labels = torch.randint(0, 17, (2, 3))
                    valid = torch.tensor([[True, True, True], [True, True, False]])
                    loss = module.SUBMISSION.token_training_loss(
                        TokenLossBatch(logits[:, :3].float(), labels, valid, None, auxiliary)
                    )
                self.assertTrue(loss.isfinite())
                loss.backward()
                bundle.optimizer.step()

    def test_orbit_transition_is_autonomous_and_residual(self) -> None:
        for experiment_id in CANDIDATES:
            with self.subTest(experiment_id=experiment_id):
                module = load_source(experiment_id)
                signature = tuple(inspect.signature(module.OrbitTransition.forward).parameters)
                self.assertEqual(len(signature), 3)
                self.assertEqual(signature[0], "self")
                transition = module.OrbitTransition()
                for parameter in transition.parameters():
                    parameter.data.zero_()
                p = torch.randn(3, module.ORBIT_WIDTH)
                c = torch.randn(3, module.D_MODEL)
                self.assertTrue(torch.equal(transition(p, c), p))

    def test_t_field_is_routed_to_readout_not_orbit_or_context(self) -> None:
        module = load_source("v3.1_damped_correction")
        model = module.SUBMISSION.build_model(self.model_spec)
        first = torch.tensor([[2, 8, 9, 3, 10, 4, 8]])
        second = torch.tensor([[2, 8, 9, 3, 10, 4, 9]])

        def encoded_roles(input_ids: torch.Tensor):
            positions = torch.arange(input_ids.shape[1])
            h = model.token_embedding(input_ids) + model.position_embedding(positions)
            h = h + model.encoder_down(
                torch.nn.functional.gelu(model.encoder_up(model.encoder_norm(h)))
            )
            valid = torch.ones_like(input_ids, dtype=torch.bool)
            n_mask = module.field_mask(input_ids, valid, 2, 3)
            x_mask = module.field_mask(input_ids, valid, 3, 4)
            t_mask = module.field_mask(input_ids, valid, 4, 5)
            context = model.context_projection(module.pool(h, n_mask, model.context_score))
            orbit = model.orbit_projection(module.pool(h, x_mask, model.orbit_score))
            query = model.query_projection(module.pool(h, t_mask, model.query_score))
            return context, orbit, query

        first_context, first_orbit, first_query = encoded_roles(first)
        second_context, second_orbit, second_query = encoded_roles(second)
        self.assertTrue(torch.equal(first_context, second_context))
        self.assertTrue(torch.equal(first_orbit, second_orbit))
        self.assertFalse(torch.equal(first_query, second_query))

    def test_candidate_sources_have_no_exact_controller_or_derivative_calls(self) -> None:
        forbidden_calls = {"backward", "grad", "jacobian", "jvp", "vjp"}
        for experiment_id in CANDIDATES:
            path = EXPERIMENTS / experiment_id / "submission.py"
            tree = ast.parse(path.read_text(), filename=str(path))
            calls = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            self.assertTrue(forbidden_calls.isdisjoint(calls))
            self.assertNotIn("parse_decimal_t", names)
            self.assertNotIn("T_MARKER", names)

    def test_depth_encoding_extrapolates_without_eval_only_parameters(self) -> None:
        control = load_source("v1_tied_gelu")
        control_encoding = control.depth_encoding(65, torch.device("cpu"), torch.float32)
        for experiment_id in CANDIDATES:
            with self.subTest(experiment_id=experiment_id):
                module = load_source(experiment_id)
                model = module.SUBMISSION.build_model(self.model_spec)
                parameter_names = {name for name, _ in model.named_parameters()}
                self.assertNotIn("trajectory_position", parameter_names)
                self.assertFalse(
                    any(
                        ("trajectory" in name or "depth" in name)
                        and parameter.ndim > 0
                        and parameter.shape[0] == module.EVAL_LOOPS + 1
                        for name, parameter in model.named_parameters()
                    )
                )
                training_prefix = module.depth_encoding(
                    module.TRAIN_LOOPS + 1, torch.device("cpu"), torch.float32
                )
                evaluation_depths = module.depth_encoding(
                    module.EVAL_LOOPS + 1, torch.device("cpu"), torch.float32
                )
                self.assertEqual(
                    tuple(evaluation_depths.shape),
                    (module.EVAL_LOOPS + 1, module.D_MODEL),
                )
                self.assertTrue(torch.equal(training_prefix, evaluation_depths[: module.TRAIN_LOOPS + 1]))
                self.assertFalse(evaluation_depths.requires_grad)
                self.assertTrue(torch.isfinite(evaluation_depths).all())
                self.assertTrue(torch.equal(evaluation_depths, control_encoding))

    def test_correction_damping_is_strict_and_orbit_is_separate(self) -> None:
        corrected = CANDIDATES[1:]
        for experiment_id in corrected:
            with self.subTest(experiment_id=experiment_id):
                model = load_source(experiment_id).SUBMISSION.build_model(self.model_spec)
                rho = model.correction.correction_rho
                self.assertGreaterEqual(float(rho.detach().min()), 0.0)
                self.assertLess(float(rho.detach().max()), 1.0)
                self.assertIsNot(model.correction, model.orbit_transition)

    def test_gabor_and_refiner_are_confined_to_correction(self) -> None:
        refiner = load_source("v3.2_local_state_refiner")
        self.assertTrue(hasattr(refiner.DampedRefinedCorrection(), "refiner"))
        gabor = load_source("v3.3_gabor_corrector")
        correction_source = inspect.getsource(gabor.GaborCorrection.forward)
        orbit_source = inspect.getsource(gabor.OrbitTransition.forward)
        self.assertIn("torch.sin", correction_source)
        self.assertIn("torch.exp", correction_source)
        self.assertNotIn("torch.sin", orbit_source)
        self.assertNotIn("torch.exp", orbit_source)

    def test_picard_map_contracts_and_anderson_agrees(self) -> None:
        picard_module = load_source("v4_attractor_picard_corrector")
        anderson_module = load_source("v4.1_attractor_anderson_corrector")
        picard = picard_module.PicardCorrector()
        anderson = anderson_module.AndersonCorrector()
        anderson.load_state_dict(picard.state_dict())

        e = torch.randn(4, 64)
        proposal = torch.randn(4, 64)
        first = picard.fixed_point_map(e, proposal)
        second = picard.fixed_point_map(first, proposal)
        self.assertLess((second - first).norm(), 0.5 * (first - e).norm())

        p, h, c = torch.randn(4, 128), torch.randn(4, 7, 128), torch.randn(4, 128)
        picard_solution = picard(p, h, c, 12)
        anderson_solution = anderson(p, h, c, 8)
        self.assertTrue(torch.allclose(picard_solution, anderson_solution, atol=2e-4, rtol=2e-4))

    def test_smoothmax_excludes_padding_and_auxiliaries_are_differentiable(self) -> None:
        module = load_source("v5_sequence_smoothmax_loss")
        logits = torch.randn(2, 3, 17, requires_grad=True)
        labels = torch.tensor([[1, 2, 3], [4, 5, -100]])
        valid = labels != -100
        loss = module.sequence_smoothmax(TokenLossBatch(logits, labels, valid, None, None))
        loss.backward()
        self.assertEqual(float(logits.grad[1, 2].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[0, 0].abs().sum()), 0.0)

        composition = load_source("v5.2_composition_stability")
        model = composition.SUBMISSION.build_model(self.model_spec)
        ids = torch.randint(0, 17, (2, 8))
        logits, auxiliary = model(ids, torch.ones_like(ids, dtype=torch.bool))
        self.assertEqual(
            set(auxiliary),
            {"prompt_logits", "prompt_ids", "prompt_mask", "composition_loss", "norm_growth"},
        )
        batch = TokenLossBatch(
            logits[:, :3].float(),
            torch.randint(0, 17, (2, 3)),
            torch.ones(2, 3, dtype=torch.bool),
            None,
            auxiliary,
        )
        full_loss = composition.training_loss(batch)
        self.assertTrue(full_loss.isfinite() and full_loss.requires_grad)

    def test_exact_t_parser_stops_at_the_next_field(self) -> None:
        research = load_source(RESEARCH, "research.py")
        separate = torch.tensor([[2, 8, 3, 9, 4, 8, 9]])
        causal = torch.tensor([[1, 2, 8, 3, 9, 4, 10, 5, 15, 16, 6]])
        self.assertEqual(research.parse_decimal_t(separate, None).tolist(), [12])
        self.assertEqual(research.parse_decimal_t(causal, None).tolist(), [3])

    def test_all_eval_rollouts_remain_finite_at_64_steps(self) -> None:
        ids = torch.randint(0, 17, (1, 6))
        mask = torch.ones_like(ids, dtype=torch.bool)
        for experiment_id in CANDIDATES:
            with self.subTest(experiment_id=experiment_id), torch.no_grad():
                model = load_source(experiment_id).SUBMISSION.build_model(self.model_spec).eval()
                logits, _ = model(ids, mask)
                self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
