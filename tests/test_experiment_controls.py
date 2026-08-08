from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from benchmark.validation import validate_optimizer, validate_submission
from experiments._infra.registry import get_experiment
from experiments._infra.validate import validate_experiment
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
CANDIDATES = (
    "v0_baseline_adamw",
    "v1_tied_gelu",
    "v1.1_untied_gelu_control",
    "v1.2_persistent_context",
    "v2_real_sine",
    "v2.1_paired_complex_sine",
    "v2.2_real_bilinear",
    "v2.3_holomorphic_bilinear",
    "v2.4_conjugate_bilinear_control",
)
RESEARCH = "v1.3_exact_t_diagnostic"
RECURRENT_CANDIDATES = CANDIDATES[1:]
EXPECTED_PARENTS = {
    "v0_baseline_adamw": None,
    "v1_tied_gelu": "v0_baseline_adamw",
    "v1.1_untied_gelu_control": "v1_tied_gelu",
    "v1.2_persistent_context": "v1_tied_gelu",
    "v1.3_exact_t_diagnostic": "v1.2_persistent_context",
    "v2_real_sine": "v1.2_persistent_context",
    "v2.1_paired_complex_sine": "v1.2_persistent_context",
    "v2.2_real_bilinear": "v1.2_persistent_context",
    "v2.3_holomorphic_bilinear": "v1.2_persistent_context",
    "v2.4_conjugate_bilinear_control": "v2.3_holomorphic_bilinear",
}


def load_module(experiment_id: str, filename: str = "submission.py"):
    path = EXPERIMENTS / experiment_id / filename
    module_name = "test_" + experiment_id.replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def structured_prompt() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [2, 8, 10, 3, 9, 4, 8, 0, 0, 0],
            [2, 8, 11, 10, 3, 12, 4, 13, 11, 0],
        ],
        dtype=torch.long,
    )
    return input_ids, input_ids.ne(0)


class ExperimentControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_spec = ModelSpec(17, 10, 500_000_000)
        cls.optimizer_spec = OptimizerSpec(1.0, "cpu")

    def test_metadata_and_directory_contract(self) -> None:
        for experiment_id, parent in EXPECTED_PARENTS.items():
            with self.subTest(experiment_id=experiment_id):
                directory = EXPERIMENTS / experiment_id
                metadata = json.loads((directory / "experiment.json").read_text())
                self.assertEqual(metadata["id"], experiment_id)
                self.assertEqual(metadata["parent"], parent)
                self.assertTrue(metadata["version"].startswith("v"))
                for key in ("title", "family", "description"):
                    self.assertTrue(metadata[key])
                if experiment_id == RESEARCH:
                    self.assertEqual(metadata["eligibility"], "research-only")
                    self.assertEqual(metadata["entrypoint"], "research.py")
                    self.assertTrue((directory / "BLOCKED_FROM_SUBMISSION").is_file())
                    self.assertFalse((directory / "submission.py").exists())
                else:
                    self.assertEqual(metadata["eligibility"], "candidate-safe")
                    self.assertEqual(metadata["entrypoint"], "submission.py")
                    self.assertTrue((directory / "submission.py").is_file())
                self.assertTrue((directory / "README.md").is_file())

    def test_shared_registry_accepts_every_control_experiment(self) -> None:
        for experiment_id in (*CANDIDATES, RESEARCH):
            with self.subTest(experiment_id=experiment_id):
                result = validate_experiment(get_experiment(experiment_id))
                self.assertTrue(result.valid, result.errors)

    def test_candidate_sources_are_standalone_and_policy_clean(self) -> None:
        prohibited = ("pow(", "torch.autograd", ".backward(", "discrete_log")
        for experiment_id in CANDIDATES:
            with self.subTest(experiment_id=experiment_id):
                path = EXPERIMENTS / experiment_id / "submission.py"
                source = path.read_text(encoding="utf-8")
                self.assertEqual(
                    validate_submission_source(path.name, source, 256 * 1024),
                    "submission.py",
                )
                ast.parse(source)
                for token in prohibited:
                    self.assertNotIn(token, source)
                module = load_module(experiment_id)
                validate_submission(module.SUBMISSION)

    def test_candidate_models_train_and_run_sixty_four_step_evaluation(self) -> None:
        input_ids, mask = structured_prompt()
        for experiment_id in CANDIDATES:
            with self.subTest(experiment_id=experiment_id):
                module = load_module(experiment_id)
                model = module.SUBMISSION.build_model(self.model_spec)
                self.assertLessEqual(
                    count_model_state_elements(model),
                    self.model_spec.maximum_model_state_elements,
                )
                bundle = module.SUBMISSION.build_optimizer(model, self.optimizer_spec)
                validate_optimizer(bundle, model, torch.device("cpu"))
                model.train()
                train_logits, auxiliary = model(input_ids, attention_mask=mask)
                self.assertEqual(tuple(train_logits.shape), (2, 10, 17))
                self.assertIsNone(auxiliary)
                self.assertTrue(torch.isfinite(train_logits).all().item())
                train_logits.square().mean().backward()
                bundle.optimizer.step()
                model.eval()
                with torch.no_grad():
                    eval_logits, _ = model(input_ids, attention_mask=mask)
                self.assertTrue(torch.isfinite(eval_logits).all().item())

    def test_candidate_transition_interfaces_exclude_controller_inputs(self) -> None:
        for experiment_id in CANDIDATES[1:]:
            with self.subTest(experiment_id=experiment_id):
                module = load_module(experiment_id)
                transition_classes = [
                    value
                    for name, value in vars(module).items()
                    if inspect.isclass(value) and ("Transition" in name or name == "GELULayer")
                ]
                self.assertTrue(transition_classes)
                for transition_class in transition_classes:
                    parameters = list(inspect.signature(transition_class.forward).parameters)
                    self.assertEqual(parameters[0], "self")
                    self.assertIn(parameters[1:], (["state"], ["state", "context"]))
                    self.assertNotIn("input_ids", parameters)
                    self.assertNotIn("step", parameters)
                    self.assertNotIn("t", parameters)

    def test_trajectory_depth_encoding_has_no_trainable_eval_only_rows(self) -> None:
        for experiment_id in RECURRENT_CANDIDATES:
            with self.subTest(experiment_id=experiment_id):
                module = load_module(experiment_id)
                model = module.SUBMISSION.build_model(self.model_spec)
                self.assertFalse(hasattr(model.readout, "position"))
                self.assertFalse(
                    any("position" in name for name, _ in model.readout.named_parameters())
                )
                first = module.depth_encoding(65, torch.device("cpu"), torch.float32)
                second = module.depth_encoding(65, torch.device("cpu"), torch.float32)
                self.assertEqual(tuple(first.shape), (65, module.D_MODEL))
                self.assertTrue(torch.equal(first, second))
                self.assertFalse(torch.equal(first[4], first[64]))

    def test_tied_and_period_four_controls_make_equal_eval_calls(self) -> None:
        input_ids, mask = structured_prompt()
        tied = load_module("v1_tied_gelu").SUBMISSION.build_model(self.model_spec)
        untied = load_module("v1.1_untied_gelu_control").SUBMISSION.build_model(
            self.model_spec
        )

        def transition_calls(model: torch.nn.Module) -> int:
            count = [0]

            def record_call(_module, _inputs, _output) -> None:
                count[0] += 1

            transitions = (
                [model.transition]
                if hasattr(model, "transition")
                else list(model.transitions)
            )
            hooks = [transition.register_forward_hook(record_call) for transition in transitions]
            try:
                model.eval()
                with torch.no_grad():
                    model(input_ids, attention_mask=mask)
            finally:
                for hook in hooks:
                    hook.remove()
            return count[0]

        self.assertEqual(tied.eval_loops, 64)
        self.assertEqual(untied.eval_loops, 64)
        self.assertEqual(transition_calls(tied), 64)
        self.assertEqual(transition_calls(untied), 64)

    def test_tied_and_untied_controls_have_the_expected_ownership(self) -> None:
        tied = load_module("v1_tied_gelu").SUBMISSION.build_model(self.model_spec)
        untied = load_module("v1.1_untied_gelu_control").SUBMISSION.build_model(self.model_spec)
        self.assertTrue(hasattr(tied, "transition"))
        self.assertEqual(len(untied.transitions), 4)
        self.assertEqual(len({id(layer) for layer in untied.transitions}), 4)
        untied_parameter_count = sum(parameter.numel() for parameter in untied.parameters())
        tied_parameter_count = sum(parameter.numel() for parameter in tied.parameters())
        self.assertGreater(untied_parameter_count, tied_parameter_count)

    def test_complex_bilinear_pair_is_parameter_matched(self) -> None:
        holomorphic = load_module("v2.3_holomorphic_bilinear")
        control = load_module("v2.4_conjugate_bilinear_control")
        holomorphic_model = holomorphic.SUBMISSION.build_model(self.model_spec)
        control_model = control.SUBMISSION.build_model(self.model_spec)
        self.assertEqual(
            count_model_state_elements(holomorphic_model),
            count_model_state_elements(control_model),
        )
        self.assertIsInstance(
            holomorphic_model.transition.core,
            holomorphic.HolomorphicBilinearCore,
        )
        self.assertIsInstance(
            control_model.transition.core,
            control.ConjugateBilinearCore,
        )
        self.assertTrue(hasattr(holomorphic_model.transition, "norm"))
        self.assertTrue(hasattr(holomorphic_model.transition, "context"))
        self.assertFalse(hasattr(holomorphic_model.transition.core, "norm"))
        self.assertFalse(hasattr(holomorphic_model.transition.core, "context"))
        holomorphic_source = (
            EXPERIMENTS / "v2.3_holomorphic_bilinear" / "submission.py"
        ).read_text()
        control_source = (
            EXPERIMENTS / "v2.4_conjugate_bilinear_control" / "submission.py"
        ).read_text()
        self.assertNotIn("self.right(real, -imag)", holomorphic_source)
        self.assertIn("self.right(real, -imag)", control_source)

    def test_strict_complex_cores_have_expected_wirtinger_behavior(self) -> None:
        torch.manual_seed(731)
        holomorphic = load_module("v2.3_holomorphic_bilinear")
        control = load_module("v2.4_conjugate_bilinear_control")
        holomorphic_core = holomorphic.HolomorphicBilinearCore().double()
        control_core = control.ConjugateBilinearCore().double()
        control_core.load_state_dict(holomorphic_core.state_dict())

        features = 0.2 * torch.randn(2, holomorphic.D_MODEL, dtype=torch.double)
        direction = torch.randn(2, holomorphic.COMPLEX_WIDTH, dtype=torch.double)
        zeros = torch.zeros_like(direction)
        real_tangent = torch.cat((direction, zeros), dim=-1)
        imag_tangent = torch.cat((zeros, direction), dim=-1)

        def cauchy_riemann_residual(core: torch.nn.Module) -> torch.Tensor:
            _, derivative_real = torch.autograd.functional.jvp(
                core, (features,), (real_tangent,)
            )
            _, derivative_imag = torch.autograd.functional.jvp(
                core, (features,), (imag_tangent,)
            )
            real_part, imag_part = derivative_real.chunk(2, dim=-1)
            multiplication_by_i = torch.cat((-imag_part, real_part), dim=-1)
            return derivative_imag - multiplication_by_i

        holomorphic_residual = cauchy_riemann_residual(holomorphic_core)
        conjugate_residual = cauchy_riemann_residual(control_core)
        self.assertLess(holomorphic_residual.abs().max().item(), 1e-9)
        self.assertGreater(conjugate_residual.norm().item(), 1e-4)

    def test_exact_t_parser_is_isolated_and_parses_decimal_depth(self) -> None:
        module = load_module(RESEARCH, "research.py")
        input_ids, mask = structured_prompt()
        _, _, t_mask = module.field_masks(input_ids, mask)
        self.assertEqual(module.parse_exact_t(input_ids, t_mask).tolist(), [1, 64])
        causal_ids = torch.tensor([[1, 2, 8, 10, 3, 9, 4, 8, 5, 16, 6]])
        _, _, causal_t_mask = module.field_masks(causal_ids, causal_ids.ne(0))
        self.assertEqual(module.parse_exact_t(causal_ids, causal_t_mask).tolist(), [1])
        model = module.SUBMISSION.build_model(self.model_spec)
        logits, auxiliary = model(input_ids, attention_mask=mask)
        self.assertEqual(tuple(logits.shape), (2, 10, 17))
        self.assertEqual(auxiliary["exact_t"].tolist(), [1, 64])
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
