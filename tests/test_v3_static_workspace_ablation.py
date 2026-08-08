from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from benchmark.validation import validate_optimizer, validate_submission
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
FULL_PATH = ROOT / "experiments/v3_factored_orbit_workspace/submission.py"
ABLATION_DIR = ROOT / "experiments/v3.6_static_workspace_ablation"
ABLATION_PATH = ABLATION_DIR / "submission.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def structured_prompt() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [2, 8, 9, 3, 10, 11, 4, 8, 9, 0, 0, 0],
            [2, 9, 10, 3, 11, 12, 4, 10, 11, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    return input_ids, input_ids.ne(0)


class StaticWorkspaceAblationTests(unittest.TestCase):
    model_spec = ModelSpec(17, 12, 500_000_000)
    optimizer_spec = OptimizerSpec(1.0, "cpu")

    @classmethod
    def setUpClass(cls) -> None:
        cls.full = load_module(FULL_PATH, "test_v3_static_workspace_full")
        cls.ablation = load_module(
            ABLATION_PATH, "test_v3_static_workspace_ablation"
        )

    def matched_models(self):
        torch.manual_seed(741)
        full = self.full.SUBMISSION.build_model(self.model_spec)
        ablation = self.ablation.SUBMISSION.build_model(self.model_spec)
        ablation.load_state_dict(full.state_dict())
        return full, ablation

    def test_metadata_and_standalone_candidate_contract(self) -> None:
        metadata = json.loads((ABLATION_DIR / "experiment.json").read_text())
        self.assertEqual(metadata["id"], "v3.6_static_workspace_ablation")
        self.assertEqual(metadata["version"], "v3.6")
        self.assertEqual(metadata["parent"], "v3_factored_orbit_workspace")
        self.assertEqual(metadata["eligibility"], "candidate-safe")
        self.assertEqual(metadata["entrypoint"], "submission.py")
        self.assertEqual(metadata["family"], "factored-state-ablation")
        source = ABLATION_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            validate_submission_source("submission.py", source, 256 * 1024),
            "submission.py",
        )
        validate_submission(self.ablation.SUBMISSION)

    def test_source_changes_only_workspace_residual_coefficient(self) -> None:
        full_source = FULL_PATH.read_text(encoding="utf-8")
        expected = full_source.replace(
            "workspace = workspace + 0.25 * torch.tanh(workspace_update)",
            "workspace = workspace + 0.0 * torch.tanh(workspace_update)",
        )
        self.assertNotEqual(expected, full_source)
        self.assertEqual(ABLATION_PATH.read_text(encoding="utf-8"), expected)

    def test_state_and_optimizer_contracts_exactly_match_v3(self) -> None:
        full, ablation = self.matched_models()
        self.assertEqual(tuple(full.state_dict()), tuple(ablation.state_dict()))
        self.assertEqual(
            {name: tuple(value.shape) for name, value in full.state_dict().items()},
            {name: tuple(value.shape) for name, value in ablation.state_dict().items()},
        )
        self.assertEqual(
            count_model_state_elements(full), count_model_state_elements(ablation)
        )
        bundle = self.ablation.SUBMISSION.build_optimizer(
            ablation, self.optimizer_spec
        )
        validate_optimizer(bundle, ablation, torch.device("cpu"))

    def test_finite_training_update_and_sixty_four_step_evaluation(self) -> None:
        input_ids, mask = structured_prompt()
        model = self.ablation.SUBMISSION.build_model(self.model_spec)
        bundle = self.ablation.SUBMISSION.build_optimizer(model, self.optimizer_spec)
        model.train()
        logits, auxiliary = model(input_ids, attention_mask=mask)
        self.assertEqual(tuple(logits.shape), (2, 12, 17))
        self.assertIsNone(auxiliary)
        self.assertTrue(torch.isfinite(logits).all().item())
        logits.square().mean().backward()
        bundle.optimizer.step()
        validate_optimizer(bundle, model, torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(input_ids, attention_mask=mask)
        self.assertTrue(torch.isfinite(eval_logits).all().item())

    def test_workspace_paths_run_but_only_full_v3_depends_on_them(self) -> None:
        input_ids, mask = structured_prompt()
        full, ablation = self.matched_models()
        full.eval()
        ablation.eval()
        calls = {
            "orbit_transition": 0,
            "workspace_local": 0,
            "orbit_to_workspace": 0,
            "context_to_workspace": 0,
        }

        def record(name: str):
            def hook(_module, _inputs, _output) -> None:
                calls[name] += 1

            return hook

        hooks = [
            getattr(ablation, name).register_forward_hook(record(name))
            for name in calls
        ]
        try:
            with torch.no_grad():
                full_before, _ = full(input_ids, attention_mask=mask)
                ablation_before, _ = ablation(input_ids, attention_mask=mask)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(calls, {name: 64 for name in calls})

        def perturb_workspace_update(model: torch.nn.Module) -> None:
            with torch.no_grad():
                model.workspace_local.weight.zero_()
                model.workspace_local.bias.fill_(0.75)
                model.orbit_to_workspace.weight.zero_()
                model.context_to_workspace.weight.zero_()

        perturb_workspace_update(full)
        perturb_workspace_update(ablation)
        with torch.no_grad():
            full_after, _ = full(input_ids, attention_mask=mask)
            ablation_after, _ = ablation(input_ids, attention_mask=mask)

        self.assertTrue(torch.equal(ablation_before, ablation_after))
        self.assertFalse(torch.equal(full_before, full_after))

    def test_disabled_workspace_gradients_are_zero_while_orbit_and_readout_are_live(self) -> None:
        input_ids, mask = structured_prompt()
        model = self.ablation.SUBMISSION.build_model(self.model_spec).train()
        logits, _ = model(input_ids, attention_mask=mask)
        logits.square().mean().backward()

        disabled = (
            model.workspace_norm,
            model.workspace_local,
            model.orbit_to_workspace,
            model.context_to_workspace,
        )
        for module in disabled:
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(float(parameter.grad.abs().max()), 0.0)

        for module in (
            model.orbit_transition,
            model.trajectory_key,
            model.query_projection,
        ):
            gradient_norm = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient_norm, 0.0)

    def test_train_and_eval_loop_counts_remain_four_and_sixty_four(self) -> None:
        input_ids, mask = structured_prompt()
        model = self.ablation.SUBMISSION.build_model(self.model_spec)

        def count_calls(training: bool) -> tuple[int, int]:
            counts = [0, 0]
            hooks = [
                model.orbit_transition.register_forward_hook(
                    lambda _module, _inputs, _output: counts.__setitem__(
                        0, counts[0] + 1
                    )
                ),
                model.workspace_local.register_forward_hook(
                    lambda _module, _inputs, _output: counts.__setitem__(
                        1, counts[1] + 1
                    )
                ),
            ]
            model.train(training)
            try:
                with torch.no_grad():
                    model(input_ids, attention_mask=mask)
            finally:
                for hook in hooks:
                    hook.remove()
            return counts[0], counts[1]

        self.assertEqual(count_calls(True), (4, 4))
        self.assertEqual(count_calls(False), (64, 64))


if __name__ == "__main__":
    unittest.main()
