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
BASELINE_ID = "v3_factored_orbit_workspace"
ABLATION_ID = "v3.7_last_state_readout_ablation"


def load_module(experiment_id: str):
    path = ROOT / "experiments" / experiment_id / "submission.py"
    spec = importlib.util.spec_from_file_location(
        "test_" + experiment_id.replace(".", "_"), path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def structured_prompt() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [2, 8, 10, 3, 9, 4, 8, 10],
            [2, 8, 11, 3, 12, 4, 9, 11],
        ],
        dtype=torch.long,
    )
    return input_ids, input_ids.ne(0)


class LastStateReadoutAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(3701)
        cls.baseline_module = load_module(BASELINE_ID)
        cls.ablation_module = load_module(ABLATION_ID)
        cls.model_spec = ModelSpec(17, 12, 500_000_000)
        cls.optimizer_spec = OptimizerSpec(1.0, "cpu")

    def matched_models(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        baseline = self.baseline_module.SUBMISSION.build_model(self.model_spec)
        ablation = self.ablation_module.SUBMISSION.build_model(self.model_spec)
        ablation.load_state_dict(baseline.state_dict(), strict=True)
        return baseline, ablation

    def test_metadata_source_and_state_are_matched_to_v3(self) -> None:
        directory = ROOT / "experiments" / ABLATION_ID
        metadata = json.loads(
            (directory / "experiment.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["version"], "v3.7")
        self.assertEqual(metadata["parent"], BASELINE_ID)
        self.assertEqual(metadata["eligibility"], "candidate-safe")
        self.assertEqual(metadata["family"], "factored-state-ablation")
        source_path = directory / "submission.py"
        self.assertEqual(
            validate_submission_source(
                source_path.name, source_path.read_text(encoding="utf-8"), 256 * 1024
            ),
            "submission.py",
        )
        validate_submission(self.ablation_module.SUBMISSION)

        baseline, ablation = self.matched_models()
        self.assertEqual(
            tuple(baseline.state_dict()),
            tuple(ablation.state_dict()),
        )
        for name, baseline_value in baseline.state_dict().items():
            self.assertEqual(
                tuple(baseline_value.shape),
                tuple(ablation.state_dict()[name].shape),
                name,
            )
        self.assertEqual(
            count_model_state_elements(baseline),
            count_model_state_elements(ablation),
        )

    def test_query_parameters_affect_v3_but_not_last_state_logits(self) -> None:
        input_ids, attention_mask = structured_prompt()

        def run_and_perturb(
            model: torch.nn.Module,
            component: str,
        ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
            calls = {"query_score": 0, "query_projection": 0}
            hooks = [
                model.query_score.register_forward_hook(
                    lambda _module, _args, _output: calls.__setitem__(
                        "query_score", calls["query_score"] + 1
                    )
                ),
                model.query_projection.register_forward_hook(
                    lambda _module, _args, _output: calls.__setitem__(
                        "query_projection", calls["query_projection"] + 1
                    )
                ),
            ]
            try:
                with torch.no_grad():
                    original, _ = model(input_ids, attention_mask=attention_mask)
                    layer = getattr(model, component)
                    layer.weight.zero_()
                    layer.bias.zero_()
                    perturbed, _ = model(input_ids, attention_mask=attention_mask)
            finally:
                for hook in hooks:
                    hook.remove()
            return original, perturbed, calls

        for component in ("query_score", "query_projection"):
            with self.subTest(component=component):
                baseline, ablation = self.matched_models()
                baseline.eval()
                ablation.eval()
                baseline_original, baseline_perturbed, baseline_calls = (
                    run_and_perturb(baseline, component)
                )
                ablation_original, ablation_perturbed, ablation_calls = (
                    run_and_perturb(ablation, component)
                )

                self.assertFalse(torch.equal(baseline_original, baseline_perturbed))
                self.assertTrue(torch.equal(ablation_original, ablation_perturbed))
                expected_calls = {"query_score": 2, "query_projection": 2}
                self.assertEqual(baseline_calls, expected_calls)
                self.assertEqual(ablation_calls, expected_calls)

    def test_final_trajectory_key_controls_ablation_logits(self) -> None:
        input_ids, attention_mask = structured_prompt()
        _, ablation = self.matched_models()
        ablation.eval()
        with torch.no_grad():
            original, _ = ablation(input_ids, attention_mask=attention_mask)

        def perturb_final_key(
            _module: torch.nn.Module,
            _args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            perturbation = torch.zeros_like(output)
            perturbation[:, -1, 0] = 1.0
            return output + perturbation

        hook = ablation.trajectory_key.register_forward_hook(perturb_final_key)
        try:
            with torch.no_grad():
                perturbed, _ = ablation(input_ids, attention_mask=attention_mask)
        finally:
            hook.remove()

        self.assertFalse(torch.equal(original, perturbed))

    def test_finite_train_eval_optimizer_and_four_sixty_four_calls(self) -> None:
        input_ids, attention_mask = structured_prompt()
        model = self.ablation_module.SUBMISSION.build_model(self.model_spec)
        bundle = self.ablation_module.SUBMISSION.build_optimizer(
            model, self.optimizer_spec
        )
        validate_optimizer(bundle, model, torch.device("cpu"))

        calls = [0]

        def record_call(_module, _args, _output) -> None:
            calls[0] += 1

        hook = model.orbit_transition.register_forward_hook(record_call)
        try:
            model.train()
            train_logits, auxiliary = model(
                input_ids, attention_mask=attention_mask
            )
            self.assertEqual(calls[0], 4)
            self.assertEqual(tuple(train_logits.shape), (2, 8, 17))
            self.assertIsNone(auxiliary)
            self.assertTrue(torch.isfinite(train_logits).all().item())
            loss = train_logits.float().square().mean()
            self.assertTrue(loss.isfinite().item())
            loss.backward()
            bundle.optimizer.step()

            calls[0] = 0
            model.eval()
            with torch.no_grad():
                eval_logits, _ = model(input_ids, attention_mask=attention_mask)
            self.assertEqual(calls[0], 64)
            self.assertTrue(torch.isfinite(eval_logits).all().item())
        finally:
            hook.remove()


if __name__ == "__main__":
    unittest.main()
