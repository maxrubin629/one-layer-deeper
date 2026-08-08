from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements


ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "experiments" / "v3_factored_orbit_workspace" / "submission.py"
FEEDBACK_PATH = ROOT / "experiments" / "v3.5_workspace_feedback_control" / "submission.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [2, 8, 10, 3, 9, 4, 8, 0],
            [2, 8, 11, 3, 12, 4, 9, 0],
        ],
        dtype=torch.long,
    )
    return input_ids, input_ids.ne(0)


class WorkspaceFeedbackControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v3 = load_module(V3_PATH, "test_workspace_feedback_v3")
        cls.feedback = load_module(FEEDBACK_PATH, "test_workspace_feedback_control")
        cls.model_spec = ModelSpec(17, 12, 500_000_000)

    def paired_models(self):
        torch.manual_seed(74)
        v3_model = self.v3.SUBMISSION.build_model(self.model_spec)
        feedback_model = self.feedback.SUBMISSION.build_model(self.model_spec)
        feedback_model.load_state_dict(v3_model.state_dict(), strict=True)
        return v3_model, feedback_model

    def test_loads_trains_evaluates_and_stays_finite(self) -> None:
        model = self.feedback.SUBMISSION.build_model(self.model_spec)
        input_ids, attention_mask = prompt()

        model.train()
        logits, auxiliary = model(input_ids, attention_mask=attention_mask)
        self.assertIsNone(auxiliary)
        self.assertTrue(torch.isfinite(logits).all().item())
        loss = logits.float().square().mean()
        loss.backward()
        bundle = self.feedback.SUBMISSION.build_optimizer(
            model, OptimizerSpec(1.0, "cpu")
        )
        bundle.optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(input_ids, attention_mask=attention_mask)
        self.assertTrue(torch.isfinite(eval_logits).all().item())

    def test_state_dict_and_persistent_state_match_v3_exactly(self) -> None:
        v3_model, feedback_model = self.paired_models()
        self.assertEqual(
            tuple(v3_model.state_dict()), tuple(feedback_model.state_dict())
        )
        self.assertEqual(
            {name: tuple(value.shape) for name, value in v3_model.state_dict().items()},
            {
                name: tuple(value.shape)
                for name, value in feedback_model.state_dict().items()
            },
        )
        self.assertEqual(
            count_model_state_elements(v3_model),
            count_model_state_elements(feedback_model),
        )

    def test_zero_scale_reproduces_v3_logits_exactly(self) -> None:
        v3_model, feedback_model = self.paired_models()
        input_ids, attention_mask = prompt()

        with mock.patch.object(self.feedback, "WORKSPACE_FEEDBACK_SCALE", 0.0):
            for training in (True, False):
                v3_model.train(training)
                feedback_model.train(training)
                with torch.no_grad():
                    expected, _ = v3_model(input_ids, attention_mask=attention_mask)
                    actual, _ = feedback_model(
                        input_ids, attention_mask=attention_mask
                    )
                self.assertTrue(torch.equal(actual, expected))

    def test_feedback_summary_perturbation_changes_later_orbit_and_logits(self) -> None:
        v3_model, feedback_model = self.paired_models()
        v3_model.train()
        feedback_model.train()
        input_ids, attention_mask = prompt()

        orbit_outputs: list[torch.Tensor] = []

        def record_orbit(_module, _args, output) -> None:
            orbit_outputs.append(output.detach().clone())

        hook = feedback_model.orbit_transition.register_forward_hook(record_orbit)
        try:
            with torch.no_grad():
                baseline_logits, _ = feedback_model(
                    input_ids, attention_mask=attention_mask
                )
            baseline_orbits = tuple(orbit_outputs)
            orbit_outputs.clear()

            original_summary = self.feedback.masked_workspace_summary

            def perturbed_summary(states, valid):
                summary = original_summary(states, valid)
                return summary + torch.full_like(summary, 0.25)

            with mock.patch.object(
                self.feedback,
                "masked_workspace_summary",
                side_effect=perturbed_summary,
            ), torch.no_grad():
                perturbed_logits, _ = feedback_model(
                    input_ids, attention_mask=attention_mask
                )
            perturbed_orbits = tuple(orbit_outputs)

            with mock.patch.object(
                self.feedback,
                "masked_workspace_summary",
                side_effect=perturbed_summary,
            ), mock.patch.object(
                self.feedback, "WORKSPACE_FEEDBACK_SCALE", 0.0
            ), torch.no_grad():
                disabled_logits, _ = feedback_model(
                    input_ids, attention_mask=attention_mask
                )
                v3_logits, _ = v3_model(
                    input_ids, attention_mask=attention_mask
                )
        finally:
            hook.remove()

        self.assertEqual(len(baseline_orbits), self.feedback.TRAIN_LOOPS)
        self.assertTrue(torch.equal(baseline_orbits[0], perturbed_orbits[0]))
        self.assertTrue(
            any(
                not torch.equal(baseline, perturbed)
                for baseline, perturbed in zip(
                    baseline_orbits[1:], perturbed_orbits[1:]
                )
            )
        )
        self.assertFalse(torch.equal(baseline_logits, perturbed_logits))
        self.assertTrue(torch.equal(disabled_logits, v3_logits))

    def test_padding_does_not_influence_feedback(self) -> None:
        model = self.feedback.SUBMISSION.build_model(self.model_spec).eval()
        padded = torch.tensor([[2, 8, 10, 3, 9, 4, 8, 0, 0]], dtype=torch.long)
        changed_padding = torch.tensor(
            [[2, 8, 10, 3, 9, 4, 8, 15, 16]], dtype=torch.long
        )
        valid = torch.tensor(
            [[True, True, True, True, True, True, True, False, False]]
        )

        with torch.no_grad():
            padded_logits, _ = model(padded, attention_mask=valid)
            changed_logits, _ = model(changed_padding, attention_mask=valid)
        self.assertTrue(torch.equal(padded_logits[:, :7], changed_logits[:, :7]))

    def test_transition_call_counts_remain_four_and_sixty_four(self) -> None:
        model = self.feedback.SUBMISSION.build_model(self.model_spec)
        input_ids, attention_mask = prompt()

        def call_count(training: bool) -> int:
            calls = [0]

            def record(_module, _args, _output) -> None:
                calls[0] += 1

            hook = model.orbit_transition.register_forward_hook(record)
            try:
                model.train(training)
                with torch.no_grad():
                    model(input_ids, attention_mask=attention_mask)
            finally:
                hook.remove()
            return calls[0]

        self.assertEqual(call_count(training=True), self.feedback.TRAIN_LOOPS)
        self.assertEqual(call_count(training=False), self.feedback.EVAL_LOOPS)


if __name__ == "__main__":
    unittest.main()
