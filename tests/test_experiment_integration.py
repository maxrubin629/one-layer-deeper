from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch, count_model_state_elements
from experiments._infra.registry import EXPERIMENTS
from experiments._infra.validate import validate_experiment


ROOT = Path(__file__).resolve().parents[1]


def _load_submission(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.SUBMISSION


class ExperimentIntegrationTests(unittest.TestCase):
    def test_registry_matches_versioned_directories(self) -> None:
        expected = {spec.experiment_id for spec in EXPERIMENTS}
        actual = {
            path.name
            for path in (ROOT / "experiments").iterdir()
            if path.is_dir() and path.name.startswith("v")
        }
        self.assertEqual(actual, expected)

    def test_all_layouts_validate_and_research_is_officially_blocked(self) -> None:
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment.experiment_id):
                local = validate_experiment(experiment)
                self.assertTrue(local.valid, local.errors)
                official = validate_experiment(experiment, official=True)
                self.assertEqual(official.valid, experiment.is_candidate)

    def test_every_candidate_builds_forwards_and_covers_optimizer_parameters(self) -> None:
        model_spec = ModelSpec(
            vocab_size=17,
            max_seq_len=10,
            maximum_model_state_elements=500_000_000,
        )
        prompts = torch.tensor(
            [
                [2, 8, 9, 3, 10, 11, 4, 8, 0, 0],
                [2, 9, 10, 3, 11, 12, 4, 9, 0, 0],
            ],
            dtype=torch.long,
        )
        attention_mask = prompts.ne(0)
        for index, experiment in enumerate(EXPERIMENTS):
            if not experiment.is_candidate:
                continue
            with self.subTest(experiment=experiment.experiment_id):
                submission = _load_submission(
                    experiment.source_path(), f"experiment_submission_{index}"
                )
                model = submission.build_model(model_spec)
                self.assertLessEqual(
                    count_model_state_elements(model),
                    model_spec.maximum_model_state_elements,
                )
                model.train()
                logits, auxiliary = model(prompts, attention_mask=attention_mask)
                self.assertEqual(tuple(logits.shape), (2, 10, 17))
                self.assertTrue(torch.isfinite(logits).all().item())
                bundle = submission.build_optimizer(
                    model, OptimizerSpec(training_time_seconds=1.0, device_type="cpu")
                )
                optimizer_ids = [
                    id(parameter)
                    for group in bundle.optimizer.param_groups
                    for parameter in group["params"]
                ]
                trainable_ids = [
                    id(parameter) for parameter in model.parameters() if parameter.requires_grad
                ]
                self.assertCountEqual(optimizer_ids, trainable_ids)
                self.assertEqual(len(optimizer_ids), len(set(optimizer_ids)))
                if submission.token_training_loss is not None:
                    token_logits = logits[:, -2:, :]
                    labels = torch.tensor([[8, 9], [9, 10]], dtype=torch.long)
                    loss = submission.token_training_loss(
                        TokenLossBatch(
                            logits=token_logits,
                            labels=labels,
                            valid_mask=torch.ones_like(labels, dtype=torch.bool),
                            target_positions=torch.tensor([[8, 9], [8, 9]]),
                            auxiliary=auxiliary,
                        )
                    )
                    self.assertEqual(loss.ndim, 0)
                    self.assertTrue(torch.isfinite(loss).item())
                    self.assertTrue(loss.requires_grad)


if __name__ == "__main__":
    unittest.main()
