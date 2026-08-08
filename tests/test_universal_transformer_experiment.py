from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from benchmark.validation import validate_optimizer, validate_submission
from experiments._infra.registry import MATRICES, get_experiment
from experiments._infra.validate import validate_experiment


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "v1.4_universal_transformer"


def load_module(experiment_id: str):
    path = ROOT / "experiments" / experiment_id / "submission.py"
    module_spec = importlib.util.spec_from_file_location(
        "test_" + experiment_id.replace(".", "_"),
        path,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
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


class UniversalTransformerExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module(EXPERIMENT_ID)
        cls.model_spec = ModelSpec(17, 13, 500_000_000)
        cls.optimizer_spec = OptimizerSpec(1.0, "cpu")

    def test_registry_metadata_and_candidate_validation_are_integrated(self) -> None:
        experiment = get_experiment(EXPERIMENT_ID)
        self.assertEqual(experiment.parent, "v0_baseline_adamw")
        self.assertEqual(experiment.family, "recurrence")
        self.assertTrue(experiment.is_candidate)
        self.assertIn(EXPERIMENT_ID, MATRICES["candidate-safe"])
        self.assertIn(EXPERIMENT_ID, MATRICES["e1-full"])
        result = validate_experiment(experiment, official=True)
        self.assertTrue(result.valid, result.errors)

        metadata = json.loads(
            (experiment.directory() / "experiment.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["architecture"]["token_state"], "full_sequence")
        self.assertEqual(metadata["architecture"]["halting"], "fixed_depth_no_act")

    def test_one_full_attention_block_is_tied_and_reapplied_to_token_states(self) -> None:
        model = self.module.SUBMISSION.build_model(self.model_spec).train()
        input_ids, attention_mask = prompt()
        block_inputs: list[torch.Tensor] = []
        block_outputs: list[torch.Tensor] = []
        block_ids: list[int] = []
        attention_shapes: list[tuple[int, ...]] = []

        def record_input(block, args) -> None:
            block_ids.append(id(block))
            block_inputs.append(args[0].detach().clone())

        def record_output(_block, _args, output) -> None:
            block_outputs.append(output.detach().clone())

        original_attention = F.scaled_dot_product_attention

        def record_attention(query, key, value, *args, **kwargs):
            attention_shapes.append(tuple(query.shape))
            return original_attention(query, key, value, *args, **kwargs)

        input_hook = model.recurrent_block.register_forward_pre_hook(record_input)
        output_hook = model.recurrent_block.register_forward_hook(record_output)
        try:
            with mock.patch.object(
                self.module.F,
                "scaled_dot_product_attention",
                side_effect=record_attention,
            ):
                model(input_ids, attention_mask=attention_mask)
        finally:
            input_hook.remove()
            output_hook.remove()

        self.assertEqual(len(block_inputs), self.module.TRAIN_STEPS)
        self.assertEqual(len(set(block_ids)), 1)
        self.assertEqual(
            len(
                [
                    child
                    for child in model.modules()
                    if isinstance(child, self.module.UniversalBlock)
                ]
            ),
            1,
        )
        self.assertEqual(
            attention_shapes,
            [(2, self.module.NUM_HEADS, 8, 32)] * self.module.TRAIN_STEPS,
        )

        encodings = self.module.time_encoding(
            self.module.TRAIN_STEPS,
            torch.device("cpu"),
            torch.float32,
        )
        positions = torch.arange(input_ids.shape[1])
        initial_states = model.token_embedding(input_ids) + model.position_embedding(
            positions
        )
        self.assertTrue(
            torch.allclose(
                block_inputs[0],
                initial_states + encodings[0].view(1, 1, -1),
            )
        )
        for step in range(1, self.module.TRAIN_STEPS):
            self.assertTrue(
                torch.allclose(
                    block_inputs[step],
                    block_outputs[step - 1] + encodings[step].view(1, 1, -1),
                )
            )

    def test_time_encoding_is_fixed_and_extrapolates_past_training(self) -> None:
        training = self.module.time_encoding(
            self.module.TRAIN_STEPS,
            torch.device("cpu"),
            torch.float32,
        )
        evaluation = self.module.time_encoding(
            self.module.EVAL_STEPS,
            torch.device("cpu"),
            torch.float32,
        )
        repeated = self.module.time_encoding(
            self.module.EVAL_STEPS,
            torch.device("cpu"),
            torch.float32,
        )
        self.assertEqual(tuple(evaluation.shape), (64, self.module.D_MODEL))
        self.assertTrue(torch.equal(training, evaluation[:4]))
        self.assertTrue(torch.equal(evaluation, repeated))
        self.assertFalse(evaluation.requires_grad)
        self.assertTrue(torch.isfinite(evaluation).all().item())
        self.assertFalse(torch.equal(evaluation[3], evaluation[63]))

        model = self.module.SUBMISSION.build_model(self.model_spec)
        self.assertFalse(
            any(
                "time" in name or "depth" in name
                for name, _parameter in model.named_parameters()
            )
        )

    def test_training_and_evaluation_use_four_and_sixty_four_steps(self) -> None:
        model = self.module.SUBMISSION.build_model(self.model_spec)
        input_ids, attention_mask = prompt()

        def call_count(training: bool) -> int:
            calls = [0]

            def record(_block, _args, _output) -> None:
                calls[0] += 1

            hook = model.recurrent_block.register_forward_hook(record)
            try:
                model.train(training)
                with torch.no_grad():
                    model(input_ids, attention_mask=attention_mask)
            finally:
                hook.remove()
            return calls[0]

        self.assertEqual(model.num_loops, 4)
        self.assertEqual(model.train_steps, 4)
        self.assertEqual(model.eval_steps, 64)
        self.assertEqual(call_count(training=True), 4)
        self.assertEqual(call_count(training=False), 64)

    def test_parameter_match_optimizer_coverage_and_finite_loss(self) -> None:
        model = self.module.SUBMISSION.build_model(self.model_spec)
        baseline = load_module("v0_baseline_adamw").SUBMISSION.build_model(
            self.model_spec
        )
        self.assertEqual(count_model_state_elements(model), 201_984)
        self.assertEqual(
            count_model_state_elements(model),
            count_model_state_elements(baseline),
        )
        validate_submission(self.module.SUBMISSION)
        bundle = self.module.SUBMISSION.build_optimizer(model, self.optimizer_spec)
        validate_optimizer(bundle, model, torch.device("cpu"))

        input_ids, attention_mask = prompt()
        model.train()
        logits, auxiliary = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(tuple(logits.shape), (2, 8, 17))
        self.assertIsNone(auxiliary)
        loss = logits.float().square().mean()
        self.assertTrue(loss.isfinite().item())
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        bundle.optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(input_ids, attention_mask=attention_mask)
        self.assertTrue(torch.isfinite(eval_logits).all().item())


if __name__ == "__main__":
    unittest.main()
