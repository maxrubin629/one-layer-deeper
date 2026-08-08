"""Fixed-depth Universal Transformer with a shared full-sequence block."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


D_MODEL = 128
NUM_HEADS = 4
TRAIN_STEPS = 4
EVAL_STEPS = 64


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


def time_encoding(
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return fixed sinusoidal features for recurrent steps 1 through N."""

    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    steps = torch.arange(
        1,
        num_steps + 1,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(-1)
    frequencies = torch.exp(
        torch.arange(0, D_MODEL, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / D_MODEL)
    )
    angles = steps * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1).to(dtype)


class UniversalBlock(nn.Module):
    """One Transformer block shared across every recurrent depth step."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, states: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = states
        states = self.attention_norm(states)
        batch, length, _ = states.shape
        query, key, value = self.qkv(states).chunk(3, dim=-1)
        query = query.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        key = key.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        value = value.view(batch, length, NUM_HEADS, -1).transpose(1, 2)

        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=states.device, dtype=torch.bool)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch, length, D_MODEL)
        )
        states = residual + self.out(attended)
        return states + self.down(F.gelu(self.up(self.mixer_norm(states))))


class Model(nn.Module):
    num_loops = TRAIN_STEPS
    train_steps = TRAIN_STEPS
    eval_steps = EVAL_STEPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.recurrent_block = UniversalBlock()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        states = self.token_embedding(input_ids) + self.position_embedding(positions)
        num_steps = self.train_steps if self.training else self.eval_steps
        step_features = time_encoding(num_steps, states.device, states.dtype)
        for step_feature in step_features.unbind(0):
            states = self.recurrent_block(
                states + step_feature.view(1, 1, D_MODEL),
                attention_mask,
            )
        return self.head(self.final_norm(states)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer)
