"""Four-layer untied GELU control with learned trajectory selection."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 4
EVAL_LOOPS = 64
EVAL_CYCLES = EVAL_LOOPS // NUM_LAYERS
N_TOKEN, X_TOKEN, T_TOKEN, ANS_TOKEN, DIGIT_OFFSET = 2, 3, 4, 5, 7


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


class EncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


def field_masks(input_ids: Tensor, attention_mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    valid = input_ids.ne(0) if attention_mask is None else attention_mask.bool()
    digits = input_ids.ge(DIGIT_OFFSET) & input_ids.lt(DIGIT_OFFSET + 10) & valid
    seen_n = input_ids.eq(N_TOKEN).cumsum(1).gt(0)
    seen_x = input_ids.eq(X_TOKEN).cumsum(1).gt(0)
    seen_t = input_ids.eq(T_TOKEN).cumsum(1).gt(0)
    seen_answer = input_ids.eq(ANS_TOKEN).cumsum(1).gt(0)
    return digits & seen_n & ~seen_x, digits & seen_x & ~seen_t, digits & seen_t & ~seen_answer


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def depth_encoding(length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Fixed sinusoidal features that extrapolate to every rollout depth."""

    depths = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(-1)
    frequencies = torch.exp(
        torch.arange(0, D_MODEL, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / D_MODEL)
    )
    angles = depths * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1).to(dtype)


class GELULayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)
        self.residual_scale = nn.Parameter(torch.empty(()))
        nn.init.uniform_(self.residual_scale, 0.2, 0.3)

    def forward(self, state: Tensor) -> Tensor:
        update = self.down(F.gelu(self.up(self.norm(state))))
        return state + self.residual_scale.tanh() * update


class TrajectoryReadout(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.state_norm = RMSNorm(D_MODEL)
        self.output_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, trajectory: Tensor, query: Tensor, token_states: Tensor) -> Tensor:
        depth_features = depth_encoding(
            trajectory.shape[1], trajectory.device, trajectory.dtype
        )
        keys = self.state_norm(trajectory) + depth_features
        scores = (keys * query.unsqueeze(1)).sum(-1) / math.sqrt(D_MODEL)
        selected = (scores.softmax(1).unsqueeze(-1) * trajectory).sum(1)
        return self.head(self.output_norm(token_states + selected.unsqueeze(1)))


class Model(nn.Module):
    num_loops = NUM_LAYERS
    train_loops = NUM_LAYERS
    eval_loops = EVAL_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.encoder = EncoderBlock()
        self.initial_projection = nn.Linear(2 * D_MODEL, D_MODEL)
        self.query_projection = nn.Linear(D_MODEL, D_MODEL)
        self.transitions = nn.ModuleList(GELULayer() for _ in range(NUM_LAYERS))
        self.readout = TrajectoryReadout(spec.vocab_size)
        self.readout.head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        token_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        token_states = self.encoder(token_states, attention_mask)
        n_mask, x_mask, t_mask = field_masks(input_ids, attention_mask)
        n_state = masked_mean(token_states, n_mask)
        x_state = masked_mean(token_states, x_mask)
        query = self.query_projection(masked_mean(token_states, t_mask))
        state = self.initial_projection(torch.cat((n_state, x_state), dim=-1))
        trajectory = [state]
        cycles = 1 if self.training else EVAL_CYCLES
        for _ in range(cycles):
            for transition in self.transitions:
                state = transition(state)
                trajectory.append(state)
        return self.readout(torch.stack(trajectory, 1), query, token_states), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    ))


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer)
