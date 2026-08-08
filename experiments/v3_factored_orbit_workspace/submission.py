"""Factored persistent-orbit and token-workspace candidate."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


D_MODEL = 128
ORBIT_WIDTH = 128
TRAIN_LOOPS = 4
EVAL_LOOPS = 64


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: Tensor) -> Tensor:
        return F.rms_norm(value, (value.shape[-1],), self.weight)


def masked_pool(states: Tensor, mask: Tensor | None, scores: nn.Linear) -> Tensor:
    weights = scores(states).squeeze(-1)
    if mask is not None:
        weights = weights.masked_fill(~mask.to(dtype=torch.bool), -torch.inf)
    return torch.einsum("bs,bsd->bd", weights.softmax(dim=-1), states)


def field_mask(input_ids: Tensor, valid: Tensor, marker: int, stop_marker: int) -> Tensor:
    started = (input_ids == marker).cumsum(dim=1) > 0
    stopped = (input_ids == stop_marker).cumsum(dim=1) > 0
    routed = started & ~stopped & (input_ids != marker) & valid
    return torch.where(routed.any(dim=1, keepdim=True), routed, valid)


def depth_encoding(length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    depth = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, D_MODEL, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / D_MODEL))
    angles = depth * frequencies[None, :]
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1).to(dtype)


class OrbitTransition(nn.Module):
    """Autonomous learned bilinear residual law; there is no global damping."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(ORBIT_WIDTH)
        self.left = nn.Linear(ORBIT_WIDTH, ORBIT_WIDTH)
        self.right = nn.Linear(ORBIT_WIDTH, ORBIT_WIDTH)
        self.left_context = nn.Linear(D_MODEL, ORBIT_WIDTH, bias=False)
        self.right_context = nn.Linear(D_MODEL, ORBIT_WIDTH, bias=False)
        self.out = nn.Linear(ORBIT_WIDTH, ORBIT_WIDTH)

    def forward(self, orbit: Tensor, context: Tensor) -> Tensor:
        normalized = self.norm(orbit)
        left = self.left(normalized) + self.left_context(context)
        right = self.right(normalized) + self.right_context(context)
        update = self.out(torch.tanh(left * right / math.sqrt(ORBIT_WIDTH)))
        return orbit + 0.25 * update


class Model(nn.Module):
    num_loops = EVAL_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.encoder_norm = RMSNorm(D_MODEL)
        self.encoder_up = nn.Linear(D_MODEL, 2 * D_MODEL)
        self.encoder_down = nn.Linear(2 * D_MODEL, D_MODEL)
        self.context_score = nn.Linear(D_MODEL, 1)
        self.orbit_score = nn.Linear(D_MODEL, 1)
        self.query_score = nn.Linear(D_MODEL, 1)
        self.context_projection = nn.Linear(D_MODEL, D_MODEL)
        self.orbit_projection = nn.Linear(D_MODEL, ORBIT_WIDTH)
        self.query_projection = nn.Linear(D_MODEL, D_MODEL)
        self.orbit_transition = OrbitTransition()
        self.workspace_norm = RMSNorm(D_MODEL)
        self.workspace_local = nn.Linear(D_MODEL, D_MODEL)
        self.orbit_to_workspace = nn.Linear(ORBIT_WIDTH, D_MODEL, bias=False)
        self.context_to_workspace = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.trajectory_key = nn.Linear(ORBIT_WIDTH, D_MODEL)
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        workspace = self.token_embedding(input_ids) + self.position_embedding(positions)
        workspace = workspace + self.encoder_down(F.gelu(self.encoder_up(self.encoder_norm(workspace))))
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        context_mask = field_mask(input_ids, valid, 2, 3)
        orbit_mask = field_mask(input_ids, valid, 3, 4)
        query_mask = field_mask(input_ids, valid, 4, 5)
        context = self.context_projection(masked_pool(workspace, context_mask, self.context_score))
        orbit = self.orbit_projection(masked_pool(workspace, orbit_mask, self.orbit_score))
        query = self.query_projection(masked_pool(workspace, query_mask, self.query_score))
        trajectory = [orbit]
        loop_count = TRAIN_LOOPS if self.training else EVAL_LOOPS
        for _ in range(loop_count):
            orbit = self.orbit_transition(orbit, context)
            workspace_update = (
                self.workspace_local(self.workspace_norm(workspace))
                + self.orbit_to_workspace(orbit)[:, None, :]
                + self.context_to_workspace(context)[:, None, :]
            )
            workspace = workspace + 0.25 * torch.tanh(workspace_update)
            trajectory.append(orbit)
        states = torch.stack(trajectory, dim=1)
        keys = self.trajectory_key(states)
        keys = keys + depth_encoding(loop_count + 1, keys.device, keys.dtype)[None]
        weights = torch.einsum("bld,bd->bl", keys, query).div(math.sqrt(D_MODEL)).softmax(dim=1)
        selected = torch.einsum("bl,bld->bd", weights, keys)
        logits = self.head(self.final_norm(workspace + selected[:, None, :]))
        return logits, None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, capturable=spec.device_type == "cuda"))


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer)
