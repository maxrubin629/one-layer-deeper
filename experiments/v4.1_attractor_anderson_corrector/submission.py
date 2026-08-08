"""Correction-only attractor with differentiable memory-three Anderson mixing."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

D_MODEL, ORBIT_WIDTH, CORRECTION_WIDTH = 128, 128, 64
TRAIN_LOOPS, EVAL_LOOPS = 4, 64
TRAIN_SOLVER_STEPS, EVAL_SOLVER_STEPS, ANDERSON_MEMORY = 3, 6, 3


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


def pool(x: Tensor, mask: Tensor | None, score: nn.Linear) -> Tensor:
    weights = score(x).squeeze(-1)
    if mask is not None:
        weights = weights.masked_fill(~mask.bool(), -torch.inf)
    return torch.einsum("bs,bsd->bd", weights.softmax(-1), x)


def field_mask(input_ids: Tensor, valid: Tensor, marker: int, stop_marker: int) -> Tensor:
    routed = (input_ids == marker).cumsum(1).bool() & ~(input_ids == stop_marker).cumsum(1).bool() & (input_ids != marker) & valid
    return torch.where(routed.any(1, keepdim=True), routed, valid)


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(x.dtype)
    return (x * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1.0)


def depth_encoding(length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    depth = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, D_MODEL, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / D_MODEL))
    angles = depth * frequencies[None]
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1).to(dtype)


class OrbitTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(ORBIT_WIDTH)
        self.left, self.right = nn.Linear(ORBIT_WIDTH, ORBIT_WIDTH), nn.Linear(ORBIT_WIDTH, ORBIT_WIDTH)
        self.left_context, self.right_context = nn.Linear(D_MODEL, ORBIT_WIDTH, bias=False), nn.Linear(D_MODEL, ORBIT_WIDTH, bias=False)
        self.out = nn.Linear(ORBIT_WIDTH, ORBIT_WIDTH)

    def forward(self, p: Tensor, c: Tensor) -> Tensor:
        z = self.norm(p)
        a, b = self.left(z) + self.left_context(c), self.right(z) + self.right_context(c)
        return p + 0.25 * self.out(torch.tanh(a * b / math.sqrt(ORBIT_WIDTH)))


class AndersonCorrector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proposal_layer = nn.Linear(ORBIT_WIDTH + D_MODEL + D_MODEL, CORRECTION_WIDTH)
        self.recurrent_weight = nn.Parameter(torch.empty(CORRECTION_WIDTH, CORRECTION_WIDTH))
        self.logit_rho = nn.Parameter(torch.zeros(CORRECTION_WIDTH))
        nn.init.normal_(self.recurrent_weight, std=0.02)

    @property
    def correction_rho(self) -> Tensor:
        return 0.45 * self.logit_rho.sigmoid()

    def proposal(self, p: Tensor, h: Tensor, c: Tensor) -> Tensor:
        workspace = h.mean(1) if h.ndim == 3 else h
        return self.proposal_layer(torch.cat((p, workspace, c), -1))

    def fixed_point_map(self, e: Tensor, proposal: Tensor) -> Tensor:
        recurrent = 0.05 * F.linear(e, torch.tanh(self.recurrent_weight)) / CORRECTION_WIDTH
        rho = self.correction_rho
        return rho * e + (1.0 - rho) * torch.tanh(proposal + recurrent)

    def forward(self, p: Tensor, h: Tensor, c: Tensor, iterations: int) -> Tensor:
        proposal = self.proposal(p, h, c)
        e = torch.tanh(proposal)
        history_x: list[Tensor] = []
        history_f: list[Tensor] = []
        for _ in range(iterations):
            mapped = self.fixed_point_map(e, proposal)
            history_x.append(e)
            history_f.append(mapped)
            history_x, history_f = history_x[-ANDERSON_MEMORY:], history_f[-ANDERSON_MEMORY:]
            if len(history_x) == 1:
                e = mapped
                continue
            x_stack, f_stack = torch.stack(history_x, 1), torch.stack(history_f, 1)
            residual = f_stack - x_stack
            gram = torch.bmm(residual.float(), residual.float().transpose(1, 2))
            identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)[None]
            gram = gram + 1e-4 * identity
            ones = torch.ones(gram.shape[0], gram.shape[1], 1, device=gram.device, dtype=gram.dtype)
            coefficients = torch.linalg.solve(gram, ones)
            coefficients = coefficients / coefficients.sum(1, keepdim=True).clamp_min(1e-6)
            e = (coefficients.to(f_stack.dtype) * f_stack).sum(1)
        return e


class Model(nn.Module):
    num_loops = EVAL_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding, self.position_embedding = nn.Embedding(spec.vocab_size, D_MODEL), nn.Embedding(spec.max_seq_len, D_MODEL)
        self.encoder_norm, self.encoder_up, self.encoder_down = RMSNorm(D_MODEL), nn.Linear(D_MODEL, 2 * D_MODEL), nn.Linear(2 * D_MODEL, D_MODEL)
        self.context_score, self.orbit_score, self.query_score = (nn.Linear(D_MODEL, 1) for _ in range(3))
        self.context_projection, self.orbit_projection, self.query_projection = nn.Linear(D_MODEL, D_MODEL), nn.Linear(D_MODEL, ORBIT_WIDTH), nn.Linear(D_MODEL, D_MODEL)
        self.orbit_transition, self.correction = OrbitTransition(), AndersonCorrector()
        self.workspace_norm, self.workspace_local = RMSNorm(D_MODEL), nn.Linear(D_MODEL, D_MODEL)
        self.orbit_to_workspace, self.correction_to_workspace, self.context_to_workspace = nn.Linear(ORBIT_WIDTH, D_MODEL, bias=False), nn.Linear(CORRECTION_WIDTH, D_MODEL, bias=False), nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.trajectory_key = nn.Linear(ORBIT_WIDTH + CORRECTION_WIDTH, D_MODEL)
        self.final_norm, self.head = RMSNorm(D_MODEL), nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        h = self.token_embedding(input_ids) + self.position_embedding(positions)
        h = h + self.encoder_down(F.gelu(self.encoder_up(self.encoder_norm(h))))
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        n_mask, x_mask, t_mask = field_mask(input_ids, valid, 2, 3), field_mask(input_ids, valid, 3, 4), field_mask(input_ids, valid, 4, 5)
        dynamics_mask = n_mask | x_mask
        c = self.context_projection(pool(h, n_mask, self.context_score))
        p = self.orbit_projection(pool(h, x_mask, self.orbit_score))
        q = self.query_projection(pool(h, t_mask, self.query_score))
        e = torch.zeros(p.shape[0], CORRECTION_WIDTH, device=p.device, dtype=p.dtype)
        trajectory = [torch.cat((p, e), -1)]
        loops, solver_steps = (TRAIN_LOOPS, TRAIN_SOLVER_STEPS) if self.training else (EVAL_LOOPS, EVAL_SOLVER_STEPS)
        for _ in range(loops):
            p = self.orbit_transition(p, c)
            e = self.correction(p, masked_mean(h, dynamics_mask), c, solver_steps)
            update = self.workspace_local(self.workspace_norm(h)) + self.orbit_to_workspace(p)[:, None] + self.correction_to_workspace(e)[:, None] + self.context_to_workspace(c)[:, None]
            h = h + 0.25 * torch.tanh(update)
            trajectory.append(torch.cat((p, e), -1))
        keys = self.trajectory_key(torch.stack(trajectory, 1))
        keys = keys + depth_encoding(loops + 1, keys.device, keys.dtype)[None]
        weights = torch.einsum("bld,bd->bl", keys, q).div(math.sqrt(D_MODEL)).softmax(1)
        selected = torch.einsum("bl,bld->bd", weights, keys)
        return self.head(self.final_norm(h + selected[:, None])), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, capturable=spec.device_type == "cuda"))


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer)
