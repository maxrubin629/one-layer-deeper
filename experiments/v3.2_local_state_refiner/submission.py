"""Damped factored state with one local real correction refiner."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

D_MODEL, ORBIT_WIDTH, CORRECTION_WIDTH = 128, 128, 64
TRAIN_LOOPS, EVAL_LOOPS = 4, 64


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


class DampedRefinedCorrection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        half = CORRECTION_WIDTH // 2
        self.logit_rho = nn.Parameter(torch.zeros(half))
        self.angle = nn.Linear(D_MODEL, half)
        self.from_orbit, self.from_workspace, self.from_context = nn.Linear(ORBIT_WIDTH, CORRECTION_WIDTH, bias=False), nn.Linear(D_MODEL, CORRECTION_WIDTH, bias=False), nn.Linear(D_MODEL, CORRECTION_WIDTH, bias=False)
        self.refiner_norm = RMSNorm(CORRECTION_WIDTH)
        self.refiner = nn.Linear(CORRECTION_WIDTH, CORRECTION_WIDTH)
        self.refiner_context = nn.Linear(D_MODEL, CORRECTION_WIDTH, bias=False)

    @property
    def correction_rho(self) -> Tensor:
        return 0.98 * self.logit_rho.sigmoid()

    def forward(self, e: Tensor, p: Tensor, h: Tensor, c: Tensor) -> Tensor:
        real, imag = e.chunk(2, -1)
        angle = self.angle(c)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotated = torch.cat((real * cosine - imag * sine, real * sine + imag * cosine), -1)
        rho = torch.cat((self.correction_rho, self.correction_rho), -1)
        proposal = torch.tanh(self.from_orbit(p) + self.from_workspace(h) + self.from_context(c))
        damped = rho * rotated + (1.0 - rho) * proposal
        return damped + 0.1 * torch.tanh(self.refiner(self.refiner_norm(damped)) + self.refiner_context(c))


class Model(nn.Module):
    num_loops = EVAL_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding, self.position_embedding = nn.Embedding(spec.vocab_size, D_MODEL), nn.Embedding(spec.max_seq_len, D_MODEL)
        self.encoder_norm, self.encoder_up, self.encoder_down = RMSNorm(D_MODEL), nn.Linear(D_MODEL, 2 * D_MODEL), nn.Linear(2 * D_MODEL, D_MODEL)
        self.context_score, self.orbit_score, self.correction_score, self.query_score = (nn.Linear(D_MODEL, 1) for _ in range(4))
        self.context_projection, self.orbit_projection = nn.Linear(D_MODEL, D_MODEL), nn.Linear(D_MODEL, ORBIT_WIDTH)
        self.correction_projection, self.query_projection = nn.Linear(D_MODEL, CORRECTION_WIDTH), nn.Linear(D_MODEL, D_MODEL)
        self.orbit_transition, self.correction = OrbitTransition(), DampedRefinedCorrection()
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
        e = self.correction_projection(pool(h, x_mask, self.correction_score))
        q = self.query_projection(pool(h, t_mask, self.query_score))
        trajectory = [torch.cat((p, e), -1)]
        loops = TRAIN_LOOPS if self.training else EVAL_LOOPS
        for _ in range(loops):
            p = self.orbit_transition(p, c)
            e = self.correction(e, p, masked_mean(h, dynamics_mask), c)
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
