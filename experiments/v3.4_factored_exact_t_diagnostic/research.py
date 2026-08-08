"""Research-only exact-T controller for the factored recurrent model."""

from __future__ import annotations

import argparse
import json
import math
import time
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

D_MODEL, ORBIT_WIDTH, CORRECTION_WIDTH, MAX_STEPS = 128, 128, 64, 64
T_MARKER, DIGIT_OFFSET, DIGIT_COUNT = 4, 7, 10


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


def parse_decimal_t(input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
    """Parse the final T field with tensor operations; this makes the file research-only."""

    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device)[None, :].expand(batch, -1)
    marker_positions = torch.where(input_ids == T_MARKER, positions, -torch.ones_like(positions))
    start = marker_positions.max(dim=1).values
    valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    is_digit = (input_ids >= DIGIT_OFFSET) & (input_ids < DIGIT_OFFSET + DIGIT_COUNT) & valid
    is_after_t = positions > start[:, None]
    delimiter = is_after_t & valid & ~is_digit
    stop = torch.where(delimiter, positions, torch.full_like(positions, length)).min(dim=1).values
    selected = is_digit & is_after_t & (positions < stop[:, None])
    value = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
    for column in range(length):
        digit = input_ids[:, column] - DIGIT_OFFSET
        value = torch.where(selected[:, column], value * 10 + digit, value)
    return value.clamp(min=0, max=MAX_STEPS)


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


class DampedCorrection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_rho = nn.Parameter(torch.zeros(CORRECTION_WIDTH))
        self.proposal = nn.Linear(ORBIT_WIDTH + D_MODEL + D_MODEL, CORRECTION_WIDTH)

    def forward(self, e: Tensor, p: Tensor, h: Tensor, c: Tensor) -> Tensor:
        rho = 0.98 * self.logit_rho.sigmoid()
        target = torch.tanh(self.proposal(torch.cat((p, h, c), -1)))
        return rho * e + (1.0 - rho) * target


class Model(nn.Module):
    num_loops = MAX_STEPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding, self.position_embedding = nn.Embedding(spec.vocab_size, D_MODEL), nn.Embedding(spec.max_seq_len, D_MODEL)
        self.encoder_norm, self.encoder_up, self.encoder_down = RMSNorm(D_MODEL), nn.Linear(D_MODEL, 2 * D_MODEL), nn.Linear(2 * D_MODEL, D_MODEL)
        self.context_score, self.orbit_score, self.correction_score = (nn.Linear(D_MODEL, 1) for _ in range(3))
        self.context_projection, self.orbit_projection, self.correction_projection = nn.Linear(D_MODEL, D_MODEL), nn.Linear(D_MODEL, ORBIT_WIDTH), nn.Linear(D_MODEL, CORRECTION_WIDTH)
        self.orbit_transition, self.correction = OrbitTransition(), DampedCorrection()
        self.workspace_norm, self.workspace_local = RMSNorm(D_MODEL), nn.Linear(D_MODEL, D_MODEL)
        self.orbit_to_workspace, self.correction_to_workspace, self.context_to_workspace = nn.Linear(ORBIT_WIDTH, D_MODEL, bias=False), nn.Linear(CORRECTION_WIDTH, D_MODEL, bias=False), nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.selected_to_workspace = nn.Linear(ORBIT_WIDTH + CORRECTION_WIDTH, D_MODEL)
        self.final_norm, self.head = RMSNorm(D_MODEL), nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        h = self.token_embedding(input_ids) + self.position_embedding(positions)
        h = h + self.encoder_down(F.gelu(self.encoder_up(self.encoder_norm(h))))
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        n_mask, x_mask = field_mask(input_ids, valid, 2, 3), field_mask(input_ids, valid, 3, 4)
        dynamics_mask = n_mask | x_mask
        c = self.context_projection(pool(h, n_mask, self.context_score))
        p = self.orbit_projection(pool(h, x_mask, self.orbit_score))
        e = self.correction_projection(pool(h, x_mask, self.correction_score))
        exact_steps = parse_decimal_t(input_ids, attention_mask)
        for step in range(MAX_STEPS):
            next_p = self.orbit_transition(p, c)
            next_e = self.correction(e, next_p, masked_mean(h, dynamics_mask), c)
            next_h = h + 0.25 * torch.tanh(self.workspace_local(self.workspace_norm(h)) + self.orbit_to_workspace(next_p)[:, None] + self.correction_to_workspace(next_e)[:, None] + self.context_to_workspace(c)[:, None])
            active = exact_steps > step
            p = torch.where(active[:, None], next_p, p)
            e = torch.where(active[:, None], next_e, e)
            h = torch.where(active[:, None, None], next_h, h)
        selected = self.selected_to_workspace(torch.cat((p, e), -1))
        logits = self.head(self.final_norm(h + selected[:, None]))
        return logits, {"exact_steps": exact_steps}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, capturable=spec.device_type == "cuda"))


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer)


def run_diagnostic(device: str = "cpu") -> dict[str, object]:
    torch_device = torch.device(device)
    torch.manual_seed(74)
    input_ids = torch.tensor(
        [
            [2, 8, 10, 3, 9, 4, 8, 0, 0, 0],
            [2, 8, 11, 10, 3, 12, 4, 13, 11, 0],
        ],
        dtype=torch.long,
        device=torch_device,
    )
    attention_mask = input_ids.ne(0)
    model = build_model(ModelSpec(17, 10, 500_000_000)).to(torch_device).eval()
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits, auxiliary = model(input_ids, attention_mask)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return {
        "experiment_id": "v3.4_factored_exact_t_diagnostic",
        "eligibility": "research-only",
        "device": str(torch_device),
        "parsed_exact_t": auxiliary["exact_steps"].tolist(),
        "maximum_masked_steps": MAX_STEPS,
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "model_state_elements": sum(
            tensor.numel() for tensor in model.state_dict().values()
        ),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.device), sort_keys=True))


if __name__ == "__main__":
    main()
