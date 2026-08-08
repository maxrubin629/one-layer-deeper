"""Research-only equilibrium-internalization diagnostic.

The proposal and contraction solve a tiny synthetic regression problem.  No
competition data, task solver, or submission interface is present here.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


EXPERIMENT_ID = "v4.3_equilibrium_internalization_research"
STATE_DIMENSION = 3


class Proposal(nn.Module):
    def __init__(self, hidden_width: int = 24) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, STATE_DIMENSION),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.network(condition.unsqueeze(-1))


def conditioning_bias(condition: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            0.70 * torch.sin(condition),
            0.50 * torch.cos(1.3 * condition) - 0.15,
            0.40 * condition + 0.10 * torch.sin(2.0 * condition),
        ),
        dim=-1,
    )


def attractor_map(state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    """A coordinatewise contraction with Lipschitz constant at most 0.4."""

    return torch.tanh(0.4 * state + conditioning_bias(condition))


def refine(
    initial_state: torch.Tensor,
    condition: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    state = initial_state
    for _ in range(iterations):
        state = attractor_map(state, condition)
    return state


@torch.no_grad()
def equilibrium(
    condition: torch.Tensor,
    *,
    max_iterations: int = 120,
    tolerance: float = 1e-12,
) -> torch.Tensor:
    state = torch.zeros(
        condition.shape[0],
        STATE_DIMENSION,
        dtype=condition.dtype,
        device=condition.device,
    )
    for _ in range(max_iterations):
        next_state = attractor_map(state, condition)
        if float((next_state - state).abs().max().item()) <= tolerance:
            return next_state
        state = next_state
    return state


def train_proposal(
    *,
    device: torch.device,
    seed: int,
    training_steps: int,
    trained_iterations: int,
) -> tuple[Proposal, list[float]]:
    torch.manual_seed(seed)
    model = Proposal().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    conditions = torch.linspace(-1.5, 1.5, 192, device=device)
    teacher = equilibrium(conditions)
    trace: list[float] = []

    for step in range(training_steps):
        optimizer.zero_grad(set_to_none=True)
        proposal = model(conditions)
        refined = refine(proposal, conditions, trained_iterations)
        # The second term makes proposal internalization measurable instead of
        # letting a contractive full solve erase proposal quality entirely.
        loss = F.mse_loss(refined, teacher) + 0.10 * F.mse_loss(proposal, teacher)
        loss.backward()
        optimizer.step()
        if step == 0 or step == training_steps - 1 or (step + 1) % 25 == 0:
            trace.append(float(loss.detach().item()))
    return model, trace


@torch.no_grad()
def evaluate_depth(
    model: Proposal,
    condition: torch.Tensor,
    teacher: torch.Tensor,
    iterations: int,
) -> dict[str, float | int]:
    start = time.perf_counter()
    proposal = model(condition)
    state = refine(proposal, condition, iterations)
    if condition.device.type == "cuda":
        torch.cuda.synchronize(condition.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    residual = attractor_map(state, condition) - state
    sign_accuracy = (state.sign() == teacher.sign()).all(dim=-1).double().mean()
    return {
        "iterations": iterations,
        "equilibrium_mse": float(F.mse_loss(state, teacher).item()),
        "fixed_point_residual": float(residual.abs().max().item()),
        "sign_accuracy": float(sign_accuracy.item()),
        "latency_ms": elapsed_ms,
    }


def run_experiment(
    *,
    device: str = "cpu",
    seed: int = 74,
    training_steps: int = 250,
    trained_iterations: int = 3,
) -> dict[str, Any]:
    if training_steps < 1:
        raise ValueError("training_steps must be positive")
    if trained_iterations < 1:
        raise ValueError("trained_iterations must be positive")
    torch_device = torch.device(device)
    model, training_trace = train_proposal(
        device=torch_device,
        seed=seed,
        training_steps=training_steps,
        trained_iterations=trained_iterations,
    )
    model.eval()
    condition = torch.linspace(-1.6, 1.6, 129, device=torch_device)
    teacher = equilibrium(condition)
    labels_and_depths = (
        ("k0_proposal_only", 0),
        ("k1", 1),
        (f"k{trained_iterations}_trained", trained_iterations),
        ("k25_full_solve", 25),
        ("k64_extra_depth", 64),
    )
    evaluations = {
        label: evaluate_depth(model, condition, teacher, depth)
        for label, depth in labels_and_depths
    }
    proposal_mse = float(evaluations["k0_proposal_only"]["equilibrium_mse"])
    full_mse = float(evaluations["k25_full_solve"]["equilibrium_mse"])
    full_scale = max(float(teacher.square().mean().item()), 1e-12)
    internalization_ratio = proposal_mse / full_scale

    return {
        "experiment_id": EXPERIMENT_ID,
        "eligibility": "research-only",
        "device": str(torch_device),
        "seed": seed,
        "training_steps": training_steps,
        "trained_iterations": trained_iterations,
        "training_loss_trace": training_trace,
        "evaluations": evaluations,
        "proposal_to_equilibrium_mse": proposal_mse,
        "full_solve_mse": full_mse,
        "internalization_ratio": internalization_ratio,
        "equilibrium_internalized": internalization_ratio < 1e-3,
        "interpretation": (
            "proposal already approximates the equilibrium; recurrence gains must be "
            "reported separately"
            if internalization_ratio < 1e-3
            else "refinement still performs measurable inference-time correction"
        ),
        "torch_version": torch.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--training-steps", type=int, default=250)
    parser.add_argument("--trained-iterations", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        device=args.device,
        seed=args.seed,
        training_steps=args.training_steps,
        trained_iterations=args.trained_iterations,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
