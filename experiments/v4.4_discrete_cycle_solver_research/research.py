"""Research-only joint solver for synthetic finite invariant cycles."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Callable, Sequence

import torch


EXPERIMENT_ID = "v4.4_discrete_cycle_solver_research"
DTYPE = torch.float64
Transform = Callable[[torch.Tensor], torch.Tensor]


def rotation_transform(cycle_length: int, device: torch.device) -> Transform:
    angle = 2.0 * math.pi / cycle_length
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=DTYPE,
        device=device,
    )
    return lambda state: state @ rotation.T


def permutation_transform(state: torch.Tensor) -> torch.Tensor:
    return torch.roll(state, shifts=1, dims=-1)


def cycle_residual(states: torch.Tensor, transform: Transform) -> torch.Tensor:
    next_states = torch.roll(states, shifts=-1, dims=0)
    return next_states - transform(states)


def solve_cycle(
    *,
    transform: Transform,
    cycle_length: int,
    dimension: int,
    anchor: torch.Tensor,
    seed: int,
    max_iterations: int = 120,
) -> tuple[torch.Tensor, int, float]:
    """Jointly optimize every state, anchoring the first to fix cycle phase."""

    generator = torch.Generator(device=anchor.device)
    generator.manual_seed(seed)
    states = torch.nn.Parameter(
        0.25
        * torch.randn(
            cycle_length,
            dimension,
            dtype=DTYPE,
            device=anchor.device,
            generator=generator,
        )
    )
    optimizer = torch.optim.LBFGS(
        [states],
        lr=0.8,
        max_iter=max_iterations,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        residual = cycle_residual(states, transform)
        anchor_loss = (states[0] - anchor).square().mean()
        loss = residual.square().mean() + 4.0 * anchor_loss
        loss.backward()
        return loss

    final_loss = optimizer.step(closure)
    return states.detach(), closure_calls, float(final_loss.detach().item())


def summarize_cycle(
    *,
    states: torch.Tensor,
    transform: Transform,
    anchor: torch.Tensor,
    closure_calls: int,
    optimizer_returned_loss: float,
) -> dict[str, float | int]:
    residual = cycle_residual(states, transform)
    per_state = torch.linalg.vector_norm(residual, dim=-1)
    closure = states[0] - transform(states[-1])
    return {
        "optimizer_closure_calls": closure_calls,
        "optimizer_initial_loss": optimizer_returned_loss,
        "cycle_residual_rms": float(residual.square().mean().sqrt().item()),
        "cycle_residual_max": float(per_state.max().item()),
        "closure_error": float(torch.linalg.vector_norm(closure).item()),
        "anchor_error": float(torch.linalg.vector_norm(states[0] - anchor).item()),
        "minimum_state_norm": float(torch.linalg.vector_norm(states, dim=-1).min().item()),
        "maximum_state_norm": float(torch.linalg.vector_norm(states, dim=-1).max().item()),
    }


def run_experiment(
    *,
    device: str = "cpu",
    seed: int = 74,
    cycle_lengths: Sequence[int] = (2, 4, 8),
    max_iterations: int = 120,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    results: dict[str, dict[str, dict[str, float | int]]] = {
        "rotation": {},
        "permutation": {},
    }
    for cycle_length in cycle_lengths:
        if cycle_length < 2:
            raise ValueError("cycle lengths must be at least two")

        rotation = rotation_transform(cycle_length, torch_device)
        rotation_anchor = torch.tensor([1.0, 0.0], dtype=DTYPE, device=torch_device)
        rotation_states, calls, initial_loss = solve_cycle(
            transform=rotation,
            cycle_length=cycle_length,
            dimension=2,
            anchor=rotation_anchor,
            seed=seed + cycle_length,
            max_iterations=max_iterations,
        )
        results["rotation"][str(cycle_length)] = summarize_cycle(
            states=rotation_states,
            transform=rotation,
            anchor=rotation_anchor,
            closure_calls=calls,
            optimizer_returned_loss=initial_loss,
        )

        permutation_anchor = torch.zeros(
            cycle_length, dtype=DTYPE, device=torch_device
        )
        permutation_anchor[0] = 1.0
        permutation_states, calls, initial_loss = solve_cycle(
            transform=permutation_transform,
            cycle_length=cycle_length,
            dimension=cycle_length,
            anchor=permutation_anchor,
            seed=seed + 100 + cycle_length,
            max_iterations=max_iterations,
        )
        results["permutation"][str(cycle_length)] = summarize_cycle(
            states=permutation_states,
            transform=permutation_transform,
            anchor=permutation_anchor,
            closure_calls=calls,
            optimizer_returned_loss=initial_loss,
        )

    maximum_residual = max(
        float(metrics["cycle_residual_max"])
        for task in results.values()
        for metrics in task.values()
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "eligibility": "research-only",
        "device": str(torch_device),
        "seed": seed,
        "phase_gauge": "first state anchored",
        "cycle_lengths": list(cycle_lengths),
        "tasks": results,
        "maximum_cycle_residual": maximum_residual,
        "torch_version": torch.__version__,
    }


def parse_cycle_lengths(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one cycle length is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--cycle-lengths", type=parse_cycle_lengths, default=(2, 4, 8))
    parser.add_argument("--max-iterations", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        device=args.device,
        seed=args.seed,
        cycle_lengths=args.cycle_lengths,
        max_iterations=args.max_iterations,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
