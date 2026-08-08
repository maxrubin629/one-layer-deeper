"""Research-only Fourier/Laurent invariant-orbit residual experiment."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch
from torch import nn


EXPERIMENT_ID = "v4.5_analytic_invariant_orbit_research"
DTYPE = torch.float64
STATE_DIMENSION = 2


def target_curve(theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Known analytic invariant used only to construct and evaluate the fixture."""

    return torch.stack(
        (
            (1.0 + 0.15 * context) * torch.cos(theta)
            + 0.20 * torch.cos(2.0 * theta),
            (1.0 - 0.10 * context) * torch.sin(theta)
            - 0.15 * torch.sin(3.0 * theta)
            + 0.10 * context,
        ),
        dim=-1,
    )


def dynamics(
    state: torch.Tensor,
    theta: torch.Tensor,
    context: torch.Tensor,
    *,
    omega: float,
    transverse_rate: float,
) -> torch.Tensor:
    """Forced dynamics whose invariant curve contracts transversely."""

    current_curve = target_curve(theta, context)
    next_curve = target_curve(theta + omega, context)
    return next_curve + transverse_rate * (state - current_curve)


class ConditionalFourierCurve(nn.Module):
    """Context-affine real Fourier representation of a Laurent trajectory."""

    def __init__(self, maximum_mode: int, *, device: torch.device) -> None:
        super().__init__()
        self.maximum_mode = maximum_mode
        self.constant = nn.Parameter(
            0.05 * torch.randn(2, STATE_DIMENSION, dtype=DTYPE, device=device)
        )
        self.cosine = nn.Parameter(
            0.05
            * torch.randn(
                2, maximum_mode, STATE_DIMENSION, dtype=DTYPE, device=device
            )
        )
        self.sine = nn.Parameter(
            0.05
            * torch.randn(
                2, maximum_mode, STATE_DIMENSION, dtype=DTYPE, device=device
            )
        )

    def forward(self, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context_basis = torch.stack((torch.ones_like(context), context), dim=-1)
        modes = torch.arange(
            1,
            self.maximum_mode + 1,
            dtype=theta.dtype,
            device=theta.device,
        )
        phases = theta.unsqueeze(-1) * modes
        value = torch.einsum("nc,cd->nd", context_basis, self.constant)
        value = value + torch.einsum(
            "nk,nc,ckd->nd", torch.cos(phases), context_basis, self.cosine
        )
        value = value + torch.einsum(
            "nk,nc,ckd->nd", torch.sin(phases), context_basis, self.sine
        )
        return value


def make_grid(
    *,
    phase_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    phases = torch.arange(phase_count, dtype=DTYPE, device=device)
    phases = phases * (2.0 * math.pi / phase_count)
    contexts = torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE, device=device)
    theta_grid, context_grid = torch.meshgrid(phases, contexts, indexing="ij")
    return theta_grid.reshape(-1), context_grid.reshape(-1)


def invariance_residual(
    curve: ConditionalFourierCurve,
    theta: torch.Tensor,
    context: torch.Tensor,
    *,
    omega: float,
    transverse_rate: float,
) -> torch.Tensor:
    current = curve(theta, context)
    predicted_next = dynamics(
        current,
        theta,
        context,
        omega=omega,
        transverse_rate=transverse_rate,
    )
    represented_next = curve(theta + omega, context)
    return represented_next - predicted_next


def fit_invariant_curve(
    *,
    device: torch.device,
    seed: int,
    maximum_mode: int,
    phase_count: int,
    omega: float,
    transverse_rate: float,
    max_iterations: int,
) -> tuple[ConditionalFourierCurve, int, float]:
    torch.manual_seed(seed)
    curve = ConditionalFourierCurve(maximum_mode, device=device)
    theta, context = make_grid(phase_count=phase_count, device=device)
    optimizer = torch.optim.LBFGS(
        curve.parameters(),
        lr=0.8,
        max_iter=max_iterations,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        residual = invariance_residual(
            curve,
            theta,
            context,
            omega=omega,
            transverse_rate=transverse_rate,
        )
        loss = residual.square().mean()
        loss.backward()
        return loss

    initial_loss = optimizer.step(closure)
    return curve, closure_calls, float(initial_loss.detach().item())


@torch.no_grad()
def transverse_stability(
    *,
    device: torch.device,
    omega: float,
    transverse_rate: float,
    steps: int = 12,
) -> dict[str, float | int]:
    theta = torch.tensor([0.37], dtype=DTYPE, device=device)
    context = torch.tensor([0.4], dtype=DTYPE, device=device)
    on_orbit = target_curve(theta, context)
    perturbed = on_orbit + torch.tensor(
        [[0.4, -0.3]], dtype=DTYPE, device=device
    )
    initial_distance = float(torch.linalg.vector_norm(perturbed - on_orbit).item())
    for _ in range(steps):
        perturbed = dynamics(
            perturbed,
            theta,
            context,
            omega=omega,
            transverse_rate=transverse_rate,
        )
        on_orbit = dynamics(
            on_orbit,
            theta,
            context,
            omega=omega,
            transverse_rate=transverse_rate,
        )
        theta = theta + omega
    final_distance = float(torch.linalg.vector_norm(perturbed - on_orbit).item())
    empirical_rate = (final_distance / initial_distance) ** (1.0 / steps)
    return {
        "steps": steps,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "empirical_transverse_rate": empirical_rate,
    }


def run_experiment(
    *,
    device: str = "cpu",
    seed: int = 74,
    maximum_mode: int = 4,
    phase_count: int = 96,
    max_iterations: int = 140,
    transverse_rate: float = 0.35,
) -> dict[str, Any]:
    if maximum_mode < 3:
        raise ValueError("maximum_mode must be at least three for this fixture")
    if not 0.0 < transverse_rate < 1.0:
        raise ValueError("transverse_rate must define a strict contraction")
    torch_device = torch.device(device)
    omega = 2.0 * math.pi / 7.0
    curve, closure_calls, initial_loss = fit_invariant_curve(
        device=torch_device,
        seed=seed,
        maximum_mode=maximum_mode,
        phase_count=phase_count,
        omega=omega,
        transverse_rate=transverse_rate,
        max_iterations=max_iterations,
    )
    theta, context = make_grid(phase_count=phase_count * 2, device=torch_device)
    with torch.no_grad():
        residual = invariance_residual(
            curve,
            theta,
            context,
            omega=omega,
            transverse_rate=transverse_rate,
        )
        represented = curve(theta, context)
        expected = target_curve(theta, context)
        per_sample_residual = torch.linalg.vector_norm(residual, dim=-1)
        curve_rmse = (represented - expected).square().mean().sqrt()
    stability = transverse_stability(
        device=torch_device,
        omega=omega,
        transverse_rate=transverse_rate,
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "eligibility": "research-only",
        "device": str(torch_device),
        "seed": seed,
        "representation": "real Fourier / conjugate-symmetric Laurent",
        "laurent_modes": list(range(-maximum_mode, maximum_mode + 1)),
        "phase_gauge": "explicit theta coordinate",
        "omega": omega,
        "transverse_rate": transverse_rate,
        "optimizer_closure_calls": closure_calls,
        "optimizer_initial_loss": initial_loss,
        "invariance_residual_rms": float(residual.square().mean().sqrt().item()),
        "invariance_residual_max": float(per_sample_residual.max().item()),
        "curve_rmse": float(curve_rmse.item()),
        "transverse_stability": stability,
        "torch_version": torch.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--maximum-mode", type=int, default=4)
    parser.add_argument("--phase-count", type=int, default=96)
    parser.add_argument("--max-iterations", type=int, default=140)
    parser.add_argument("--transverse-rate", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        device=args.device,
        seed=args.seed,
        maximum_mode=args.maximum_mode,
        phase_count=args.phase_count,
        max_iterations=args.max_iterations,
        transverse_rate=args.transverse_rate,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
