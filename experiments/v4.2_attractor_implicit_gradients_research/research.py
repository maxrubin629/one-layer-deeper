"""Research-only fixed-point solver and gradient-estimator comparison.

This is intentionally a standalone synthetic program.  It is not, and must not
be transformed into, a competition submission.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch


EXPERIMENT_ID = "v4.2_attractor_implicit_gradients_research"
DTYPE = torch.float64


def attractor_map(
    state: torch.Tensor,
    theta: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    """An elementwise contraction when ``abs(theta) < 1``."""

    return torch.tanh(theta * state + condition)


def objective(state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (state - target).square().mean()


def fixed_point_residual(
    state: torch.Tensor,
    theta: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    return (attractor_map(state, theta, condition) - state).abs().max()


def solve_picard(
    theta: torch.Tensor,
    condition: torch.Tensor,
    *,
    max_iterations: int = 160,
    tolerance: float = 1e-12,
) -> tuple[torch.Tensor, int, float]:
    state = torch.zeros_like(condition)
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        next_state = attractor_map(state, theta, condition)
        residual = float((next_state - state).abs().max().item())
        state = next_state
        if residual <= tolerance:
            break
    return state, iteration, residual


def solve_anderson(
    theta: torch.Tensor,
    condition: torch.Tensor,
    *,
    memory: int = 3,
    max_iterations: int = 80,
    tolerance: float = 1e-12,
    regularization: float = 1e-8,
    mixing: float = 1.0,
) -> tuple[torch.Tensor, int, float]:
    """Batched Anderson acceleration for independent scalar equilibria."""

    if memory < 1:
        raise ValueError("memory must be positive")
    state = torch.zeros_like(condition)
    history_state: list[torch.Tensor] = []
    history_mapped: list[torch.Tensor] = []
    residual = float("inf")

    for iteration in range(1, max_iterations + 1):
        mapped = attractor_map(state, theta, condition)
        residual = float((mapped - state).abs().max().item())
        if residual <= tolerance:
            state = mapped
            break

        history_state.append(state)
        history_mapped.append(mapped)
        history_state = history_state[-memory:]
        history_mapped = history_mapped[-memory:]
        count = len(history_state)

        if count == 1:
            state = mapped
            continue

        xs = torch.stack(history_state, dim=1)
        fs = torch.stack(history_mapped, dim=1)
        differences = fs - xs
        gram = differences.unsqueeze(2) * differences.unsqueeze(1)
        eye = torch.eye(count, dtype=condition.dtype, device=condition.device)
        gram = gram + regularization * eye.unsqueeze(0)

        # Minimize the mixed residual subject to coefficients summing to one.
        batch = condition.numel()
        system = torch.zeros(
            batch,
            count + 1,
            count + 1,
            dtype=condition.dtype,
            device=condition.device,
        )
        system[:, 0, 1:] = 1.0
        system[:, 1:, 0] = 1.0
        system[:, 1:, 1:] = gram
        rhs = torch.zeros(
            batch, count + 1, 1, dtype=condition.dtype, device=condition.device
        )
        rhs[:, 0, 0] = 1.0
        coefficients = torch.linalg.solve(system, rhs)[:, 1:, 0]
        mixed_f = (coefficients * fs).sum(dim=1)
        mixed_x = (coefficients * xs).sum(dim=1)
        state = mixing * mixed_f + (1.0 - mixing) * mixed_x

    residual = float(fixed_point_residual(state, theta, condition).item())
    return state, iteration, residual


def conditioned_map(
    state: torch.Tensor,
    theta: torch.Tensor,
    condition: torch.Tensor,
    *,
    persistent_conditioning: bool,
) -> torch.Tensor:
    injected = condition if persistent_conditioning else torch.zeros_like(condition)
    return torch.tanh(theta * state + injected)


def solve_picard_variant(
    theta: torch.Tensor,
    condition: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    persistent_conditioning: bool,
    max_iterations: int,
    tolerance: float,
    fixed_iterations: bool,
) -> tuple[torch.Tensor, int, float]:
    state = initial_state.clone()
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        next_state = conditioned_map(
            state,
            theta,
            condition,
            persistent_conditioning=persistent_conditioning,
        )
        residual = float((next_state - state).abs().max().item())
        state = next_state
        if not fixed_iterations and residual <= tolerance:
            break
    final_residual = conditioned_map(
        state,
        theta,
        condition,
        persistent_conditioning=persistent_conditioning,
    ) - state
    return state, iteration, float(final_residual.abs().max().item())


def solve_anderson_variant(
    theta: torch.Tensor,
    condition: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    persistent_conditioning: bool,
    max_iterations: int,
    tolerance: float,
    fixed_iterations: bool,
    memory: int = 3,
    regularization: float = 1e-8,
) -> tuple[torch.Tensor, int, float]:
    state = initial_state.clone()
    history_state: list[torch.Tensor] = []
    history_mapped: list[torch.Tensor] = []
    for iteration in range(1, max_iterations + 1):
        mapped = conditioned_map(
            state,
            theta,
            condition,
            persistent_conditioning=persistent_conditioning,
        )
        residual = float((mapped - state).abs().max().item())
        if not fixed_iterations and residual <= tolerance:
            state = mapped
            break
        history_state.append(state)
        history_mapped.append(mapped)
        history_state = history_state[-memory:]
        history_mapped = history_mapped[-memory:]
        count = len(history_state)
        if count == 1:
            state = mapped
            continue
        xs = torch.stack(history_state, dim=1)
        fs = torch.stack(history_mapped, dim=1)
        differences = fs - xs
        gram = differences.unsqueeze(2) * differences.unsqueeze(1)
        eye = torch.eye(count, dtype=state.dtype, device=state.device)
        gram = gram + regularization * eye.unsqueeze(0)
        ones = torch.ones(
            condition.numel(), count, 1, dtype=state.dtype, device=state.device
        )
        coefficients = torch.linalg.solve(gram, ones)
        coefficients = coefficients / coefficients.sum(dim=1, keepdim=True).clamp_min(
            1e-12
        )
        state = (coefficients[..., 0] * fs).sum(dim=1)
    final_residual = conditioned_map(
        state,
        theta,
        condition,
        persistent_conditioning=persistent_conditioning,
    ) - state
    return state, iteration, float(final_residual.abs().max().item())


def fit_polynomial_proposal(
    train_condition: torch.Tensor,
    train_equilibrium: torch.Tensor,
    evaluation_condition: torch.Tensor,
) -> torch.Tensor:
    def features(values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (torch.ones_like(values), values, values.square(), values.pow(3)), dim=1
        )

    coefficients = torch.linalg.lstsq(
        features(train_condition), train_equilibrium[:, None]
    ).solution
    return (features(evaluation_condition) @ coefficients)[:, 0]


def solver_ablation_matrix(
    theta: torch.Tensor,
    condition: torch.Tensor,
) -> tuple[dict[str, dict[str, Any]], torch.Tensor]:
    persistent_reference, _, _ = solve_picard_variant(
        theta,
        condition,
        torch.zeros_like(condition),
        persistent_conditioning=True,
        max_iterations=240,
        tolerance=1e-14,
        fixed_iterations=False,
    )
    initial_only_reference = torch.zeros_like(condition)
    train_condition = torch.linspace(
        -1.0, 1.0, 65, dtype=condition.dtype, device=condition.device
    )
    train_equilibrium, _, _ = solve_picard_variant(
        theta,
        train_condition,
        torch.zeros_like(train_condition),
        persistent_conditioning=True,
        max_iterations=240,
        tolerance=1e-14,
        fixed_iterations=False,
    )
    generator = torch.Generator(device=condition.device).manual_seed(74)
    initializations = {
        "zero": torch.zeros_like(condition),
        "gaussian": 0.35
        * torch.randn(
            condition.shape,
            generator=generator,
            dtype=condition.dtype,
            device=condition.device,
        ),
        "learned_proposal": fit_polynomial_proposal(
            train_condition, train_equilibrium, condition
        ),
    }
    matrix: dict[str, dict[str, Any]] = {}
    for initialization, initial_state in initializations.items():
        for conditioning in ("initial_only", "persistent"):
            persistent = conditioning == "persistent"
            reference = persistent_reference if persistent else initial_only_reference
            for solver_name, solver in (
                ("picard", solve_picard_variant),
                ("anderson", solve_anderson_variant),
            ):
                for stopping, iterations, fixed in (
                    ("fixed_12", 12, True),
                    ("residual", 160, False),
                ):
                    if condition.device.type == "cuda":
                        torch.cuda.synchronize(condition.device)
                    started = time.perf_counter()
                    solution, used_iterations, residual = solver(
                        theta,
                        condition,
                        initial_state,
                        persistent_conditioning=persistent,
                        max_iterations=iterations,
                        tolerance=1e-12,
                        fixed_iterations=fixed,
                    )
                    if condition.device.type == "cuda":
                        torch.cuda.synchronize(condition.device)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    distance = (solution - reference).abs()
                    key = f"{initialization}:{conditioning}:{solver_name}:{stopping}"
                    matrix[key] = {
                        "initialization": initialization,
                        "conditioning": conditioning,
                        "solver": solver_name,
                        "stopping": stopping,
                        "iterations": used_iterations,
                        "fixed_point_residual": residual,
                        "proposal_to_equilibrium_distance": float(
                            (initial_state - reference).abs().max().item()
                        ),
                        "solution_to_equilibrium_distance": float(distance.max().item()),
                        "coordinate_accuracy_at_1e-6": float(
                            distance.le(1e-6).double().mean().item()
                        ),
                        "latency_ms": latency_ms,
                    }
    return matrix, persistent_reference


def explicit_gradient(
    theta_value: float,
    condition: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
) -> tuple[float, float]:
    theta = torch.tensor(theta_value, dtype=DTYPE, device=condition.device)
    theta.requires_grad_(True)
    state = torch.zeros_like(condition)
    for _ in range(iterations):
        state = attractor_map(state, theta, condition)
    loss = objective(state, target)
    gradient = torch.autograd.grad(loss, theta)[0]
    return float(gradient.item()), float(loss.item())


def truncated_gradient(
    method: str,
    equilibrium: torch.Tensor,
    theta_value: float,
    condition: torch.Tensor,
    target: torch.Tensor,
    phantom_steps: int,
) -> float:
    theta = torch.tensor(theta_value, dtype=DTYPE, device=condition.device)
    theta.requires_grad_(True)
    state = equilibrium.detach()
    if method == "one_step":
        state = attractor_map(state, theta, condition)
    elif method == "phantom":
        for _ in range(phantom_steps):
            mapped = attractor_map(state, theta, condition)
            state = 0.5 * state + 0.5 * mapped
    else:
        raise ValueError(f"unknown truncated-gradient method: {method}")
    loss = objective(state, target)
    return float(torch.autograd.grad(loss, theta)[0].item())


def implicit_gradient(
    equilibrium: torch.Tensor,
    theta_value: float,
    condition: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Differentiate ``z = F(z, theta)`` via an exact tiny linear solve."""

    theta = torch.tensor(theta_value, dtype=DTYPE, device=condition.device)
    theta.requires_grad_(True)
    state = equilibrium.detach().clone().requires_grad_(True)
    mapped = attractor_map(state, theta, condition)
    loss = objective(state, target)
    loss_state = torch.autograd.grad(loss, state, retain_graph=True)[0]

    jacobian = torch.autograd.functional.jacobian(
        lambda value: attractor_map(value, theta, condition),
        state,
        create_graph=False,
    )
    identity = torch.eye(
        state.numel(), dtype=state.dtype, device=state.device
    )
    adjoint = torch.linalg.solve(identity - jacobian.T, loss_state)
    gradient = torch.autograd.grad(mapped, theta, grad_outputs=adjoint)[0]
    return float(gradient.item())


def relative_error(estimate: float, reference: float) -> float:
    return abs(estimate - reference) / max(abs(reference), 1e-15)


def run_experiment(
    *,
    device: str = "cpu",
    theta_value: float = 0.55,
    explicit_iterations: int = 120,
    phantom_steps: int = 5,
) -> dict[str, Any]:
    if not 0.0 < abs(theta_value) < 1.0:
        raise ValueError("theta must define a strict contraction")
    torch_device = torch.device(device)
    condition = torch.linspace(-0.8, 0.8, 9, dtype=DTYPE, device=torch_device)
    target = 0.35 * torch.sin(1.7 * condition) - 0.1 * condition
    theta = torch.tensor(theta_value, dtype=DTYPE, device=torch_device)

    picard, picard_iterations, picard_residual = solve_picard(theta, condition)
    anderson, anderson_iterations, anderson_residual = solve_anderson(
        theta, condition
    )
    solver_matrix, persistent_reference = solver_ablation_matrix(theta, condition)
    jacobian_diagonal = theta * (1.0 - persistent_reference.square())

    explicit, loss = explicit_gradient(
        theta_value, condition, target, explicit_iterations
    )
    one_step = truncated_gradient(
        "one_step",
        picard,
        theta_value,
        condition,
        target,
        phantom_steps,
    )
    phantom = truncated_gradient(
        "phantom",
        picard,
        theta_value,
        condition,
        target,
        phantom_steps,
    )
    implicit = implicit_gradient(
        picard, theta_value, condition, target
    )

    gradients = {
        "explicit": {
            "value": explicit,
            "relative_error_to_explicit": 0.0,
        },
        "one_step": {
            "value": one_step,
            "relative_error_to_explicit": relative_error(one_step, explicit),
        },
        "phantom": {
            "value": phantom,
            "relative_error_to_explicit": relative_error(phantom, explicit),
            "steps": phantom_steps,
        },
        "implicit": {
            "value": implicit,
            "relative_error_to_explicit": relative_error(implicit, explicit),
        },
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "eligibility": "research-only",
        "device": str(torch_device),
        "dtype": str(DTYPE),
        "theta": theta_value,
        "loss": loss,
        "solvers": {
            "picard": {
                "iterations": picard_iterations,
                "fixed_point_residual": picard_residual,
            },
            "anderson": {
                "iterations": anderson_iterations,
                "memory": 3,
                "fixed_point_residual": anderson_residual,
            },
            "solution_max_distance": float((picard - anderson).abs().max().item()),
        },
        "solver_matrix": solver_matrix,
        "jacobian_estimate": {
            "spectral_radius": float(jacobian_diagonal.abs().max().item()),
            "frobenius_norm": float(jacobian_diagonal.norm().item()),
            "contraction_upper_bound": abs(theta_value),
        },
        "gradients": gradients,
        "torch_version": torch.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--theta", type=float, default=0.55)
    parser.add_argument("--explicit-iterations", type=int, default=120)
    parser.add_argument("--phantom-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        device=args.device,
        theta_value=args.theta,
        explicit_iterations=args.explicit_iterations,
        phantom_steps=args.phantom_steps,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
