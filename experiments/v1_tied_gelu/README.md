# v1 — Tied GELU recurrence

A learned prompt encoder initializes an autonomous latent state. One shared
GELU transition produces a trajectory, and a learned `T`-field query reads from
that trajectory without exposing raw `T` or a loop index to the transition.

- **Hypothesis:** weight tying improves extrapolation to unseen composition depth.
- **Eligibility:** candidate-safe.
- **Parent:** `v0_baseline_adamw`.
