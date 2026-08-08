# v4.3 — Equilibrium internalization research

This research-only program asks whether a learned initial proposal has absorbed
the work of an attractor solve. It trains a small proposal network against a
three-dimensional synthetic contraction, then evaluates the same checkpoint at:

- `K=0` (proposal only / corrector removed);
- `K=1`;
- `K=3` (the trained refinement depth);
- `K=25` (full solve);
- `K=64` (extra iterations).

For each depth it reports distance to a high-precision equilibrium, fixed-point
residual, sign accuracy, and latency. A proposal that matches the full solve at
`K=0` is explicitly labeled as internalization; it is not counted as evidence
that inference-time recurrence is doing useful computation.

Run it with:

```bash
python experiments/v4.3_equilibrium_internalization_research/research.py
```

The final stdout line is machine-readable JSON. This experiment is synthetic,
research-only, and blocked from candidate rendering.
