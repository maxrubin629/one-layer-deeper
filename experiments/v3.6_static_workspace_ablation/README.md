# v3.6 — Static workspace ablation

This candidate keeps v3's autonomous recurrent orbit and learned trajectory
readout, but prevents the token workspace from changing inside the recurrent
loop. The unchanged workspace update and nonlinearity are still evaluated and
connected through a zero coefficient, preserving v3's learned state, tensor
shapes, and principal compute graph.

- **Causal question:** does recurrent token-local workspace refinement cause
  v3's gain?
- **Controlled difference from v3:** the per-step workspace residual coefficient
  changes from `0.25` to `0.0`.
- **Eligibility:** candidate-safe.
