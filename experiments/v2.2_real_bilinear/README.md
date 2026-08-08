# v2.2 — Real bilinear transition

Two independent learned real projections are multiplied elementwise and mapped
back to the recurrent state. This isolates multiplicative degree growth from
complex structure.

- **Hypothesis:** learned bilinear features improve repeated composition.
- **Eligibility:** candidate-safe.
- **Parent:** `v1.2_persistent_context`.
