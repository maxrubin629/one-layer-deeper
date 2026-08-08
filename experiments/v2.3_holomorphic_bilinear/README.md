# v2.3 — Holomorphic bilinear transition

The paired-real core applies two independent learned complex-linear projections,
multiplies their outputs as complex numbers, and maps the result back with a
third learned complex-linear projection. It contains no conjugate dependence
and does not hard-code complex squaring.

- **Hypothesis:** holomorphic multiplicative structure composes more cleanly than real bilinear structure.
- **Eligibility:** candidate-safe.
- **Parent:** `v1.2_persistent_context`.
