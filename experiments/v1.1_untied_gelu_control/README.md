# v1.1 — Untied GELU control

This inference-compute-matched control replaces the shared transition with four
distinct GELU cells. Training applies each cell once; evaluation cycles the
four-cell sequence sixteen times for a period-four, 64-call rollout. It tests
whether gains come from single-cell weight tying rather than depth alone.

- **Hypothesis:** the single-cell tied recurrence should outperform this period-four control out of depth.
- **Eligibility:** candidate-safe.
- **Parent:** `v1_tied_gelu`.
