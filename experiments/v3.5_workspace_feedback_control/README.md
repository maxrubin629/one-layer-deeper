# v3.5 — Workspace feedback control

This control changes only v3's asymmetric information flow. The first orbit
transition is unchanged; after each workspace update, a parameter-free masked
mean of the current valid token workspace is added to the persistent context
for the next orbit transition. Padding never contributes to that summary.

Setting `WORKSPACE_FEEDBACK_SCALE = 0.0` restores the unchanged autonomous v3
computation exactly with shared weights. Learned modules, persistent state,
optimizer, field routing, recurrence depths, workspace update, and trajectory
readout are otherwise unchanged.

- **Causal question:** does preventing workspace-to-orbit feedback help v3?
- **Controlled difference from v3:** workspace feedback only.
- **Eligibility:** candidate-safe.
