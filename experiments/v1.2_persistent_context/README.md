# v1.2 — Persistent modulus context

This variant injects a learned representation of the marked `N` field into the
shared transition at every recurrent step. The marked `X` field is used only to
initialize state, and learned `T` information remains in the readout.

- **Hypothesis:** persistent modulus conditioning improves cross-`N` generalization.
- **Eligibility:** candidate-safe.
- **Parent:** `v1_tied_gelu`.
