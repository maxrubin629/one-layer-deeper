# v5.1 — Prompt reconstruction

This child retains v5's smooth-max objective and adds a `0.01` reconstruction
loss over valid input tokens.  Reconstruction logits are produced from the
token workspace already used by the predictor.  No synthetic data or
intermediate modular target is introduced.
