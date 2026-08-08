# v3.3 — Gabor corrector

This candidate replaces v3.1's ordinary correction proposal with a localized
periodic (Gabor-like) activation.  Learned frequency, phase, and positive
envelope rate are context-conditioned.  The Gabor activation is confined to
the transient correction subsystem; the persistent orbit uses the same learned
bilinear residual law as its parent.
