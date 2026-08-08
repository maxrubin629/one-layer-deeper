# v5.2 — Composition and stability auxiliaries

This child retains v5.1 and adds two unsupervised penalties.  A learned skip
operator is trained to agree with two applications of the autonomous orbit
transition, and a soft norm-growth penalty discourages explosive orbit steps.
The auxiliaries act on learned states only; they do not generate or supervise
intermediate task answers.
