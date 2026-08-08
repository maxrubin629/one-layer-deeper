# v3.1 — Damped oscillatory correction

This candidate adds a transient correction state `e_t` to v3.  The correction
undergoes a learned context-conditioned rotation and a strictly bounded learned
damping factor before receiving a fresh proposal from `p_t`, `h_t`, and `c_N`.
The persistent orbit remains a residual, noncontractive state.
