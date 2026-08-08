# v4.5 — Analytic invariant-orbit research

This research-only experiment replaces point-attractor solving with an analytic
invariant trajectory. It represents a context-conditioned curve using a real
Fourier basis (equivalently a conjugate-symmetric Laurent series on the unit
circle):

```text
h(theta; c) = a0(c) + sum_k [ak(c) cos(k theta) + bk(c) sin(k theta)].
```

The coefficients are learned only from the invariance residual

```text
h(theta + omega; c) - F(h(theta; c), theta, c),
```

not by regressing onto the known fixture curve. The synthetic dynamics have a
persistent phase orbit and contraction only transverse to that orbit. The known
curve is used after fitting solely to measure recovery error. Making `theta`
explicit fixes the phase gauge.

Run it with:

```bash
python experiments/v4.5_analytic_invariant_orbit_research/research.py
```

The final stdout line is machine-readable JSON. This solver and its synthetic
fixture are research-only and blocked from candidate rendering.
