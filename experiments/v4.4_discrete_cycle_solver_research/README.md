# v4.4 — Discrete invariant-cycle solver research

Instead of seeking one fixed point, this research-only experiment solves all
states of a finite invariant cycle simultaneously:

```text
z[j + 1] = F(z[j]),    z[0] = F(z[q - 1]).
```

It uses joint L-BFGS optimization for cycle lengths `q ∈ {2, 4, 8}` on two
controlled systems:

- planar rotation by `2π/q`;
- a repeated permutation of `q` coordinates.

The first state is anchored to remove the phase/gauge degeneracy that would
otherwise permit shifted or zero-amplitude solutions. Reported metrics include
the full cycle residual, explicit last-to-first closure error, and anchor error.

Run it with:

```bash
python experiments/v4.4_discrete_cycle_solver_research/research.py
```

The final stdout line is JSON. This is a synthetic research solver, not a
competition submission.
