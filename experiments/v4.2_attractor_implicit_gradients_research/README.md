# v4.2 — Attractor implicit-gradient research

This research-only experiment compares four ways to differentiate a tiny
contractive equilibrium:

- full explicit differentiation through a long Picard solve;
- one-step differentiation at a detached equilibrium;
- a short damped phantom unroll at a detached equilibrium;
- implicit differentiation through the fixed-point equation.

It also runs the complete zero/Gaussian/learned-proposal initialization matrix
with initial-only versus persistent conditioning, Picard versus memory-three
Anderson solving, and fixed-count versus residual-based stopping. Each cell
records residual, iterations, proposal/equilibrium distance, coordinate
accuracy, and latency. A Jacobian spectral-radius estimate is reported beside
the gradient-agreement results. The system is deliberately small enough that
the implicit linear system can be formed exactly, making gradient agreement
directly testable rather than inferred from training accuracy.

Run it with:

```bash
python experiments/v4.2_attractor_implicit_gradients_research/research.py
```

The final stdout line is a JSON object. This program is a synthetic diagnostic,
contains no competition model, and is mechanically blocked from submission.
