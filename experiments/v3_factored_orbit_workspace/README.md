# v3 — Factored orbit and workspace

This candidate separates a persistent orbit `p_t` from the token workspace
`h_t`.  The learned bilinear orbit update is autonomous once initialized: it
receives only `p_t` and the persistent prompt context `c_N`.  Information flows
from the orbit into the workspace, while the workspace cannot write back into
the orbit.  A learned query selects from the complete recurrent trajectory.

The orbit is a residual state and is never multiplied by a global contraction
factor.  This experiment intentionally has no correction state; v3.1 adds one.
