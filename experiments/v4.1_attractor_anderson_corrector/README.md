# v4.1 — Anderson attractor corrector

This candidate accelerates v4's correction-only fixed-point solve using
memory-three Anderson mixing with a regularized batched linear solve.  The
underlying fixed-point map and outer orbit are unchanged.  Coefficients are
computed in float32 for BF16 safety, and all operations remain differentiable
through normal PyTorch autograd.
