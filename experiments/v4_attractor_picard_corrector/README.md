# v4 — Picard attractor corrector

The outer orbit remains the noncontractive learned bilinear recurrence from the
factored family.  At each outer step, this candidate builds a learned correction
proposal and performs three Picard iterations in training or six in evaluation.
The proposal, current orbit, workspace summary, and persistent context are
injected into every correction iteration.  Only `e_t` is attracted.

All iterations are explicit PyTorch operations and use ordinary evaluator-owned
autograd; the candidate contains no custom backward or derivative calls.
