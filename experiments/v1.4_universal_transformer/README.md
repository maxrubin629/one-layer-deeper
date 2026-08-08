# v1.4 — Fixed-depth Universal Transformer

This candidate recurrently applies one shared full Transformer block to every
token state. Each of four training steps and sixty-four evaluation steps adds a
fixed sinusoidal recurrent-time encoding before the same multi-head
self-attention and position-wise feed-forward parameters are reapplied. There
is no Adaptive Computation Time (ACT), pooling recurrence, exact-`T` controller,
or task-specific solver path.

- **Hypothesis:** repeatedly exchanging information across all token states is
  the missing Universal Transformer mechanism in `v1_tied_gelu`, whose
  self-attention runs once before a pooled-state MLP recurrence.
- **Eligibility:** candidate-safe.
- **Parent:** `v0_baseline_adamw`.
- **Parameter control:** exactly parameter-matched to `v0` for every model
  specification. It retains the same tied embedding/head, learned token
  positions, one Transformer block, and final RMSNorm; the time encoding has no
  persistent state.
- **Compute caveat:** it is parameter-matched, not compute-matched, to `v0`.
  Training applies the block four times and evaluation applies it sixty-four
  times, matching the recurrent schedule of `v1_tied_gelu`.
- **Fidelity caveat:** this isolates the paper's fixed-depth shared-attention
  recurrence and sinusoidal step signal inside the benchmark's decoder-only
  pre-norm baseline. It is not a reproduction of the paper's full
  encoder-decoder, post-norm, or ACT variants.
