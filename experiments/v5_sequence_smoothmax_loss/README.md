# v5 — Sequence smooth-max loss

This experiment keeps the v3.1 architecture and replaces flat token CE with a
boundary-aware smooth maximum: first across valid output digits in each row,
then across rows in the batch.  The loss uses `TokenLossBatch.valid_mask`, so
padding never participates.  It is a differentiable surrogate for exact-match
failure and contains no intermediate-answer supervision.
