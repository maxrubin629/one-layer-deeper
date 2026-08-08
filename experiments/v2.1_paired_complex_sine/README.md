# v2.1 — Paired-real complex sine

The recurrent core represents 64 complex channels as 128 real scalars and
applies the exact paired-real complex sine identities after a learned
complex-linear projection. Normalization remains outside that analytic core.

- **Hypothesis:** an entire complex transition improves stability beyond real periodic features.
- **Eligibility:** candidate-safe.
- **Parent:** `v1.2_persistent_context`.
