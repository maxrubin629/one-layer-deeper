# v1.3 — Exact-T diagnostic (research only)

This diagnostic parses the marked decimal `T` field on-device and freezes each
example after exactly that many shared transition steps. It answers whether a
candidate's learned trajectory controller, rather than its transition, is the
failure mode.

This implementation is deliberately blocked from official rendering and must
never be treated as a competition candidate or used to transfer weights.

- **Eligibility:** research-only.
- **Parent:** `v1.2_persistent_context`.

Running `python research.py --device cpu` executes a two-row `T=1,64`
diagnostic and emits one JSON result object. The isolated Colab research job
passes `--device cuda` after verifying the requested GPU.
