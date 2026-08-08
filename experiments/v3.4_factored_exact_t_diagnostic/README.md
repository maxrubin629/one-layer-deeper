# v3.4 — Factored exact-T diagnostic

This research-only diagnostic uses the v3.1 factored model but parses the
provided decimal `T` field on-device and freezes each row after exactly that
many autonomous transitions.  It distinguishes a learned trajectory-controller
failure from a transition-law failure.

Exact parsing is intentionally isolated from candidate sources.  The blocker
file and `research.py` entrypoint make this directory ineligible for official
rendering or submission.

Running `python research.py --device cpu` executes a two-row `T=1,64`
diagnostic and emits one JSON result object. The isolated Colab research job
passes `--device cuda` after verifying the requested GPU.
