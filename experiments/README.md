# Versioned Experiment Program

This directory contains the reproducible architecture program for the One Layer
Deeper repeated-composition benchmark. Every experiment is version-prefixed,
names its comparison parent, and is either a standalone competition candidate or
a mechanically blocked research program.

## Scientific spine

1. `v0` establishes the published baseline.
2. `v1` isolates tied recurrence and persistent context.
3. `v2` compares periodic, real-bilinear, holomorphic, and conjugate-dependent
   transition classes.
4. `v3` factors the state into a persistent orbit, transient correction, and
   token workspace.
5. `v4` tests Attractor Models machinery only in the correction/workspace
   subsystem and separately houses paper-faithful research.
6. `v5` tests exact-match-aware and auxiliary losses.

The orbit is never globally contracted. Candidate-safe transitions do not
receive raw `T`, a loop index, or repeatedly injected `x`. Literal modular
arithmetic, factorization, discrete logs, residue tables, fixed complex
squaring, and intermediate modular-answer supervision are out of scope.

## Commands

```bash
python -m experiments list
python -m experiments validate --all
python -m experiments render v2.3_holomorphic_bilinear
python -m experiments run v0_baseline_adamw \
  --manifest experiments/_infra/manifests/contract_cpu.json \
  --score-class contract-smoke
python -m experiments report artifacts/one_layer_deeper/runs
python -m experiments report artifacts/one_layer_deeper/logs/a100-e1-screen.log
python -m experiments build-colab-job --mode preflight --output /tmp/a100.py
python -m experiments build-colab-job --mode sweep \
  --experiment v0_baseline_adamw --experiment v4_attractor_picard_corrector \
  --manifest benchmark/manifests/h100_easy_e1.json --accelerator A100 \
  --output artifacts/one_layer_deeper/jobs/a100-finalists-e1.py
```

Generated jobs, rendered copies, logs, and immutable records belong under
`artifacts/one_layer_deeper/` and are intentionally ignored by Git.
When `report` receives a captured Colab log, it verifies each complete JSONL
envelope, stores a hash-addressed copy of the raw log, writes one immutable
`record.json` per candidate, and then reports all accumulated run records.

## GPU evidence policy

A100 is the primary development GPU while H100 is unavailable. An A100-40GB,
A100-80GB, and H100 are distinct hardware groups. Every group starts with its
own `v0` baseline; absolute measurements are never mixed across groups. A100
results are research results, not H100-equivalent competition scores.

No Colab allocation or hosted submission is performed by the local build and
validation commands.
