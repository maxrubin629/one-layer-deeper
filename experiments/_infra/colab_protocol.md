# Colab A100 Execution Protocol

The installed `google-colab-cli` version is currently `0.6.0`. Recheck its
package metadata and bundled `COLAB_SKILL.md` immediately before allocation.

## Read-only authentication check

```bash
/Users/max_rubin/.local/bin/colab \
  --logtostderr \
  --config /private/tmp/one-layer-deeper-colab-sessions.json \
  --auth=adc \
  sessions
```

`--logtostderr` avoids the restricted `~/.config/colab-cli/colab.log` path.
The isolated config keeps this project's local session bookkeeping separate.

If ADC scopes are missing, stop and ask the user to run:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Authentication verification does not authorize allocating a VM.

## Approval gate

Before `colab run`, state the requested accelerator, job count, purpose, and
maximum timeout. A100 is the only automatic development request. If A100 is
unavailable, stop; never fall back to L4 or T4 without a new decision.

## CPU environment preflight (A1)

Build this payload locally:

```bash
python -m experiments build-colab-job \
  --mode environment-preflight --accelerator CPU \
  --output artifacts/one_layer_deeper/jobs/cpu-environment-preflight.py
```

After explicit execution approval, run it without a GPU request. It verifies
clone, pinned checkout, Python 3.13.5, frozen dependency installation, the
locked Torch runtime, JSONL envelopes, and exit propagation, then exits without
training. Verify teardown with `sessions` exactly as for GPU jobs.

```bash
/Users/max_rubin/.local/bin/colab \
  --logtostderr \
  --config /private/tmp/one-layer-deeper-colab-sessions.json \
  --auth=adc \
  run --session old-cpu-environment-preflight --timeout 900 \
  artifacts/one_layer_deeper/jobs/cpu-environment-preflight.py
```

## A100 preflight

```bash
/Users/max_rubin/.local/bin/colab \
  --logtostderr \
  --config /private/tmp/one-layer-deeper-colab-sessions.json \
  --auth=adc \
  run --gpu A100 --session old-a100-preflight --timeout 300 \
  artifacts/one_layer_deeper/jobs/a100-preflight.py
```

Never pass `--keep`. The preflight verifies the actual device name contains
`A100` and that the runtime reports BF16 support before cloning or installing
anything. A100-40GB and A100-80GB are separate hardware groups and each begins
with `v0_baseline_adamw`.

## Offline A4 payload builds

```bash
python -m experiments build-colab-job \
  --mode sweep --matrix e1-full --accelerator A100 \
  --manifest experiments/_infra/manifests/a100_e1_screen_10s.json \
  --output artifacts/one_layer_deeper/jobs/a100-e1-screen-10s.py

python -m experiments build-colab-job \
  --mode research --matrix exact-t-diagnostics --accelerator A100 \
  --manifest experiments/_infra/manifests/a100_e1_screen_10s.json \
  --output artifacts/one_layer_deeper/jobs/a100-e1-exact-t-screen-10s.py
```

Building these files is local and allocates nothing. The exact-T job uses the
benchmark runner but remains labeled and mechanically isolated as research.

## Capturing and ingesting an approved job

After explicit execution approval, capture the complete CLI output rather than
leaving the only copy in a terminal. The log may contain CLI diagnostics around
the job's JSONL envelopes; the ingester ignores unrelated lines.

```bash
mkdir -p artifacts/one_layer_deeper/logs
/Users/max_rubin/.local/bin/colab \
  --logtostderr \
  --config /private/tmp/one-layer-deeper-colab-sessions.json \
  --auth=adc \
  run --gpu A100 --session old-a100-e1-screen --timeout 7200 \
  artifacts/one_layer_deeper/jobs/a100-e1-screen-10s.py \
  > artifacts/one_layer_deeper/logs/a100-e1-screen.log 2>&1

python -m experiments report \
  artifacts/one_layer_deeper/logs/a100-e1-screen.log
```

`report` copies the raw log to a SHA-256-addressed location and writes one
immutable `record.json` for every candidate or research envelope. Re-ingesting
the identical log is idempotent; a run-ID collision with different contents is
an error. The generated report includes all previously ingested run records, so
a baseline captured in an earlier approved job remains available for pairing.

## Cleanup

Run the read-only `sessions` command after every job. If this project's exact
named session remains, stop that session and check again. Do not stop unrelated
user sessions.

## Evidence labels

- Mac/CPU: `contract-smoke`
- A100 derived or exact-manifest work: `research-screen`
- H100 public manifest whose bytes match the blob at the pinned Git revision:
  `tier-faithful`
- Hosted service: `hosted-authoritative`

A100 results select finalists; they are never described as H100-equivalent.
