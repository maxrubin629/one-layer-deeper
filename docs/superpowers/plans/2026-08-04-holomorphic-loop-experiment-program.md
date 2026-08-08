# Holomorphic Loop Architecture Experiment Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, competition-compliant experiment system that tests whether tied analytic/holomorphic dynamics, local residue refinement, and learned Koopman/skip operators outperform the published one-block Transformer on repeated modular squaring.

**Architecture:** Keep the competition's public runner as the only scoring loop. Generate standalone `submission.py` files from explicit experiment specifications, run them first on matched local/Colab hardware, and promote only paired winners to hosted H100 evaluations. Separate the real prompt encoder and output decoder from a small paired-real complex core so holomorphicity, recurrence, conditioning, refinement, readout, loss, and optimizer can be ablated independently.

**Tech Stack:** Python 3.13.5, PyTorch 2.12.1, the public `benchmark` API, `unittest`, `uv`, Google Colab CLI 0.6.0, JSON/JSONL result artifacts.

---

## 1. Verified constraints and planning conclusions

Sources of truth are the current checkout at commit `e32c2f985f8ed4107c96d00271448777954ecc0c`, the [official competition repository](https://github.com/tilde-research/one-layer-deeper), the [live leaderboard](https://onelayerdeeper.ai/), and the [official Colab CLI repository](https://github.com/googlecolab/google-colab-cli).

### Competition envelope

| Constraint | Consequence for the experiments |
|---|---|
| Deadline: August 31, 2026 at 10:00 PM PT | Finish broad screening before the final week; reserve hosted Hard attempts for a fully validated finalist. |
| Exactly one UTF-8 `submission.py`, at most 256 KiB | Every candidate must be rendered and validated as a standalone file. Experiment helpers are local tooling only. |
| At most 500,000,000 persistent state elements | Count trainable parameters, frozen parameters, and persistent buffers. Shared/tied state counts once. |
| Randomly initialized, trained weights; no loaded or hard-coded weights | Do not ship pretrained charts, residue tables, character bases, codebooks, or fixed squaring coefficients. |
| Learned forward pass only | No modular arithmetic, factorization, discrete logs, deterministic residue lookup, or other task solver in `forward`. |
| End-to-end GPU graph | No CPU parsing/offload and no gradient break between final loss and predictive parameters. Use paired real channels rather than CPU/native-complex fallbacks. |
| Evaluator owns data order, forward/loss calls, backward, clipping, optimizer cadence, and scoring | Use the official runner unchanged for every comparable score. Local short-budget manifests are explicitly non-scoring screens. |
| No data inspection, augmentation, custom training loops, participant backward, or manifest overrides in official submissions | Synthetic/oracle diagnostics are research-only. Never use their weights or code path in a submitted candidate. |
| Recurrence, adaptive depth, depth curricula, and `self.training` train/eval branches are allowed | A slow recurrent training teacher and fast evaluation student are admissible if both are learned and remain inside the normal model call. |
| `OptimizerBundle` permits 1-8 evaluator-owned backward passes and up to 8 updates reusing a batch | Optimizer experiments must use these documented hooks, never nested model/loss/backward calls. |
| Easy/Medium/Hard training budgets: 60/600/3,600 H100 seconds | Construction/import/compilation consume training time; final evaluation gets a separate 30/300/1,800 seconds. |
| Daily accepted attempts: 60 Easy, 6 Medium, 1 Hard | Use Colab for breadth, hosted Easy/Medium for calibration, and Hard only after the promotion gate. |
| Easy/Medium use bidirectional prompt attention; Hard is hidden | Do not assume Hard preserves public prompt format beyond the published `ModelSpec`/forward contract. |
| Hard ranking is lexicographic | Optimize seen-`N` certified T, then OOD-`N` certified T, then first-uncertified-rung accuracies—not only average token loss. |

The live Hard leaderboard currently has no entry certified at T=1. Therefore early experiments must retain dense diagnostics—first-rung exact accuracy, mean exact accuracy, loss, completed steps, and evaluation completion—instead of treating `Max T = <1` as a tie.

### Hardware conclusion

The development machine is a verified Apple M5 Max MacBook Pro with 128 GB unified memory. It is useful for source validation, unit tests, CPU smoke runs, result analysis, and optional research-only MPS sanity checks. It is not a score-comparable baseline because official manifests require `cuda:0`, BF16 AMP, and an H100.

Use three labels without mixing them:

1. **Contract smoke:** M5/CPU `smoke_cpu.json`; proves that the submission loads, trains, and evaluates.
2. **Research baseline:** the published baseline and candidates on the same Colab accelerator, exact repo commit, dataset, seed, and budget.
3. **Competition baseline:** hosted Easy/Medium results from the official service; this is authoritative when Colab cannot allocate an H100 or its software/hardware environment differs.

### Submission-legality ledger

| Mechanism | Status | Rule for implementation |
|---|---|---|
| Learned general polynomial basis `1, z, z²` with random trainable coefficients | Candidate-safe | Never initialize or freeze coefficients to perform squaring. |
| Paired-real implementation of complex-linear algebra | Candidate-safe | Treat it as a general layer; keep every tensor on GPU. |
| Learned prompt pooling and learned trajectory readout | Candidate-safe | The transition receives context/state, not hard-parsed `N`, `x`, `T`, or a loop index. |
| Full-prompt reconstruction and state/semigroup penalties returned through `auxiliary` | Candidate-safe | Use input tokens already received by the model; combine only through `token_training_loss`. |
| Oracle field extraction or `gather(states, true_T)` | Research-only | Use only to diagnose whether the core is capable; never render into an official candidate. |
| Synthetic pretraining or augmented modular examples | Research-only | Use only for mechanism tests from fresh random weights; never transfer weights into a submission. |
| Deterministic on-GPU `T` parser used only as a rollout selector | Organizer clarification required | Default official path is learned trajectory attention. Do not submit until organizers approve in writing. |
| Repeated matrix squaring of a learned operator | Organizer clarification required | It is generic operator composition, but confirm it is not viewed as a task-specific solver. |
| A fixed complex-squaring basis with only surrounding transforms learned | Organizer clarification required | Prefer the unrestricted learned polynomial family until clarified. |
| Hard-coded modular reduction, factorization, characters, or residue tables | Prohibited | Do not implement, even as a convenience path. |

## 2. Hypotheses and controlled architecture families

The central claim is:

> A modulus-conditioned, autonomous holomorphic quadratic generator, repeatedly composed and locally reprojected onto a learned residue manifold, extrapolates in T better than a generic tied nonlinear block under the same state and time budgets.

Test it as separable hypotheses:

- **H1 — recurrence:** shared transition weights generalize to unseen depth better than a one-pass or untied compute-matched control.
- **H2 — periodicity:** sine/cosine improves depth extrapolation beyond GELU.
- **H3 — analyticity:** quadratic or entire-function cells improve law learning independent of periodicity.
- **H4 — holomorphicity:** excluding conjugate dependence improves composition stability beyond a parameter-matched widely-linear complex control.
- **H5 — degree match:** a learned quadratic cell benefits specifically because degree grows as `2^T` under composition.
- **H6 — autonomous dynamics:** keeping T, step index, and repeated x injection out of the transition improves generator learning.
- **H7 — persistent context:** injecting a learned modulus/context representation at each step improves cross-N generalization.
- **H8 — local correction:** one small non-holomorphic refiner after each analytic transition reduces phase/error drift without collapsing cycles to a fixed point.
- **H9 — decodable state:** shared readout and prompt/state reconstruction make intermediate states represent residues rather than arbitrary scratch space.
- **H10 — learned operator acceleration:** Koopman/skip students preserve recurrence quality while fitting the half-budget evaluation clock.
- **H11 — exact-match loss:** sequence-level smooth-max loss improves worst-digit/worst-example behavior over flat token CE.

### Candidate interfaces

All templates expose the same internal boundary:

```python
class EncodedPrompt(NamedTuple):
    token_states: Tensor       # [B, S, D]
    initial_state: Tensor      # [B, 2K], paired real/imaginary
    persistent_context: Tensor # [B, C]
    readout_query: Tensor      # [B, D]

class Transition(nn.Module):
    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        raise NotImplementedError

class TrajectoryReadout(nn.Module):
    def forward(
        self,
        states: Tensor,        # [B, L + 1, 2K]
        query: Tensor,
        token_states: Tensor,
    ) -> Tensor:               # logits [B, S, vocab_size]
        raise NotImplementedError
```

The `Transition.forward` signature is intentionally incapable of receiving T, loop index, or raw input IDs. The learned readout may attend to the ordered trajectory; oracle selection exists only in a separate research template.

### E1 mechanism screen

Use real width 128 or complex width 64 so each latent state has 128 real scalars. Set `L_train=4`, `L_eval=64`, batch size 512, and AdamW defaults initially.

| ID | Transition | Purpose |
|---|---|---|
| B0 | Published one-block GELU baseline | Absolute reference. |
| R1 | Tied GELU residual block | Isolate weight tying/recurrence. |
| R2 | Untied GELU stack with matched forward FLOPs | Separate depth from tying; report extra state. |
| S1 | Tied real sine/cosine block | Test periodicity without holomorphicity. |
| S2 | Tied paired-real complex sine block | Test entire holomorphic periodic dynamics; constrain imaginary growth with residual scale. |
| Q1 | Tied real quadratic block | Test analytic degree match without complex structure. |
| Q2 | Tied holomorphic polynomial `a0 + a1*z + a2*z²` | Test the main inductive bias. |
| Q3 | Parameter-matched `z`/conjugate-`z` quadratic block | Negative control isolating holomorphicity. |
| Q4 | Q2 plus one learned residue refiner | Test local correction. |

Each family must share the same encoder, trajectory readout, normalization placement, optimizer, and auxiliary weights during the mechanism screen. Record actual state count, completed updates, and evaluation time; do not claim compute matching from widths alone.

### Follow-up ablations

- **E3, cross-N:** no persistent context; initial-only context; additive context every step; FiLM context every step; low-rank hypernetwork modulation. Compare initial-only x against persistent x reinjection as a negative control.
- **E5, joint N/T:** learned trajectory attention; learned adaptive-halting weights; research-only oracle T selection; T-conditioned transition negative control; step-index-conditioned transition negative control; `L_train` in `{4, 8, 16}` with `L_eval=64`.
- **Stability:** refiner steps `{0, 1, 2}`; residual scale `{0.1, 0.25, 0.5}`; prompt reconstruction weight `{0, 0.01}`; phase/amplitude penalty `{0, 0.001}`; composition penalty `{0, 0.01}`.
- **Loss:** flat valid-token CE; per-sequence mean CE; per-sequence smooth max; smooth max over both digits and batch. Start temperatures at `tau_token=0.25`, `tau_batch=0.5`.
- **Optimization after architecture freezes:** AdamW learning rate `{3e-4, 1e-3, 3e-3}`; weight decay `{0, 0.01, 0.1}`; constant versus 5% warmup + cosine decay; batch size `{256, 512, 1024}`; then one 2-pass optimizer experiment through the documented `OptimizerBundle` API.
- **Koopman/student:** diagonal complex spectrum; contractive-plus-unitary block spectrum; shared dense operator with low-rank context modulation; sequential application versus generic learned operator powering; quadratic teacher plus skip student at powers `{1,2,4,8,16,32,64}`.

## 3. Scoring, replication, and promotion protocol

### Run record

Every run writes an immutable JSON record containing:

```python
@dataclass(frozen=True)
class RunRecord:
    run_id: str
    candidate_id: str
    legality: str
    score_class: Literal[
        "contract-smoke",
        "research-screen",
        "tier-faithful",
        "hosted-authoritative",
    ]
    comparable: bool
    git_commit: str
    submission_sha256: str
    manifest_sha256: str
    dataset_config_sha256: str
    accelerator: str
    gpu_name: str
    gpu_memory_bytes: int
    python_version: str
    torch_version: str
    cuda_version: str | None
    colab_cli_version: str | None
    started_at_utc: str
    finished_at_utc: str
    result_json: dict | None
    failure: dict | None
```

Derive the ranking tuple from the official result:

```python
promotion_key = (
    seen_max_certified_t or 0,
    ood_n_max_certified_t or 0,
    seen_first_uncertified_accuracy,
    ood_n_first_uncertified_accuracy,
    mean_exact_accuracy,
)
```

Report training loss, completed training steps, optimizer-state elements, training/evaluation seconds, and per-split exact accuracy as diagnostics, not as higher-priority score components.

### Replication levels

1. **Screen:** derived 10-second research manifest, seed 74, one matched accelerator. Results are directional and never called competition scores.
2. **Confirm:** research seeds 74/75/76; a candidate must beat B0 or the current stage winner on the promotion tuple in at least two seeds and on the median tuple.
3. **Tier-faithful local/Colab:** unchanged official manifest and seed 74. The entire evaluation must finish inside its half-budget.
4. **Hosted authority:** official Easy/Medium service. Compare the exact rendered submission hash used in Colab.

### Promotion funnel

- **Stage 0:** run B0 through Mac CPU smoke, then exact E1/E3/E5 on the chosen Colab accelerator. Also submit B0 once to hosted E1/E3/E5 so Colab-to-host drift is measured.
- **Stage 1:** run B0 and R1/R2/S1/S2/Q1/Q2/Q3/Q4 on E1 10-second screens. Promote the top four valid candidates.
- **Stage 2:** confirm the top four on E1 across seeds 74/75/76, then run unchanged official E1. Promote the top two that complete evaluation and strictly improve the median promotion tuple.
- **Stage 3:** apply persistent-context and x-reinjection ablations on E3. Promote only a variant that improves OOD-N first-rung accuracy without regressing seen-N certification.
- **Stage 4:** apply learned trajectory-readout and depth ablations on E5. Use the oracle-T template only to distinguish controller failure from transition failure.
- **Stage 5:** tune refiner, auxiliary losses, learning rate, schedule, and batch size one family at a time on the Stage 4 winner. Do not cross the full grid.
- **Stage 6:** compare the winner with Koopman/skip variants. A fast evaluation branch must preserve the teacher's promotion tuple and reduce evaluation seconds by at least 25%.
- **Stage 7:** run finalists on all Easy datasets, then M1/M3/M5, then M2/M4 as scale replications. Promote only candidates that beat B0 on the joint tasks E5 and M5 and do not introduce an evaluation timeout.
- **Stage 8:** use hosted Hard only after the candidate is rule-cleared, certifies T=1 on at least one public joint task, and strictly improves first-rung accuracy on both E5 and M5 relative to B0.

Exclude a run before ranking if it OOMs, times out, returns non-finite loss, fails standalone validation, omits any scored split, uses a research-only mechanism, or has an unresolved organizer-clarification flag.

## 4. Colab CLI execution design

The installed `google-colab-cli` is version 0.6.0 and includes the bundled `colab skill`. `colab run` provisions a fresh VM, executes a local Python script, propagates the script exit code, and tears the VM down unless `--keep` is passed. Its local default execution timeout is only 30 seconds, so every benchmark command must set `--timeout` explicitly.

### Authentication and cost gate

Always spell out ADC because the installed help text and bundled skill disagree about the default auth mode:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
colab --auth=adc sessions
```

The browser-based `gcloud` step is human-run and needed only if `colab --auth=adc sessions` reports missing/expired scopes. Do not create a GPU session before the user approves the accelerator/cost gate.

Generate a self-contained preflight job and request the exact accelerator:

```bash
uv run python experiments/build_colab_job.py \
  --mode preflight \
  --output artifacts/one_layer_deeper/jobs/h100-preflight.py

colab --auth=adc run \
  --gpu H100 \
  --session old-h100-preflight \
  --timeout 300 \
  artifacts/one_layer_deeper/jobs/h100-preflight.py \
  > artifacts/one_layer_deeper/h100-preflight.stdout.jsonl \
  2> artifacts/one_layer_deeper/h100-preflight.stderr.log

colab --auth=adc sessions
```

If H100 allocation returns a quota/entitlement error, stop and report it. Do not silently fall back. An explicitly approved A100/L4/T4 run becomes a different hardware group and must receive its own B0 baseline.

### One-shot sweep job

`experiments/build_colab_job.py` must embed the rendered standalone submissions, pin the benchmark repository SHA, and generate only the requested public dataset. On the VM it will:

1. Work under `/content/one-layer-deeper`.
2. clone and check out the pinned benchmark commit;
3. install the pinned environment with `uv` and Python 3.13.5;
4. write embedded candidate files and verify their SHA-256 hashes;
5. generate the selected dataset before starting any timed runner;
6. launch a fresh subprocess for every candidate so GPU memory is released between trials;
7. call `python -m benchmark.runner` unchanged;
8. parse the final `RESULT_JSON=` line and emit one JSONL envelope per candidate;
9. continue after candidate-level OOM/timeouts but fail the job on setup or dataset corruption.

Concrete E1 screen command:

```bash
uv run python experiments/build_colab_job.py \
  --matrix e1_mechanism_screen \
  --output artifacts/one_layer_deeper/jobs/e1-mechanism-screen-h100.py

colab --auth=adc run \
  --gpu H100 \
  --session old-e1-mechanism-screen \
  --timeout 7200 \
  artifacts/one_layer_deeper/jobs/e1-mechanism-screen-h100.py \
  > artifacts/one_layer_deeper/e1-mechanism-screen-h100.stdout.jsonl \
  2> artifacts/one_layer_deeper/e1-mechanism-screen-h100.stderr.log

colab --auth=adc sessions
```

Never add `--keep` to automated sweeps. Use a named persistent session only for interactive debugging, with an explicit `colab stop -s NAME` in a `finally` path and a final `colab sessions` check.

## 5. File structure

| Path | Responsibility |
|---|---|
| `experiments/protocol.md` | Human-readable hypotheses, legality ledger, score labels, promotion gates, and organizer questions. |
| `experiments/matrix.py` | Typed candidate specs, screening matrices, dataset-generation configs, and research/official manifest selection. |
| `experiments/templates/control_submission.py.tmpl` | Standalone baseline/tied GELU/sine/real-quadratic controls. |
| `experiments/templates/holomorphic_submission.py.tmpl` | Paired-real complex encoder/core/refiner/readout and custom loss. |
| `experiments/templates/koopman_submission.py.tmpl` | Learned operator and teacher/student variants. |
| `experiments/render.py` | Render one validated standalone `submission.py` with an immutable config literal. |
| `experiments/run_local.py` | Run the official benchmark subprocess, parse `RESULT_JSON`, and write `RunRecord`. |
| `experiments/build_colab_job.py` | Build a single self-contained `colab run` payload with pinned source and embedded submissions. |
| `experiments/report.py` | Validate records, compute promotion tuples/paired deltas, and produce Markdown/CSV reports. |
| `tests/test_experiment_matrix.py` | Matrix uniqueness, legality, and promotion-policy tests. |
| `tests/test_experiment_rendering.py` | Standalone contract, size, import, model-state, and optimizer tests. |
| `tests/test_holomorphic_components.py` | Shape, gradient, Cauchy-Riemann, autonomy, and stability tests. |
| `tests/test_experiment_orchestration.py` | Fake-subprocess tests for result parsing, failure capture, SHA pinning, and Colab job cleanup behavior. |
| `.gitignore` | Ignore generated submissions and `artifacts/one_layer_deeper/`, while keeping experiment source/specs tracked. |

## 6. Implementation tasks

### Task 1: Lock the protocol and typed matrix

**Files:**
- Create: `experiments/protocol.md`
- Create: `experiments/matrix.py`
- Create: `tests/test_experiment_matrix.py`

- [ ] **Step 1: Write failing tests for unique IDs, legal statuses, and exact stage membership**

```python
def test_e1_screen_contains_all_controls_once():
    ids = [trial.candidate_id for trial in matrix("e1_mechanism_screen")]
    assert ids == ["B0", "R1", "R2", "S1", "S2", "Q1", "Q2", "Q3", "Q4"]

def test_official_matrices_exclude_non_candidate_legal_statuses():
    for name in official_matrix_names():
        assert {trial.legality for trial in matrix(name)} == {"candidate-safe"}
```

- [ ] **Step 2: Run the tests and verify they fail because the module does not exist**

Run: `uv run python -m unittest tests.test_experiment_matrix -v`

- [ ] **Step 3: Implement frozen `CandidateSpec`, `TrialSpec`, and named matrices**

Use `Literal` legality values `candidate-safe`, `research-only`, and `needs-clarification`. Encode all constants from Sections 2-3, including `L_train`, `L_eval`, width, optimizer, loss weights, dataset, budget class, and reference candidate.

- [ ] **Step 4: Run the focused test and the repository suite**

Run: `uv run python -m unittest tests.test_experiment_matrix -v`

Run: `uv run python -m unittest discover -s tests`

- [ ] **Step 5: Commit**

```bash
git add experiments/protocol.md experiments/matrix.py tests/test_experiment_matrix.py
git commit -m "test: define holomorphic architecture experiment matrix"
```

### Task 2: Render and validate standalone candidates

**Files:**
- Create: `experiments/render.py`
- Create: `experiments/templates/control_submission.py.tmpl`
- Create: `experiments/templates/holomorphic_submission.py.tmpl`
- Create: `experiments/templates/koopman_submission.py.tmpl`
- Create: `tests/test_experiment_rendering.py`

- [ ] **Step 1: Write tests that render every candidate and enforce the competition contract**

For every `CandidateSpec`, assert UTF-8 size `<= 256 * 1024`, filename `submission.py`, successful `validate_submission_source`, exactly one exported `SUBMISSION`, no imports from `data`, `model`, `optim`, or `experiments`, and deterministic source SHA for the same spec.

- [ ] **Step 2: Verify failure before renderer implementation**

Run: `uv run python -m unittest tests.test_experiment_rendering -v`

- [ ] **Step 3: Implement literal-config rendering**

Each template contains one sentinel `EXPERIMENT_CONFIG = __EXPERIMENT_CONFIG__`. Replace it with a sorted Python literal, reject any remaining sentinel, validate source before writing, and place generated files under `artifacts/one_layer_deeper/rendered/<candidate-id>/<sha256>/submission.py`.

- [ ] **Step 4: Add subprocess import validation**

Import each rendered file in a fresh process using the current `benchmark` package, build a smoke model, call `assert_model_state`, build the optimizer, and verify every trainable parameter appears exactly once.

- [ ] **Step 5: Run tests and commit**

Run: `uv run python -m unittest tests.test_experiment_rendering -v`

```bash
git add experiments/render.py experiments/templates tests/test_experiment_rendering.py
git commit -m "feat: render standalone experiment submissions"
```

### Task 3: Implement the matched control family

**Files:**
- Modify: `experiments/templates/control_submission.py.tmpl`
- Modify: `tests/test_experiment_rendering.py`

- [ ] **Step 1: Add failing shape/state/parameter-sharing tests for B0/R1/R2/S1/Q1**

Assert logits `[B,S,V]`, scalar finite training loss, R1 block parameter identities reused across loops, R2 distinct block parameters, and no loop index/T argument on recurrent transitions.

- [ ] **Step 2: Implement the shared real encoder/readout and each control transition**

Use the published token/position embedding and bidirectional attention structure. R1 uses tied GELU; R2 uses untied blocks; S1 uses learned linear mixing followed by `sin` and `cos`; Q1 learns constant, linear, and quadratic coefficients. All use the same residual scale and learned trajectory readout.

- [ ] **Step 3: Run CPU smoke for every control**

Run each rendered candidate with `benchmark/manifests/smoke_cpu.json`; require a `RESULT_JSON` and finite loss, but do not compare smoke scores.

- [ ] **Step 4: Commit**

```bash
git add experiments/templates/control_submission.py.tmpl tests/test_experiment_rendering.py
git commit -m "feat: add matched recurrent control architectures"
```

### Task 4: Implement and verify the holomorphic residue loop

**Files:**
- Modify: `experiments/templates/holomorphic_submission.py.tmpl`
- Create: `tests/test_holomorphic_components.py`

- [ ] **Step 1: Write failing tests for paired-real complex algebra**

Numerically compare `ComplexLinear` and elementwise multiplication with Python complex64 references in float32. Hold context fixed and verify the predictor's finite-difference Wirtinger derivative with respect to conjugate input is below `1e-4`; verify Q3's conjugate control is above `1e-3`.

- [ ] **Step 2: Implement paired-real complex primitives**

```python
def complex_linear(xr, xi, wr, wi, br, bi):
    yr = F.linear(xr, wr, br) - F.linear(xi, wi)
    yi = F.linear(xr, wi, bi) + F.linear(xi, wr)
    return yr, yi

def complex_square(xr, xi):
    return xr.square() - xi.square(), 2.0 * xr * xi
```

All polynomial coefficients and mixing matrices are randomly initialized trainable parameters. `complex_square` is only a basis operation inside the unrestricted learned polynomial; no coefficient is fixed to select it as the transition law.

- [ ] **Step 3: Implement Q2/Q3/Q4**

Q2 computes learned `a0 + a1*z + a2*z²` after a learned complex-linear projection. Q3 adds parameter-matched conjugate terms. Q4 passes Q2's prediction through one small real-valued residual refiner with context FiLM. Keep normalization/refinement outside the holomorphic predictor.

- [ ] **Step 4: Implement learned prompt pooling and trajectory readout**

Use learned queries over all prompt token states to produce `initial_state`, `persistent_context`, and `readout_query`; do not hard-code field token IDs or positions. Stack all recurrent states and let the readout query cross-attend to the trajectory. No transition method accepts T, step index, or input IDs.

- [ ] **Step 5: Test long rollout stability and gradients**

In float32 and CUDA BF16 smoke, run 64 evaluation transitions; require finite states/logits, finite gradients in training mode, and successful evaluation within the smoke deadline.

- [ ] **Step 6: Commit**

```bash
git add experiments/templates/holomorphic_submission.py.tmpl tests/test_holomorphic_components.py
git commit -m "feat: add holomorphic residue loop candidates"
```

### Task 5: Add exact-match-aware auxiliary losses

**Files:**
- Modify: `experiments/templates/control_submission.py.tmpl`
- Modify: `experiments/templates/holomorphic_submission.py.tmpl`
- Modify: `tests/test_holomorphic_components.py`

- [ ] **Step 1: Write masking and gradient tests**

Construct padded `TokenLossBatch` values and assert invalid slots have zero influence, each loss returns one finite differentiable scalar, the smooth maximum approaches the worst valid digit as temperature decreases, and auxiliary penalties backpropagate only when their configured weight is nonzero.

- [ ] **Step 2: Implement the loss family**

```python
token_ce = F.cross_entropy(
    batch.logits.transpose(1, 2),
    batch.labels,
    ignore_index=-100,
    reduction="none",
)
masked = token_ce.masked_fill(~batch.valid_mask, float("-inf"))
target_counts = batch.valid_mask.sum(dim=1)
has_targets = target_counts > 0
seq_loss = tau_token * (
    torch.logsumexp(masked[has_targets] / tau_token, dim=1)
    - target_counts[has_targets].log()
)
task_loss = tau_batch * torch.logsumexp(seq_loss / tau_batch, dim=0)
task_loss = task_loss - tau_batch * math.log(seq_loss.numel())
return task_loss + sum(weight * batch.auxiliary[name] for name, weight in weights.items())
```

Supply full-prompt reconstruction, phase/amplitude, transition-denoising, and composition penalties as scalar tensors in `auxiliary`. Never parse or supervise intermediate modular answers.

- [ ] **Step 3: Run token-loss contract tests and commit**

Run: `uv run python -m unittest tests.test_token_training_loss tests.test_holomorphic_components -v`

```bash
git add experiments/templates tests/test_holomorphic_components.py
git commit -m "feat: add sequence-level and auxiliary experiment losses"
```

### Task 6: Build local runner, immutable records, and reports

**Files:**
- Create: `experiments/run_local.py`
- Create: `experiments/report.py`
- Create: `tests/test_experiment_orchestration.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write fake-runner tests**

Cover successful `RESULT_JSON` parsing, missing result marker, nonzero exit, timeout, OOM text, malformed/non-finite JSON, immutable hashes, paired hardware grouping, first-uncertified-rung extraction, and lexicographic promotion ordering.

- [ ] **Step 2: Implement one-subprocess-per-trial execution**

The runner accepts a named matrix, renders candidates, verifies dataset presence, invokes the unchanged benchmark runner, stores stdout/stderr and one `RunRecord`, and never edits official manifests. Research screens use generated manifests under `artifacts/one_layer_deeper/research_manifests/` with an explicit `comparable=false` field in their surrounding record.

- [ ] **Step 3: Implement reporting**

Group only identical accelerator/GPU name, repo SHA, manifest SHA, dataset SHA, seed, and budget. Produce promotion tuple, paired delta from the named reference, completed-step ratio, and evaluation-time ratio. Label cross-hardware rows as non-comparable.

- [ ] **Step 4: Run tests and commit**

```bash
uv run python -m unittest tests.test_experiment_orchestration -v
git add experiments/run_local.py experiments/report.py tests/test_experiment_orchestration.py .gitignore
git commit -m "feat: add reproducible experiment runner and reports"
```

### Task 7: Build the self-contained Colab one-shot job

**Files:**
- Create: `experiments/build_colab_job.py`
- Modify: `tests/test_experiment_orchestration.py`

- [ ] **Step 1: Write static job-payload tests**

Assert the payload pins the benchmark SHA, embeds submission bytes and hashes, uses `/content/one-layer-deeper`, generates only named datasets, passes an explicit subprocess timeout, emits JSONL, never prints credentials/environment wholesale, never uses `--keep`, and preserves per-candidate failures.

- [ ] **Step 2: Implement preflight and sweep payload modes**

Preflight prints only Python/Torch/CUDA/GPU/VRAM metadata. Sweep mode performs the nine remote steps in Section 4 and includes a final summary count of passed, failed, OOM, and timed-out trials.

- [ ] **Step 3: Run a local syntax-only build**

```bash
uv run python experiments/build_colab_job.py --mode preflight --output /tmp/old-colab-preflight.py
uv run python -m py_compile /tmp/old-colab-preflight.py
```

- [ ] **Step 4: Run the approved CPU Colab preflight, verify teardown, then commit**

Use `colab run` without `--gpu`, a named session, and `--timeout 300`. Require exit code zero and `colab sessions` to show no new assignment.

```bash
git add experiments/build_colab_job.py tests/test_experiment_orchestration.py
git commit -m "feat: add self-cleaning Colab experiment jobs"
```

### Task 8: Calibrate B0 before architecture sweeps

**Files:**
- Generated only: `artifacts/one_layer_deeper/**`

- [ ] **Step 1: Record Mac contract smoke**

Run B0 with `smoke_cpu.json`; store the record with `score_class=contract-smoke`.

- [ ] **Step 2: Run approved H100 preflight**

Record exact GPU model/VRAM and software versions. If allocation fails, stop and report without changing accelerator.

- [ ] **Step 3: Run B0 on exact E1/E3/E5 manifests on that accelerator**

Require complete final evaluation and retain the raw `RESULT_JSON` and logs.

- [ ] **Step 4: Submit the identical B0 source to hosted E1/E3/E5**

Use the competition CLI only after login; record submission IDs and downloaded metrics. These three hosted results become the competition baseline and quantify Colab drift.

- [ ] **Step 5: Produce `baseline-report.md`**

Report all score components, completed updates, evaluation time, state elements, source hash, hardware, and whether Colab and hosted results are comparable or only directionally aligned.

### Task 9: Execute the staged architecture program

**Files:**
- Generated only: `artifacts/one_layer_deeper/**`

- [ ] **Step 1:** Run and report the E1 mechanism screen.
- [ ] **Step 2:** Confirm the top four across seeds 74/75/76 and exact E1.
- [ ] **Step 3:** Run E3 persistent-context and x-reinjection ablations.
- [ ] **Step 4:** Run E5 learned-readout/depth ablations plus the research-only oracle diagnostic.
- [ ] **Step 5:** Tune stability, auxiliary loss, then optimizer knobs one family at a time.
- [ ] **Step 6:** Implement/run Koopman and teacher-student candidates under the same promotion protocol.
- [ ] **Step 7:** Run all Easy datasets and M1/M3/M5; use M2/M4 as scale replications.
- [ ] **Step 8:** Obtain organizer answers for every `needs-clarification` mechanism and render a candidate-safe finalist.
- [ ] **Step 9:** Run hosted Easy/Medium confirmation and apply the Hard gate exactly as written.

### Task 10: Self-review and final acceptance

- [ ] **Step 1: Run all repository tests and standalone validation**

```bash
uv run python -m unittest discover -s tests
one-layer validate artifacts/one_layer_deeper/finalist/submission.py
```

- [ ] **Step 2: Check source policy and artifact integrity**

Verify the finalist is at most 256 KiB, imports only allowed dependencies, contains no research-only/oracle branches, hashes to the source used for final Colab/hosted confirmation, and includes all parameters exactly once in its optimizer.

- [ ] **Step 3: Check experimental claims**

Every conclusion must name the paired control, identical hardware/manifests/seeds, replication count, score tuple, and throughput/evaluation tradeoff. Describe cross-hardware observations as directional.

- [ ] **Step 4: Check Colab cleanup**

Run `colab --auth=adc sessions`; report zero experiment sessions or stop each named experiment session before completion.

- [ ] **Step 5: Commit the experiment framework and protocol only**

Do not commit generated submissions, datasets, logs, credentials, or run artifacts.

## 7. Acceptance criteria

- The published baseline and every candidate run through the unchanged official runner.
- Every score is labeled contract-smoke, research-screen, tier-faithful, or hosted-authoritative.
- Every official candidate is one self-contained validated file under 256 KiB and the 500M-state ceiling.
- The experimental design can separately falsify recurrence, periodicity, analyticity, holomorphicity, persistent context, local correction, learned readout, fast operator composition, and exact-match loss claims.
- At least three research seeds confirm every promoted mechanism; exact official manifests confirm every stage winner.
- Colab jobs pin code/data/submission hashes, set an explicit timeout, use named sessions, avoid silent accelerator fallback, and leave no active VM.
- No research-only or unresolved rule-risk mechanism reaches hosted submission.
- The final report includes raw score components, paired deltas, compute/throughput, failure modes, and enough metadata to reproduce every promoted run.
