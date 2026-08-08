"""Frozen experiment registry and named comparison matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Eligibility = Literal["candidate-safe", "research-only", "needs-clarification"]
Entrypoint = Literal["submission.py", "research.py"]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    version: str
    title: str
    parent: str | None
    eligibility: Eligibility
    entrypoint: Entrypoint
    family: str
    description: str

    @property
    def is_candidate(self) -> bool:
        return self.eligibility == "candidate-safe"

    def directory(self, root: Path | None = None) -> Path:
        base = root or Path(__file__).resolve().parents[1]
        return base / self.experiment_id

    def source_path(self, root: Path | None = None) -> Path:
        return self.directory(root) / self.entrypoint


def _spec(
    experiment_id: str,
    version: str,
    title: str,
    parent: str | None,
    eligibility: Eligibility,
    entrypoint: Entrypoint,
    family: str,
    description: str,
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        version=version,
        title=title,
        parent=parent,
        eligibility=eligibility,
        entrypoint=entrypoint,
        family=family,
        description=description,
    )


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    _spec("v0_baseline_adamw", "v0", "Published AdamW baseline", None, "candidate-safe", "submission.py", "baseline", "Exact published single-pass Transformer baseline."),
    _spec("v1_tied_gelu", "v1", "Tied GELU recurrence", "v0_baseline_adamw", "candidate-safe", "submission.py", "recurrence", "Shared GELU transition with learned trajectory readout."),
    _spec("v1.1_untied_gelu_control", "v1.1", "Untied GELU control", "v1_tied_gelu", "candidate-safe", "submission.py", "recurrence", "Compute-matched untied control."),
    _spec("v1.2_persistent_context", "v1.2", "Persistent context", "v1_tied_gelu", "candidate-safe", "submission.py", "recurrence", "Persistent learned modulus context."),
    _spec("v1.3_exact_t_diagnostic", "v1.3", "Exact-T recurrence diagnostic", "v1.2_persistent_context", "research-only", "research.py", "diagnostic", "GPU-side exact-T masked recurrence diagnostic."),
    _spec("v1.4_universal_transformer", "v1.4", "Fixed-depth Universal Transformer", "v0_baseline_adamw", "candidate-safe", "submission.py", "recurrence", "Shared full-sequence attention and feed-forward block with fixed recurrent-time encoding."),
    _spec("v2_real_sine", "v2", "Real sine transition", "v1.2_persistent_context", "candidate-safe", "submission.py", "function-class", "Real periodic recurrent transition."),
    _spec("v2.1_paired_complex_sine", "v2.1", "Paired-complex sine", "v1.2_persistent_context", "candidate-safe", "submission.py", "function-class", "Paired-real complex entire-function transition."),
    _spec("v2.2_real_bilinear", "v2.2", "Real bilinear transition", "v1.2_persistent_context", "candidate-safe", "submission.py", "function-class", "Learned real multiplicative transition."),
    _spec("v2.3_holomorphic_bilinear", "v2.3", "Holomorphic bilinear transition", "v1.2_persistent_context", "candidate-safe", "submission.py", "function-class", "Learned complex bilinear transition without conjugate dependence."),
    _spec("v2.4_conjugate_bilinear_control", "v2.4", "Conjugate bilinear control", "v2.3_holomorphic_bilinear", "candidate-safe", "submission.py", "function-class", "Parameter-matched conjugate-dependent control."),
    _spec("v3_factored_orbit_workspace", "v3", "Factored orbit and workspace", "v2.3_holomorphic_bilinear", "candidate-safe", "submission.py", "factored-state", "Separate persistent orbit and token workspace."),
    _spec("v3.1_damped_correction", "v3.1", "Damped correction state", "v3_factored_orbit_workspace", "candidate-safe", "submission.py", "factored-state", "Adds damped oscillatory correction channels."),
    _spec("v3.2_local_state_refiner", "v3.2", "Local state refiner", "v3.1_damped_correction", "candidate-safe", "submission.py", "factored-state", "Adds one learned real post-transition refiner."),
    _spec("v3.3_gabor_corrector", "v3.3", "Gabor corrector", "v3.1_damped_correction", "candidate-safe", "submission.py", "factored-state", "Uses a localized periodic activation only in the corrector."),
    _spec("v3.4_factored_exact_t_diagnostic", "v3.4", "Factored exact-T diagnostic", "v3.1_damped_correction", "research-only", "research.py", "diagnostic", "Exact-T diagnostic for the factored model."),
    _spec("v4_attractor_picard_corrector", "v4", "Picard attractor corrector", "v3.1_damped_correction", "candidate-safe", "submission.py", "attractor", "Explicit Picard refinement of the correction state."),
    _spec("v4.1_attractor_anderson_corrector", "v4.1", "Anderson attractor corrector", "v4_attractor_picard_corrector", "candidate-safe", "submission.py", "attractor", "Fixed-memory Anderson acceleration of the correction state."),
    _spec("v4.2_attractor_implicit_gradients_research", "v4.2", "Implicit-gradient attractor research", "v4_attractor_picard_corrector", "research-only", "research.py", "attractor-research", "Compares explicit, one-step, phantom, and implicit gradients."),
    _spec("v4.3_equilibrium_internalization_research", "v4.3", "Equilibrium internalization", "v4.2_attractor_implicit_gradients_research", "research-only", "research.py", "attractor-research", "Compares proposal-only, one-step, and solved-equilibrium inference."),
    _spec("v4.4_discrete_cycle_solver_research", "v4.4", "Discrete cycle solver", "v4.2_attractor_implicit_gradients_research", "research-only", "research.py", "attractor-research", "Solves finite invariant cycles jointly."),
    _spec("v4.5_analytic_invariant_orbit_research", "v4.5", "Analytic invariant orbit", "v4.4_discrete_cycle_solver_research", "research-only", "research.py", "attractor-research", "Solves a phase-indexed Fourier/Laurent invariant orbit."),
    _spec("v5_sequence_smoothmax_loss", "v5", "Sequence smooth-max loss", "v3.1_damped_correction", "candidate-safe", "submission.py", "loss", "Exact-match-aware smooth maximum loss."),
    _spec("v5.1_prompt_reconstruction", "v5.1", "Prompt reconstruction", "v5_sequence_smoothmax_loss", "candidate-safe", "submission.py", "loss", "Adds prompt reconstruction auxiliary loss."),
    _spec("v5.2_composition_stability", "v5.2", "Composition stability", "v5.1_prompt_reconstruction", "candidate-safe", "submission.py", "loss", "Adds unsupervised composition and norm-growth penalties."),
)

_BY_ID = {spec.experiment_id: spec for spec in EXPERIMENTS}

if len(_BY_ID) != len(EXPERIMENTS):
    raise RuntimeError("experiment IDs must be unique")

for _experiment in EXPERIMENTS:
    if not _experiment.experiment_id.startswith(_experiment.version + "_"):
        raise RuntimeError(f"invalid version prefix: {_experiment.experiment_id}")
    if _experiment.parent is not None and _experiment.parent not in _BY_ID:
        raise RuntimeError(f"unknown parent for {_experiment.experiment_id}")


MATRICES: dict[str, tuple[str, ...]] = {
    "candidate-safe": tuple(spec.experiment_id for spec in EXPERIMENTS if spec.is_candidate),
    "e1-full": tuple(spec.experiment_id for spec in EXPERIMENTS if spec.is_candidate),
    "exact-t-diagnostics": (
        "v1.3_exact_t_diagnostic",
        "v3.4_factored_exact_t_diagnostic",
    ),
    "attractor-research": tuple(
        spec.experiment_id
        for spec in EXPERIMENTS
        if spec.family == "attractor-research"
    ),
}


def get_experiment(experiment_id: str) -> ExperimentSpec:
    try:
        return _BY_ID[experiment_id]
    except KeyError as exc:
        choices = ", ".join(_BY_ID)
        raise KeyError(f"unknown experiment {experiment_id!r}; choose from: {choices}") from exc


def get_matrix(name: str) -> tuple[ExperimentSpec, ...]:
    try:
        ids = MATRICES[name]
    except KeyError as exc:
        raise KeyError(f"unknown matrix {name!r}; choose from: {', '.join(MATRICES)}") from exc
    return tuple(get_experiment(experiment_id) for experiment_id in ids)
