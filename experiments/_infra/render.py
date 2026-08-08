"""Render canonical tracked candidates into immutable hash-addressed copies."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

from .registry import ExperimentSpec
from .validate import validate_experiment


def render_candidate(
    spec: ExperimentSpec,
    *,
    output_root: Path,
    experiment_root: Path | None = None,
    official: bool = True,
) -> tuple[Path, str]:
    if official and not spec.is_candidate:
        raise ValueError(
            f"{spec.experiment_id} is {spec.eligibility} and cannot render in official mode"
        )
    result = validate_experiment(spec, root=experiment_root, official=official)
    if not result.valid:
        raise ValueError(
            f"invalid experiment {spec.experiment_id}: " + "; ".join(result.errors)
        )
    source_path = spec.source_path(experiment_root)
    source = source_path.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    destination = output_root / spec.experiment_id / source_hash / source_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != source:
        raise FileExistsError(f"immutable render collision at {destination}")
    if not destination.exists():
        shutil.copyfile(source_path, destination)
    return destination, source_hash
