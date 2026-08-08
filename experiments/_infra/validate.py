"""Static validation for experiment metadata and standalone submissions."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from submission_validation import validate_submission_source

from .registry import EXPERIMENTS, ExperimentSpec


MAX_SUBMISSION_BYTES = 256 * 1024
REQUIRED_METADATA = {
    "id",
    "version",
    "parent",
    "title",
    "eligibility",
    "entrypoint",
    "family",
    "description",
}


def is_exact_public_h100_manifest(
    manifest_path: Path,
    *,
    repo_root: Path,
    repository_sha: str,
) -> bool:
    """Return whether ``manifest_path`` matches its pinned public Git blob.

    A path or filename alone is not evidence that a manifest is competition
    faithful: a dirty worktree can modify a public manifest while preserving
    both.  Comparing with the blob at the recorded repository revision keeps
    the evidence label tied to immutable evaluator-owned contents.
    """

    root = repo_root.resolve()
    path = manifest_path.resolve()
    if not path.is_file():
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if relative.parent != Path("benchmark/manifests"):
        return False
    if not relative.name.startswith("h100_") or relative.suffix != ".json":
        return False
    try:
        tracked = subprocess.run(
            ["git", "show", f"{repository_sha}:{relative.as_posix()}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    return hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(tracked).digest()


@dataclass(frozen=True)
class ValidationResult:
    experiment_id: str
    valid: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    source_bytes: int | None = None


def _load_metadata(spec: ExperimentSpec, root: Path | None) -> tuple[dict, list[str]]:
    errors: list[str] = []
    path = spec.directory(root) / "experiment.json"
    if not path.is_file():
        return {}, ["missing experiment.json"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"invalid experiment.json: {exc}"]
    missing = REQUIRED_METADATA - payload.keys()
    if missing:
        errors.append(f"experiment.json missing fields: {sorted(missing)}")
    expected = {
        "id": spec.experiment_id,
        "version": spec.version,
        "parent": spec.parent,
        "eligibility": spec.eligibility,
        "entrypoint": spec.entrypoint,
        "family": spec.family,
    }
    for key, value in expected.items():
        if key in payload and payload[key] != value:
            errors.append(
                f"experiment.json {key}={payload[key]!r}; expected {value!r}"
            )
    return payload, errors


class _CandidatePolicyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "pow":
            self.errors.append(f"line {node.lineno}: builtin pow is disallowed")
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "backward":
                self.errors.append(
                    f"line {node.lineno}: participant-controlled backward is disallowed"
                )
            if node.func.attr == "grad" and isinstance(node.func.value, ast.Attribute):
                if node.func.value.attr == "autograd":
                    self.errors.append(
                        f"line {node.lineno}: torch.autograd.grad is disallowed"
                    )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mod):
            self.errors.append(
                f"line {node.lineno}: modulo operations are disallowed in candidates"
            )
        self.generic_visit(node)


def _validate_candidate_source(spec: ExperimentSpec, root: Path | None) -> tuple[list[str], list[str], int | None]:
    errors: list[str] = []
    warnings: list[str] = []
    source_path = spec.source_path(root)
    if not source_path.is_file():
        return [f"missing {spec.entrypoint}"], warnings, None
    source = source_path.read_text(encoding="utf-8")
    source_bytes = len(source.encode("utf-8"))
    try:
        validate_submission_source(
            source_path.name,
            source,
            MAX_SUBMISSION_BYTES,
            required_filename="submission.py",
        )
    except ValueError as exc:
        errors.append(str(exc))
        return errors, warnings, source_bytes
    tree = ast.parse(source, filename=str(source_path))
    visitor = _CandidatePolicyVisitor()
    visitor.visit(tree)
    errors.extend(visitor.errors)
    exported = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "SUBMISSION"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
    ]
    if len(exported) != 1:
        errors.append("candidate must define SUBMISSION exactly once at module scope")
    lowered = source.lower()
    for phrase in ("discrete_log", "factorization", "residue_table"):
        if phrase in lowered:
            errors.append(f"candidate contains prohibited task-solver marker: {phrase}")
    if (spec.directory(root) / "BLOCKED_FROM_SUBMISSION").exists():
        errors.append("candidate-safe experiment must not be blocked from submission")
    return errors, warnings, source_bytes


def validate_experiment(
    spec: ExperimentSpec,
    *,
    root: Path | None = None,
    official: bool = False,
) -> ValidationResult:
    directory = spec.directory(root)
    errors: list[str] = []
    warnings: list[str] = []
    if not directory.is_dir():
        return ValidationResult(spec.experiment_id, False, ("missing directory",))
    if not (directory / "README.md").is_file():
        errors.append("missing README.md")
    _, metadata_errors = _load_metadata(spec, root)
    errors.extend(metadata_errors)

    source_bytes: int | None = None
    if spec.is_candidate:
        source_errors, source_warnings, source_bytes = _validate_candidate_source(
            spec, root
        )
        errors.extend(source_errors)
        warnings.extend(source_warnings)
    else:
        if official:
            errors.append(
                f"{spec.eligibility} experiment is mechanically blocked from official mode"
            )
        if not (directory / "BLOCKED_FROM_SUBMISSION").is_file():
            errors.append("research experiment missing BLOCKED_FROM_SUBMISSION")
        if not spec.source_path(root).is_file():
            errors.append(f"missing {spec.entrypoint}")
        elif spec.entrypoint == "research.py":
            try:
                ast.parse(
                    spec.source_path(root).read_text(encoding="utf-8"),
                    filename=str(spec.source_path(root)),
                )
            except SyntaxError as exc:
                errors.append(f"invalid research.py: {exc}")

    return ValidationResult(
        experiment_id=spec.experiment_id,
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        source_bytes=source_bytes,
    )


def validate_all(
    specs: Iterable[ExperimentSpec] = EXPERIMENTS,
    *,
    root: Path | None = None,
    official: bool = False,
) -> tuple[ValidationResult, ...]:
    return tuple(
        validate_experiment(spec, root=root, official=official) for spec in specs
    )
