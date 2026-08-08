"""Execute canonical candidates with the unchanged benchmark runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
from typing import Any, Literal

from .registry import ExperimentSpec
from .render import render_candidate
from .validate import is_exact_public_h100_manifest


ScoreClass = Literal[
    "contract-smoke",
    "research-screen",
    "tier-faithful",
    "hosted-authoritative",
]


@dataclass(frozen=True)
class RunRecord:
    schema_version: int
    run_id: str
    candidate_id: str
    parent_id: str | None
    eligibility: str
    score_class: ScoreClass
    comparable: bool
    repository_sha: str
    git_commit: str
    git_dirty: bool
    source_sha256: str
    submission_sha256: str
    manifest_sha256: str
    dataset_config_sha256: str | None
    requested_accelerator: str | None
    accelerator: str
    gpu_name: str | None
    gpu_memory_bytes: int | None
    python_version: str
    torch_version: str | None
    cuda_version: str | None
    driver_version: str | None
    bf16_supported: bool | None
    started_at_utc: str
    finished_at_utc: str
    elapsed_seconds: float
    seeds: tuple[int, ...]
    completed_steps: int
    training_seconds: float
    evaluation_seconds: float
    attractor_solver_metrics: dict[str, Any] | None
    command: tuple[str, ...]
    result_json: dict[str, Any] | None
    failure: dict[str, Any] | None
    stdout_path: str
    stderr_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return commit, bool(status.strip())


def _dataset_hash(manifest_path: Path, repo_root: Path) -> str | None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = payload.get("data", {}).get("data_root")
    if not data_root:
        return None
    root = repo_root / data_root
    for name in ("config.json", "dataset_config.json"):
        path = root / name
        if path.is_file():
            return _sha256(path)
    return None


def _hardware() -> dict[str, Any]:
    data: dict[str, Any] = {
        "accelerator": "cpu",
        "gpu_name": None,
        "gpu_memory_bytes": None,
        "torch_version": None,
        "cuda_version": None,
        "driver_version": None,
        "bf16_supported": False,
    }
    try:
        import torch
    except ImportError:
        return data
    data["torch_version"] = torch.__version__
    data["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        data.update(
            accelerator="cuda",
            gpu_name=properties.name,
            gpu_memory_bytes=properties.total_memory,
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
        )
        try:
            data["driver_version"] = str(torch._C._cuda_getDriverVersion())
        except (AttributeError, RuntimeError):
            pass
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        data["accelerator"] = "mps"
    return data


def _parse_result(stdout: str) -> dict[str, Any]:
    markers = [
        line.removeprefix("RESULT_JSON=")
        for line in stdout.splitlines()
        if line.startswith("RESULT_JSON=")
    ]
    if not markers:
        raise ValueError("benchmark output did not contain RESULT_JSON")
    return json.loads(markers[-1])


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _run_totals(result: dict[str, Any] | None) -> tuple[tuple[int, ...], int, float, float]:
    seed_results = (result or {}).get("seeds", [])
    return (
        tuple(int(seed["seed"]) for seed in seed_results if "seed" in seed),
        sum(int(seed.get("completed_training_steps") or 0) for seed in seed_results),
        sum(float(seed.get("training_seconds") or 0.0) for seed in seed_results),
        sum(float(seed.get("evaluation_seconds") or 0.0) for seed in seed_results),
    )


def _attractor_solver_metrics(spec: ExperimentSpec) -> dict[str, Any] | None:
    metadata = json.loads(
        (spec.directory() / "experiment.json").read_text(encoding="utf-8")
    )
    solver = metadata.get("solver")
    return dict(solver) if isinstance(solver, dict) else None


def run_candidate(
    spec: ExperimentSpec,
    *,
    manifest_path: Path,
    repo_root: Path,
    artifact_root: Path,
    score_class: ScoreClass,
    requested_accelerator: str | None = None,
    timeout_seconds: float | None = None,
    comparable: bool = False,
) -> RunRecord:
    if not spec.is_candidate:
        raise ValueError(f"{spec.experiment_id} is not candidate-safe")
    if score_class == "hosted-authoritative":
        raise ValueError("hosted-authoritative records cannot be created by run_local")
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    commit, dirty = _git_state(repo_root)
    hardware = _hardware()
    actual_name = str(hardware.get("gpu_name") or hardware["accelerator"])
    if requested_accelerator:
        expected = requested_accelerator.upper()
        if expected == "CPU":
            matched = hardware["accelerator"] == "cpu"
        else:
            matched = expected in actual_name.upper()
        if not matched:
            raise RuntimeError(
                f"requested accelerator {requested_accelerator}, found {actual_name}"
            )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest_payload.get("runtime", {}).get("dtype") == "bfloat16"
        and not hardware.get("bf16_supported")
    ):
        raise ValueError("manifest requires BF16 but the active runtime does not support it")
    if score_class == "tier-faithful":
        if "H100" not in actual_name.upper():
            raise ValueError(
                "tier-faithful is reserved for exact H100 runs; use research-screen"
            )
        if not is_exact_public_h100_manifest(
            manifest_path,
            repo_root=repo_root,
            repository_sha=commit,
        ):
            raise ValueError(
                "tier-faithful requires an unmodified public H100 manifest from "
                "the recorded repository revision"
            )
    rendered, source_hash = render_candidate(
        spec,
        output_root=artifact_root / "rendered",
        official=True,
    )
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:10]
    )
    run_dir = artifact_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command = (
        sys.executable,
        "-m",
        "benchmark.runner",
        "--manifest",
        str(manifest_path),
        "--submission-file",
        str(rendered),
    )
    started = datetime.now(timezone.utc)
    monotonic_start = time.monotonic()
    result_json: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
        stdout, stderr = completed.stdout, completed.stderr
        if completed.returncode != 0:
            kind = (
                "oom"
                if "out of memory" in stderr.lower()
                or "cuda error: out of memory" in stderr.lower()
                else "nonzero-exit"
            )
            failure = {
                "kind": kind,
                "returncode": completed.returncode,
            }
        else:
            try:
                result_json = _parse_result(stdout)
            except (ValueError, json.JSONDecodeError) as exc:
                failure = {"kind": "missing-or-invalid-result", "message": str(exc)}
    except subprocess.TimeoutExpired as exc:
        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr)
        failure = {"kind": "timeout", "timeout_seconds": timeout_seconds}
    finished = datetime.now(timezone.utc)
    elapsed = time.monotonic() - monotonic_start
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    seeds, completed_steps, training_seconds, evaluation_seconds = _run_totals(
        result_json
    )
    record = RunRecord(
        schema_version=1,
        run_id=run_id,
        candidate_id=spec.experiment_id,
        parent_id=spec.parent,
        eligibility=spec.eligibility,
        score_class=score_class,
        comparable=comparable,
        repository_sha=commit,
        git_commit=commit,
        git_dirty=dirty,
        source_sha256=source_hash,
        submission_sha256=source_hash,
        manifest_sha256=_sha256(manifest_path),
        dataset_config_sha256=_dataset_hash(manifest_path, repo_root),
        requested_accelerator=requested_accelerator,
        accelerator=hardware["accelerator"],
        gpu_name=hardware["gpu_name"],
        gpu_memory_bytes=hardware["gpu_memory_bytes"],
        python_version=platform.python_version(),
        torch_version=hardware["torch_version"],
        cuda_version=hardware["cuda_version"],
        driver_version=hardware["driver_version"],
        bf16_supported=hardware["bf16_supported"],
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        elapsed_seconds=elapsed,
        seeds=seeds,
        completed_steps=completed_steps,
        training_seconds=training_seconds,
        evaluation_seconds=evaluation_seconds,
        attractor_solver_metrics=_attractor_solver_metrics(spec),
        command=command,
        result_json=result_json,
        failure=failure,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )
    record_path = run_dir / "record.json"
    with record_path.open("x", encoding="utf-8") as handle:
        json.dump(asdict(record), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return record
