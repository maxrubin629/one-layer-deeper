"""Build self-contained one-shot Colab jobs without allocating a session."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Literal

from .registry import ExperimentSpec
from .validate import is_exact_public_h100_manifest, validate_experiment


JobMode = Literal["preflight", "environment-preflight", "sweep", "research"]
DEFAULT_REPOSITORY_URL = "https://github.com/tilde-research/one-layer-deeper.git"
RUNTIME_PROBE = r'''from __future__ import annotations
import json
import platform
import subprocess

import torch

data = {
    "accelerator": "cpu",
    "gpu_name": None,
    "gpu_memory_bytes": None,
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "driver_version": None,
    "bf16_supported": False,
}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    data.update(
        accelerator="cuda",
        gpu_name=props.name,
        gpu_memory_bytes=props.total_memory,
        bf16_supported=bool(torch.cuda.is_bf16_supported()),
    )
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        data["driver_version"] = completed.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
print(json.dumps(data, sort_keys=True))
'''


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _dataset_command(repo_root: Path, data_root: str | None) -> str | None:
    if not data_root:
        return None
    script = (repo_root / "scripts" / "generate_datasets.sh").read_text(
        encoding="utf-8"
    )
    blocks = re.findall(
        r"python -m data\.squaring_mod \\\n(?:  .*?(?:\\\n|\n))+?(?=\n(?:#|python)|\Z)",
        script,
        flags=re.MULTILINE,
    )
    for block in blocks:
        if f"--output_dir {data_root}" in block:
            command = block.rstrip("\\\n \n")
            return command.replace(
                "python -m data.squaring_mod",
                "uv run --python 3.13.5 python -m data.squaring_mod",
                1,
            )
    raise ValueError(f"no generation command found for {data_root}")


def _encode_source(spec: ExperimentSpec) -> dict[str, Any]:
    result = validate_experiment(spec, official=spec.is_candidate)
    if not result.valid:
        raise ValueError(
            f"invalid experiment {spec.experiment_id}: " + "; ".join(result.errors)
        )
    source = spec.source_path().read_bytes()
    metadata = json.loads(
        (spec.directory() / "experiment.json").read_text(encoding="utf-8")
    )
    return {
        "id": spec.experiment_id,
        "entrypoint": spec.entrypoint,
        "sha256": _sha256(source),
        "source_b64": base64.b64encode(source).decode("ascii"),
        "eligibility": spec.eligibility,
        "parent_id": spec.parent,
        "family": spec.family,
        "attractor_solver_metrics": metadata.get("solver"),
    }


def _remote_script(payload: dict[str, Any]) -> str:
    encoded = base64.b64encode(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    return f'''#!/usr/bin/env python3
"""Generated One Layer Deeper Colab job. Do not edit."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

PAYLOAD = json.loads(base64.b64decode({encoded!r}))
RUNTIME_PROBE = {RUNTIME_PROBE!r}


def emit(kind, **values):
    print(json.dumps({{"type": kind, **values}}, sort_keys=True), flush=True)


def redact(value, limit=4000):
    words = str(value).split()
    cleaned = []
    hide_next = False
    for word in words:
        lowered = word.lower()
        if hide_next:
            cleaned.append("[REDACTED]")
            hide_next = False
            continue
        if lowered.rstrip(":") == "bearer":
            cleaned.append(word)
            hide_next = True
            continue
        if word.startswith("AIza"):
            cleaned.append("[REDACTED]")
            continue
        if any(marker in lowered for marker in ("api_key=", "apikey=", "token=", "secret=", "password=")):
            cleaned.append(word.split("=", 1)[0] + "=[REDACTED]")
            continue
        cleaned.append(word)
    return " ".join(cleaned)[-limit:]


def run(command, *, cwd=None, timeout=None, capture=False):
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )


def hardware():
    try:
        import torch
    except ImportError:
        return {{
            "accelerator": "cpu",
            "gpu_name": None,
            "gpu_memory_bytes": None,
            "python_version": platform.python_version(),
            "torch_version": None,
            "cuda_version": None,
            "driver_version": None,
            "bf16_supported": False,
        }}
    if not torch.cuda.is_available():
        return {{
            "accelerator": "cpu",
            "gpu_name": None,
            "gpu_memory_bytes": None,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "driver_version": None,
            "bf16_supported": False,
        }}
    props = torch.cuda.get_device_properties(0)
    info = {{
        "accelerator": "cuda",
        "gpu_name": props.name,
        "gpu_memory_bytes": props.total_memory,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "driver_version": None,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        info["driver_version"] = completed.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return info


def verify_accelerator(info):
    expected = PAYLOAD["requested_accelerator"].upper()
    if expected == "CPU":
        if info.get("accelerator") != "cpu":
            raise RuntimeError(f"requested CPU but received {{info}}")
        actual = "CPU"
    else:
        actual = str(info.get("gpu_name", "")).upper()
        if expected not in actual:
            raise RuntimeError(f"requested {{expected}} but received {{actual or 'no GPU'}}")
    if PAYLOAD.get("requires_bf16") and not info.get("bf16_supported"):
        raise RuntimeError(
            f"requested runtime requires BF16 but {{actual or 'the allocated GPU'}} "
            "does not report BF16 support"
        )


def setup_repo():
    root = Path("/content/one-layer-deeper")
    if root.exists():
        raise RuntimeError(f"refusing to overwrite existing {{root}}")
    run(["git", "clone", PAYLOAD["repository_url"], str(root)])
    run(["git", "checkout", "--detach", PAYLOAD["repository_sha"]], cwd=root)
    head = run(["git", "rev-parse", "HEAD"], cwd=root, capture=True).stdout.strip()
    if head != PAYLOAD["repository_sha"]:
        raise RuntimeError(f"repository SHA mismatch: {{head}}")
    run([sys.executable, "-m", "pip", "install", "uv==0.12.1"])
    run(["uv", "python", "install", "3.13.5"], cwd=root)
    run(["uv", "sync", "--frozen", "--python", "3.13.5"], cwd=root)
    return root


def runtime_hardware(root):
    completed = run(
        ["uv", "run", "--python", "3.13.5", "python", "-c", RUNTIME_PROBE],
        cwd=root,
        capture=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("locked runtime hardware probe produced no output")
    info = json.loads(lines[-1])
    verify_accelerator(info)
    if info.get("python_version") != "3.13.5":
        raise RuntimeError(f"locked runtime used Python {{info.get('python_version')}}")
    return info


def dataset_config_sha256(root):
    data_root = PAYLOAD.get("data_root")
    if not data_root:
        return None
    path = root / data_root / "dataset_config.json"
    if not path.is_file():
        raise RuntimeError(f"generated dataset is missing {{path}}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_manifest_and_dataset(root):
    manifest = base64.b64decode(PAYLOAD["manifest_b64"])
    if not manifest:
        return None, None
    if hashlib.sha256(manifest).hexdigest() != PAYLOAD["manifest_sha256"]:
        raise RuntimeError("manifest hash mismatch")
    manifest_path = root / "remote_manifests" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest)
    dataset_command = PAYLOAD.get("dataset_command")
    if dataset_command:
        run(["bash", "-lc", dataset_command], cwd=root)
    return manifest_path, dataset_config_sha256(root)


def write_embedded(root, item, subdir):
    source = base64.b64decode(item["source_b64"])
    digest = hashlib.sha256(source).hexdigest()
    if digest != item["sha256"]:
        raise RuntimeError(f"embedded source hash mismatch for {{item['id']}}")
    target = root / subdir / item["id"] / item["entrypoint"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source)
    return target


def run_totals(result):
    seeds = (result or {{}}).get("seeds", [])
    return (
        [int(seed["seed"]) for seed in seeds if "seed" in seed],
        sum(int(seed.get("completed_training_steps") or 0) for seed in seeds),
        sum(float(seed.get("training_seconds") or 0.0) for seed in seeds),
        sum(float(seed.get("evaluation_seconds") or 0.0) for seed in seeds),
    )


def new_run_id(item):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    material = f"{{time.time_ns()}}:{{item['id']}}".encode("utf-8")
    return f"{{stamp}}-{{hashlib.sha256(material).hexdigest()[:10]}}"


def make_record(
    item,
    runtime_info,
    dataset_hash,
    *,
    run_id,
    started_at_utc,
    elapsed_seconds,
    command,
    result,
    failure,
):
    seeds, completed_steps, training_seconds, evaluation_seconds = run_totals(result)
    return {{
        "schema_version": PAYLOAD["schema_version"],
        "run_id": run_id,
        "candidate_id": item["id"],
        "parent_id": item.get("parent_id"),
        "eligibility": item["eligibility"],
        "family": item.get("family"),
        "score_class": PAYLOAD["score_class"],
        "comparable": bool(PAYLOAD.get("comparable")),
        "repository_url": PAYLOAD["repository_url"],
        "repository_sha": PAYLOAD["repository_sha"],
        "git_commit": PAYLOAD["repository_sha"],
        "git_dirty": False,
        "source_sha256": item["sha256"],
        "submission_sha256": item["sha256"],
        "manifest_sha256": PAYLOAD["manifest_sha256"],
        "dataset_config_sha256": dataset_hash,
        "requested_accelerator": PAYLOAD["requested_accelerator"],
        "accelerator": runtime_info.get("accelerator"),
        "gpu_name": runtime_info.get("gpu_name"),
        "gpu_memory_bytes": runtime_info.get("gpu_memory_bytes"),
        "python_version": runtime_info.get("python_version"),
        "torch_version": runtime_info.get("torch_version"),
        "cuda_version": runtime_info.get("cuda_version"),
        "driver_version": runtime_info.get("driver_version"),
        "bf16_supported": runtime_info.get("bf16_supported"),
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "seeds": seeds,
        "completed_steps": completed_steps,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "attractor_solver_metrics": item.get("attractor_solver_metrics"),
        "command": command,
        "result_json": result,
        "failure": failure,
        "stdout_path": "",
        "stderr_path": "",
    }}


def failure_kind(stderr):
    lowered = str(stderr).lower()
    return "oom" if "out of memory" in lowered else "model-failure"


def run_sweep(root, runtime_info):
    manifest_path, dataset_hash = prepare_manifest_and_dataset(root)
    if manifest_path is None:
        raise RuntimeError("sweep job is missing its manifest")
    passed = failed = timed_out = 0
    for item in PAYLOAD["experiments"]:
        source_path = write_embedded(root, item, "remote_candidates")
        command = [
            "uv", "run", "--python", "3.13.5", "python", "-m",
            "benchmark.runner", "--manifest", str(manifest_path),
            "--submission-file", str(source_path),
        ]
        run_id = new_run_id(item)
        started_at_utc = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=PAYLOAD["candidate_timeout_seconds"],
            )
            marker = next(
                (line.removeprefix("RESULT_JSON=") for line in reversed(completed.stdout.splitlines()) if line.startswith("RESULT_JSON=")),
                None,
            )
            if completed.returncode == 0 and marker is not None:
                try:
                    result = json.loads(marker)
                except json.JSONDecodeError as exc:
                    failed += 1
                    record = make_record(
                        item, runtime_info, dataset_hash,
                        run_id=run_id, started_at_utc=started_at_utc,
                        elapsed_seconds=time.monotonic()-started, command=command,
                        result=None,
                        failure={{"kind": "invalid-result", "message": redact(exc)}},
                    )
                    emit("candidate_failure", record=record)
                else:
                    passed += 1
                    record = make_record(
                        item, runtime_info, dataset_hash,
                        run_id=run_id, started_at_utc=started_at_utc,
                        elapsed_seconds=time.monotonic()-started, command=command,
                        result=result, failure=None,
                    )
                    emit("candidate_result", record=record)
            else:
                failed += 1
                stderr = completed.stderr or ""
                kind = "missing-result" if completed.returncode == 0 else failure_kind(stderr)
                record = make_record(
                    item, runtime_info, dataset_hash,
                    run_id=run_id, started_at_utc=started_at_utc,
                    elapsed_seconds=time.monotonic()-started, command=command,
                    result=None,
                    failure={{
                        "kind": kind,
                        "returncode": completed.returncode,
                        "stderr_tail": redact(stderr),
                    }},
                )
                emit("candidate_failure", record=record)
        except subprocess.TimeoutExpired as exc:
            timed_out += 1
            record = make_record(
                item, runtime_info, dataset_hash,
                run_id=run_id, started_at_utc=started_at_utc,
                elapsed_seconds=time.monotonic()-started, command=command,
                result=None,
                failure={{
                    "kind": "timeout",
                    "timeout_seconds": PAYLOAD["candidate_timeout_seconds"],
                    "stderr_tail": redact(exc.stderr or ""),
                }},
            )
            emit("candidate_timeout", record=record)
    emit("sweep_summary", passed=passed, failed=failed, timed_out=timed_out, total=len(PAYLOAD["experiments"]))
    return 0 if failed == 0 and timed_out == 0 else 1


def run_research(root, runtime_info):
    manifest_path, dataset_hash = prepare_manifest_and_dataset(root)
    passed = failed = timed_out = 0
    for item in PAYLOAD["experiments"]:
        source_path = write_embedded(root, item, "remote_research")
        if manifest_path is None:
            device = "cuda" if runtime_info.get("accelerator") == "cuda" else "cpu"
            command = [
                "uv", "run", "--python", "3.13.5", "python", str(source_path),
                "--device", device,
            ]
            execution = "standalone-research"
        else:
            command = [
                "uv", "run", "--python", "3.13.5", "python", "-m",
                "benchmark.runner", "--manifest", str(manifest_path),
                "--submission-file", str(source_path),
            ]
            execution = "benchmark-diagnostic"
        run_id = new_run_id(item)
        started_at_utc = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command, cwd=root, text=True, capture_output=True,
                timeout=PAYLOAD["candidate_timeout_seconds"],
            )
            if completed.returncode == 0:
                if manifest_path is None:
                    encoded_result = completed.stdout
                else:
                    encoded_result = next(
                        (
                            line.removeprefix("RESULT_JSON=")
                            for line in reversed(completed.stdout.splitlines())
                            if line.startswith("RESULT_JSON=")
                        ),
                        "",
                    )
                try:
                    result = json.loads(encoded_result)
                except json.JSONDecodeError as exc:
                    failed += 1
                    record = make_record(
                        item, runtime_info, dataset_hash,
                        run_id=run_id, started_at_utc=started_at_utc,
                        elapsed_seconds=time.monotonic()-started, command=command,
                        result=None,
                        failure={{"kind": "invalid-result", "message": redact(exc)}},
                    )
                    record["research_execution"] = execution
                    emit("research_failure", record=record)
                else:
                    passed += 1
                    record = make_record(
                        item, runtime_info, dataset_hash,
                        run_id=run_id, started_at_utc=started_at_utc,
                        elapsed_seconds=time.monotonic()-started, command=command,
                        result=result, failure=None,
                    )
                    record["research_execution"] = execution
                    emit("research_result", record=record)
            else:
                failed += 1
                stderr = completed.stderr or ""
                record = make_record(
                    item, runtime_info, dataset_hash,
                    run_id=run_id, started_at_utc=started_at_utc,
                    elapsed_seconds=time.monotonic()-started, command=command,
                    result=None,
                    failure={{
                        "kind": failure_kind(stderr),
                        "returncode": completed.returncode,
                        "stderr_tail": redact(stderr),
                    }},
                )
                record["research_execution"] = execution
                emit("research_failure", record=record)
        except subprocess.TimeoutExpired as exc:
            timed_out += 1
            record = make_record(
                item, runtime_info, dataset_hash,
                run_id=run_id, started_at_utc=started_at_utc,
                elapsed_seconds=time.monotonic()-started, command=command,
                result=None,
                failure={{
                    "kind": "timeout",
                    "timeout_seconds": PAYLOAD["candidate_timeout_seconds"],
                    "stderr_tail": redact(exc.stderr or ""),
                }},
            )
            record["research_execution"] = execution
            emit("research_timeout", record=record)
    emit("research_summary", passed=passed, failed=failed, timed_out=timed_out, total=len(PAYLOAD["experiments"]))
    return 0 if failed == 0 and timed_out == 0 else 1


def main():
    info = hardware()
    emit("preflight", requested_accelerator=PAYLOAD["requested_accelerator"], hardware=info)
    verify_accelerator(info)
    if PAYLOAD["mode"] == "preflight":
        return 0
    root = setup_repo()
    runtime_info = runtime_hardware(root)
    emit("setup_complete", repository_sha=PAYLOAD["repository_sha"], hardware=runtime_info)
    if PAYLOAD["mode"] == "environment-preflight":
        emit("environment_preflight_complete", repository_sha=PAYLOAD["repository_sha"])
        return 0
    if PAYLOAD["mode"] == "sweep":
        return run_sweep(root, runtime_info)
    return run_research(root, runtime_info)


def entrypoint():
    try:
        return main()
    except Exception as exc:
        emit("setup_failure", error_type=type(exc).__name__, message=redact(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(entrypoint())
'''


def build_colab_job(
    *,
    mode: JobMode,
    output_path: Path,
    repo_root: Path,
    experiments: tuple[ExperimentSpec, ...] = (),
    manifest_path: Path | None = None,
    requested_accelerator: str = "A100",
    repository_url: str = DEFAULT_REPOSITORY_URL,
    repository_sha: str | None = None,
    candidate_timeout_seconds: int = 900,
) -> Path:
    if mode in {"preflight", "environment-preflight"} and (
        experiments or manifest_path is not None
    ):
        raise ValueError(f"{mode} jobs do not accept experiments or a manifest")
    if mode in {"sweep", "research"} and not experiments:
        raise ValueError(f"{mode} jobs require at least one experiment")
    if mode == "sweep" and manifest_path is None:
        raise ValueError("sweep jobs require a manifest")
    if mode == "sweep" and any(not spec.is_candidate for spec in experiments):
        raise ValueError("sweep jobs accept candidate-safe experiments only")
    if mode == "research" and any(spec.is_candidate for spec in experiments):
        raise ValueError("research jobs accept research-only experiments only")
    if (
        mode == "research"
        and manifest_path is not None
        and any(spec.family != "diagnostic" for spec in experiments)
    ):
        raise ValueError(
            "manifest-backed research jobs accept exact-controller diagnostics only"
        )
    pinned_repository_sha = repository_sha or _git_commit(repo_root)
    manifest_bytes = b""
    manifest_payload: dict[str, Any] = {}
    dataset_command = None
    data_root = None
    if manifest_path is not None:
        manifest_bytes = manifest_path.read_bytes()
        manifest_payload = json.loads(manifest_bytes)
        data_root = manifest_payload.get("data", {}).get("data_root")
        dataset_command = _dataset_command(repo_root, data_root)
    requested_upper = requested_accelerator.upper()
    requires_bf16 = any(
        model in requested_upper for model in ("A100", "H100")
    ) or (
        manifest_payload.get("runtime", {}).get("dtype") == "bfloat16"
    )
    if requested_upper == "CPU":
        score_class = "contract-smoke"
    elif (
        "H100" in requested_upper
        and mode == "sweep"
        and manifest_path is not None
        and is_exact_public_h100_manifest(
            manifest_path,
            repo_root=repo_root,
            repository_sha=pinned_repository_sha,
        )
    ):
        score_class = "tier-faithful"
    else:
        score_class = "research-screen"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "requested_accelerator": requested_accelerator,
        "requires_bf16": requires_bf16,
        "score_class": score_class,
        "comparable": mode == "sweep",
        "repository_url": repository_url,
        "repository_sha": pinned_repository_sha,
        "candidate_timeout_seconds": candidate_timeout_seconds,
        "experiments": [_encode_source(spec) for spec in experiments],
        "manifest_b64": base64.b64encode(manifest_bytes).decode("ascii"),
        "manifest_sha256": _sha256(manifest_bytes),
        "dataset_command": dataset_command,
        "data_root": data_root,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_remote_script(payload), encoding="utf-8")
    return output_path
