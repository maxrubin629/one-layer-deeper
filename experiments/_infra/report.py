"""Aggregate immutable records without mixing incompatible hardware groups."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
from typing import Any, Iterable


PROMOTABLE_SCORE_CLASSES = frozenset({"research-screen", "tier-faithful"})
COLAB_RECORD_ENVELOPE_TYPES = frozenset(
    {
        "candidate_result",
        "candidate_failure",
        "candidate_timeout",
        "research_result",
        "research_failure",
        "research_timeout",
    }
)
REQUIRED_RUN_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "candidate_id",
        "eligibility",
        "score_class",
        "repository_sha",
        "source_sha256",
        "manifest_sha256",
        "requested_accelerator",
        "accelerator",
        "python_version",
        "started_at_utc",
        "finished_at_utc",
        "elapsed_seconds",
        "result_json",
        "failure",
    }
)


def _colab_envelope_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        stripped = line.strip()
        object_start = stripped.find("{")
        if object_start < 0:
            continue
        stripped = stripped[object_start:]
        try:
            envelope, _ = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(envelope, dict) or envelope.get("type") not in COLAB_RECORD_ENVELOPE_TYPES:
            continue
        record = envelope.get("record")
        if not isinstance(record, dict):
            raise ValueError(
                f"Colab envelope on line {line_number} does not contain a record object"
            )
        missing = REQUIRED_RUN_RECORD_FIELDS - record.keys()
        if missing:
            raise ValueError(
                f"Colab record on line {line_number} is missing fields: {sorted(missing)}"
            )
        records.append(dict(record))
    return records


def is_colab_result_log(path: Path) -> bool:
    return path.is_file() and bool(_colab_envelope_records(path))


def ingest_colab_log(path: Path, *, artifact_root: Path) -> list[dict[str, Any]]:
    """Persist complete Colab envelopes and their raw log immutably."""

    path = path.resolve()
    records = _colab_envelope_records(path)
    if not records:
        raise ValueError(f"no candidate or research records found in {path}")

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    log_path = artifact_root / "logs" / "colab" / f"{digest}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        if log_path.read_bytes() != raw:
            raise FileExistsError(f"immutable Colab log collision at {log_path}")
    else:
        shutil.copyfile(path, log_path)

    persisted: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        run_id = str(record["run_id"])
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ValueError(f"unsafe Colab run_id: {run_id!r}")
        record["stdout_path"] = str(log_path.resolve())
        record["stderr_path"] = str(log_path.resolve())
        destination = artifact_root / "runs" / run_id / "record.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
        if destination.exists():
            if destination.read_text(encoding="utf-8") != encoded:
                raise FileExistsError(
                    f"immutable Colab record collision at {destination}"
                )
        else:
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        persisted.append(record)
    return persisted


def load_records(root: Path) -> list[dict[str, Any]]:
    if root.is_file():
        records = _colab_envelope_records(root)
        if records:
            return records
        payload = json.loads(root.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "candidate_id" not in payload:
            raise ValueError(f"{root} is neither a RunRecord nor a Colab result log")
        return [payload]
    paths = sorted(root.rglob("record.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def hardware_group(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("accelerator"),
        record.get("gpu_name"),
        record.get("gpu_memory_bytes"),
        record.get("python_version"),
        record.get("torch_version"),
        record.get("cuda_version"),
        record.get("driver_version"),
        record.get("bf16_supported"),
        record.get("score_class"),
        record.get("git_commit"),
        record.get("manifest_sha256"),
        record.get("dataset_config_sha256"),
    )


def _first_uncertified_accuracy(seed: dict[str, Any], key: str) -> float:
    profile = seed.get("depth_profile", {})
    rungs_key = "rungs" if key == "seen" else "ood_n_rungs"
    for rung in profile.get(rungs_key, []):
        if rung.get("status") != "certified":
            return float(rung.get("exact_accuracy", 0.0))
    return 1.0 if profile.get(rungs_key) else 0.0


def promotion_key(record: dict[str, Any]) -> tuple[float, ...]:
    result = record.get("result_json") or {}
    profile = result.get("depth_profile", {})
    seeds = result.get("seeds", [])
    seen = min((_first_uncertified_accuracy(seed, "seen") for seed in seeds), default=0.0)
    ood = min((_first_uncertified_accuracy(seed, "ood") for seed in seeds), default=0.0)
    return (
        float(profile.get("max_certified_time_steps") or 0),
        float(profile.get("ood_n_max_certified_time_steps") or 0),
        seen,
        ood,
        float(result.get("score", {}).get("mean_exact_accuracy") or 0.0),
    )


def seed_promotion_key(seed: dict[str, Any]) -> tuple[float, ...]:
    """Build the lexicographic promotion tuple for one matched seed."""

    profile = seed.get("depth_profile", {})
    measurements = [
        float(metrics.get("exact_accuracy") or 0.0)
        for metrics in seed.get("evaluation", {}).values()
    ]
    return (
        float(profile.get("max_certified_time_steps") or 0),
        float(profile.get("ood_n_max_certified_time_steps") or 0),
        _first_uncertified_accuracy(seed, "seen"),
        _first_uncertified_accuracy(seed, "ood"),
        statistics.fmean(measurements) if measurements else 0.0,
    )


def median_promotion_key(record: dict[str, Any]) -> tuple[float, ...]:
    """Return the component-wise median tuple across a record's seeds."""

    keys = [
        seed_promotion_key(seed)
        for seed in (record.get("result_json") or {}).get("seeds", [])
    ]
    if not keys:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return tuple(float(statistics.median(column)) for column in zip(*keys))


def _seed_map(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(seed["seed"]): seed
        for seed in (record.get("result_json") or {}).get("seeds", [])
        if "seed" in seed
    }


def _latest_by_candidate(
    records: Iterable[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for record in records:
        if record.get("eligibility") != "candidate-safe":
            continue
        candidate_id = record.get("candidate_id")
        if not candidate_id:
            continue
        bucket = grouped.setdefault(hardware_group(record), {})
        previous = bucket.get(str(candidate_id))
        if previous is None or str(record.get("finished_at_utc") or "") >= str(
            previous.get("finished_at_utc") or ""
        ):
            bucket[str(candidate_id)] = record
    return grouped


def paired_comparisons(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare each latest candidate with its parent inside one evidence group.

    The grouping key includes actual GPU identity and memory, the complete software
    stack, score class, repository commit, manifest, and dataset configuration. A
    comparison therefore cannot silently cross A100 memory variants or mix A100
    research evidence with H100 tier-faithful evidence.
    """

    comparisons: list[dict[str, Any]] = []
    for _group, candidates in _latest_by_candidate(records).items():
        baseline = candidates.get("v0_baseline_adamw")
        baseline_present = baseline is not None
        baseline_valid = bool(
            baseline is not None
            and baseline.get("failure") is None
            and baseline.get("result_json") is not None
            and baseline.get("comparable")
            and baseline.get("score_class") in PROMOTABLE_SCORE_CLASSES
        )
        ranked = sorted(
            (
                record
                for candidate_id, record in candidates.items()
                if baseline_valid
                and candidate_id != "v0_baseline_adamw"
                and record.get("failure") is None
                and record.get("result_json") is not None
                and record.get("comparable")
                and record.get("score_class") in PROMOTABLE_SCORE_CLASSES
            ),
            key=promotion_key,
            reverse=True,
        )
        ranks = {
            str(record.get("run_id")): rank
            for rank, record in enumerate(ranked, start=1)
        }
        for candidate_id, candidate in candidates.items():
            run_id = str(candidate.get("run_id"))
            parent_id = candidate.get("parent_id")
            rank = ranks.get(run_id)
            base = {
                "run_id": run_id,
                "baseline_present": baseline_present,
                "baseline_valid": baseline_valid,
                "screen_rank": rank,
                "screen_top_four": bool(rank is not None and rank <= 4),
                "common_seed_count": 0,
                "seed_wins": 0,
                "median_beats_parent": False,
                "confirmation_pass": False,
                "parent_run_id": None,
                "comparison_status": "unpaired",
            }
            if candidate_id == "v0_baseline_adamw":
                if candidate.get("failure") is not None or candidate.get("result_json") is None:
                    status = "baseline-failed"
                elif candidate.get("score_class") not in PROMOTABLE_SCORE_CLASSES:
                    status = "non-promotable-score-class"
                elif not candidate.get("comparable"):
                    status = "not-marked-comparable"
                else:
                    status = "baseline"
                comparisons.append({**base, "comparison_status": status})
                continue
            if not baseline_present:
                comparisons.append(
                    {**base, "comparison_status": "missing-v0-baseline"}
                )
                continue
            if baseline.get("failure") is not None or baseline.get("result_json") is None:
                comparisons.append(
                    {**base, "comparison_status": "invalid-v0-baseline"}
                )
                continue
            promotable_score_class = candidate.get("score_class") in PROMOTABLE_SCORE_CLASSES
            if not promotable_score_class:
                comparisons.append(
                    {**base, "comparison_status": "non-promotable-score-class"}
                )
                continue
            if candidate.get("failure") is not None or candidate.get("result_json") is None:
                comparisons.append(
                    {**base, "comparison_status": "candidate-failed"}
                )
                continue
            if not candidate.get("comparable") or not baseline.get("comparable"):
                comparisons.append(
                    {**base, "comparison_status": "not-marked-comparable"}
                )
                continue
            parent = candidates.get(str(parent_id))
            if parent is None:
                comparisons.append(
                    {**base, "comparison_status": "missing-matched-parent"}
                )
                continue
            if parent.get("failure") is not None or parent.get("result_json") is None:
                comparisons.append(
                    {
                        **base,
                        "parent_run_id": parent.get("run_id"),
                        "comparison_status": "parent-failed",
                    }
                )
                continue
            candidate_seeds = _seed_map(candidate)
            parent_seeds = _seed_map(parent)
            common_seeds = sorted(candidate_seeds.keys() & parent_seeds.keys())
            candidate_keys = [
                seed_promotion_key(candidate_seeds[seed]) for seed in common_seeds
            ]
            parent_keys = [
                seed_promotion_key(parent_seeds[seed]) for seed in common_seeds
            ]
            seed_wins = sum(
                candidate_key > parent_key
                for candidate_key, parent_key in zip(candidate_keys, parent_keys)
            )
            if candidate_keys:
                candidate_median = tuple(
                    float(statistics.median(column))
                    for column in zip(*candidate_keys)
                )
                parent_median = tuple(
                    float(statistics.median(column)) for column in zip(*parent_keys)
                )
            else:
                candidate_median = parent_median = (0.0,) * 5
            median_beats_parent = candidate_median > parent_median
            comparable = bool(candidate.get("comparable")) and bool(
                parent.get("comparable")
            )
            confirmation_pass = bool(
                comparable
                and promotable_score_class
                and len(common_seeds) >= 3
                and seed_wins >= 2
                and median_beats_parent
                and candidate.get("failure") is None
                and parent.get("failure") is None
            )
            if not comparable:
                status = "not-marked-comparable"
            elif len(common_seeds) < 3:
                status = "insufficient-matched-seeds"
            elif confirmation_pass:
                status = "passes-confirmation"
            else:
                status = "fails-confirmation"
            comparisons.append(
                {
                    **base,
                    "parent_run_id": parent.get("run_id"),
                    "common_seed_count": len(common_seeds),
                    "seed_wins": seed_wins,
                    "median_beats_parent": median_beats_parent,
                    "confirmation_pass": confirmation_pass,
                    "comparison_status": status,
                }
            )
    return comparisons


def summarize(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = list(records)
    comparisons = {
        comparison["run_id"]: comparison
        for comparison in paired_comparisons(records)
    }
    rows: list[dict[str, Any]] = []
    for record in records:
        key = promotion_key(record)
        row = {
            "run_id": record.get("run_id"),
            "candidate_id": record.get("candidate_id"),
            "parent_id": record.get("parent_id"),
            "score_class": record.get("score_class"),
            "gpu_name": record.get("gpu_name"),
            "gpu_memory_bytes": record.get("gpu_memory_bytes"),
            "comparable": record.get("comparable"),
            "seen_max_t": key[0],
            "ood_n_max_t": key[1],
            "seen_next_accuracy": key[2],
            "ood_n_next_accuracy": key[3],
            "mean_exact_accuracy": key[4],
            "failure": record.get("failure"),
        }
        row.update(comparisons.get(str(record.get("run_id")), {}))
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    if not fieldnames:
        fieldnames = ["run_id", "candidate_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "candidate_id",
        "score_class",
        "gpu_name",
        "seen_max_t",
        "ood_n_max_t",
        "seen_next_accuracy",
        "ood_n_next_accuracy",
        "mean_exact_accuracy",
        "screen_rank",
        "seed_wins",
        "median_beats_parent",
        "confirmation_pass",
        "comparison_status",
        "failure",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
