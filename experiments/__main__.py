"""Command-line interface for the versioned experiment suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from ._infra.build_colab_job import build_colab_job
from ._infra.registry import EXPERIMENTS, MATRICES, get_experiment, get_matrix
from ._infra.render import render_candidate
from ._infra.report import (
    ingest_colab_log,
    is_colab_result_log,
    load_records,
    summarize,
    write_csv,
    write_markdown,
)
from ._infra.run_local import run_candidate
from ._infra.validate import validate_all, validate_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "one_layer_deeper"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list experiments and matrices")

    validate = sub.add_parser("validate", help="validate experiment sources")
    validate.add_argument("experiment_id", nargs="?")
    validate.add_argument("--all", action="store_true")
    validate.add_argument("--official", action="store_true")

    render = sub.add_parser("render", help="create an immutable rendered copy")
    render.add_argument("experiment_id")
    render.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT / "rendered")

    run = sub.add_parser("run", help="run one candidate through benchmark.runner")
    run.add_argument("experiment_id")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument(
        "--score-class",
        choices=("contract-smoke", "research-screen", "tier-faithful", "hosted-authoritative"),
        required=True,
    )
    run.add_argument("--requested-accelerator")
    run.add_argument("--timeout", type=float)
    run.add_argument("--comparable", action="store_true")
    run.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)

    report = sub.add_parser("report", help="summarize immutable records")
    report.add_argument("records", type=Path)
    report.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_ROOT / "reports" / "summary")
    report.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)

    colab = sub.add_parser("build-colab-job", help="build an offline one-shot Colab payload")
    colab.add_argument(
        "--mode",
        choices=("preflight", "environment-preflight", "sweep", "research"),
        required=True,
    )
    colab.add_argument("--matrix", choices=tuple(MATRICES))
    colab.add_argument(
        "--experiment",
        dest="experiment_ids",
        action="append",
        help="include one experiment; repeat for a promoted-finalist subset",
    )
    colab.add_argument("--manifest", type=Path)
    colab.add_argument("--output", type=Path, required=True)
    colab.add_argument("--accelerator", default="A100")
    colab.add_argument("--candidate-timeout", type=int, default=900)
    return parser


def _list() -> int:
    for spec in EXPERIMENTS:
        print(
            f"{spec.experiment_id}\t{spec.eligibility}\tparent={spec.parent or '-'}\t{spec.title}"
        )
    print("\nMatrices:")
    for name, ids in MATRICES.items():
        print(f"{name}\t{len(ids)} experiments")
    return 0


def _validate(args: argparse.Namespace) -> int:
    if args.all:
        results = validate_all(official=args.official)
    elif args.experiment_id:
        results = (validate_experiment(get_experiment(args.experiment_id), official=args.official),)
    else:
        raise SystemExit("validate requires an experiment ID or --all")
    failed = False
    for result in results:
        status = "ok" if result.valid else "invalid"
        print(f"{status}\t{result.experiment_id}\tbytes={result.source_bytes}")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        for error in result.errors:
            failed = True
            print(f"  error: {error}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        return _list()
    if args.command == "validate":
        return _validate(args)
    if args.command == "render":
        path, digest = render_candidate(
            get_experiment(args.experiment_id), output_root=args.output_root, official=True
        )
        print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
        return 0
    if args.command == "run":
        record = run_candidate(
            get_experiment(args.experiment_id),
            manifest_path=args.manifest,
            repo_root=REPO_ROOT,
            artifact_root=args.artifact_root,
            score_class=args.score_class,
            requested_accelerator=args.requested_accelerator,
            timeout_seconds=args.timeout,
            comparable=args.comparable,
        )
        print(json.dumps(asdict(record), sort_keys=True))
        return 1 if record.failure else 0
    if args.command == "report":
        if is_colab_result_log(args.records):
            ingest_colab_log(args.records, artifact_root=args.artifact_root)
            records = load_records(args.artifact_root / "runs")
        else:
            records = load_records(args.records)
        rows = summarize(records)
        write_csv(rows, args.output.with_suffix(".csv"))
        write_markdown(rows, args.output.with_suffix(".md"))
        print(json.dumps({"records": len(rows), "output": str(args.output)}, sort_keys=True))
        return 0
    if args.command == "build-colab-job":
        if args.matrix and args.experiment_ids:
            raise SystemExit("choose either --matrix or repeated --experiment")
        specs = (
            get_matrix(args.matrix)
            if args.matrix
            else tuple(get_experiment(item) for item in (args.experiment_ids or ()))
        )
        build_colab_job(
            mode=args.mode,
            output_path=args.output,
            repo_root=REPO_ROOT,
            experiments=specs,
            manifest_path=args.manifest,
            requested_accelerator=args.accelerator,
            candidate_timeout_seconds=args.candidate_timeout,
        )
        print(json.dumps({"output": str(args.output), "mode": args.mode}, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
