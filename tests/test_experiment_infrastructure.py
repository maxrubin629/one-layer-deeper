from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments.__main__ import main as experiments_main
from experiments._infra.build_colab_job import build_colab_job
from experiments._infra.registry import EXPERIMENTS, MATRICES, get_experiment
from experiments._infra.render import render_candidate
from experiments._infra.report import (
    hardware_group,
    ingest_colab_log,
    load_records,
    paired_comparisons,
    promotion_key,
)
from experiments._infra.run_local import run_candidate
from experiments._infra.validate import is_exact_public_h100_manifest


ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):
    def test_ids_are_unique_version_prefixed_and_parents_exist(self) -> None:
        ids = {spec.experiment_id for spec in EXPERIMENTS}
        self.assertEqual(len(ids), len(EXPERIMENTS))
        for spec in EXPERIMENTS:
            self.assertTrue(spec.experiment_id.startswith(spec.version + "_"))
            if spec.parent is not None:
                self.assertIn(spec.parent, ids)

    def test_candidate_matrix_excludes_research(self) -> None:
        for experiment_id in MATRICES["candidate-safe"]:
            self.assertTrue(get_experiment(experiment_id).is_candidate)

    def test_renderer_is_deterministic_and_hash_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first, first_hash = render_candidate(
                get_experiment("v0_baseline_adamw"),
                output_root=Path(temporary),
            )
            second, second_hash = render_candidate(
                get_experiment("v0_baseline_adamw"),
                output_root=Path(temporary),
            )
            self.assertEqual(first, second)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first.parent.name, first_hash)


class ReportTests(unittest.TestCase):
    def test_hardware_group_distinguishes_a100_memory(self) -> None:
        base = {
            "accelerator": "cuda",
            "gpu_name": "NVIDIA A100",
            "gpu_memory_bytes": 40,
            "python_version": "3.13.5",
            "torch_version": "2.12.1",
            "cuda_version": "13",
            "git_commit": "abc",
            "manifest_sha256": "m",
            "dataset_config_sha256": "d",
        }
        other = {**base, "gpu_memory_bytes": 80}
        self.assertNotEqual(hardware_group(base), hardware_group(other))

    def test_promotion_key_uses_certification_before_mean(self) -> None:
        record = {
            "result_json": {
                "depth_profile": {
                    "max_certified_time_steps": 4,
                    "ood_n_max_certified_time_steps": 2,
                },
                "score": {"mean_exact_accuracy": 0.9},
                "seeds": [],
            }
        }
        self.assertEqual(promotion_key(record), (4.0, 2.0, 0.0, 0.0, 0.9))

    def test_local_runner_cannot_mint_hosted_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "hosted-authoritative"):
            run_candidate(
                get_experiment("v0_baseline_adamw"),
                manifest_path=ROOT / "benchmark" / "manifests" / "smoke_cpu.json",
                repo_root=ROOT,
                artifact_root=ROOT / "artifacts" / "one_layer_deeper",
                score_class="hosted-authoritative",
            )

    def test_local_runner_cannot_label_a_derived_manifest_tier_faithful(self) -> None:
        hardware = {
            "accelerator": "cuda",
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_memory_bytes": 80_000_000_000,
            "torch_version": "2.12.1",
            "cuda_version": "13.0",
            "driver_version": "600.1",
            "bf16_supported": True,
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "experiments._infra.run_local._hardware", return_value=hardware
        ), self.assertRaisesRegex(ValueError, "unmodified public H100 manifest"):
            run_candidate(
                get_experiment("v0_baseline_adamw"),
                manifest_path=ROOT
                / "experiments"
                / "_infra"
                / "manifests"
                / "contract_cpu.json",
                repo_root=ROOT,
                artifact_root=Path(temporary),
                score_class="tier-faithful",
            )

    def test_three_seed_confirmation_is_paired_inside_one_hardware_group(self) -> None:
        def record(
            candidate_id: str,
            parent_id: str | None,
            accuracies: tuple[float, float, float],
        ) -> dict:
            return {
                "run_id": candidate_id,
                "candidate_id": candidate_id,
                "parent_id": parent_id,
                "eligibility": "candidate-safe",
                "score_class": "research-screen",
                "comparable": True,
                "accelerator": "cuda",
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "gpu_memory_bytes": 85_056_012_288,
                "python_version": "3.13.5",
                "torch_version": "2.12.1",
                "cuda_version": "13.0",
                "driver_version": "600.1",
                "git_commit": "abc",
                "manifest_sha256": "manifest",
                "dataset_config_sha256": "dataset",
                "finished_at_utc": "2026-08-07T00:00:00+00:00",
                "failure": None,
                "result_json": {
                    "score": {"mean_exact_accuracy": sum(accuracies) / 3},
                    "seeds": [
                        {
                            "seed": seed,
                            "evaluation": {
                                "test": {"exact_accuracy": accuracy}
                            },
                            "depth_profile": {
                                "rungs": [],
                                "ood_n_rungs": [],
                            },
                        }
                        for seed, accuracy in zip((74, 75, 76), accuracies)
                    ],
                },
            }

        baseline = record("v0_baseline_adamw", None, (0.2, 0.2, 0.2))
        candidate = record("v1_tied_gelu", "v0_baseline_adamw", (0.3, 0.3, 0.1))
        comparison = {
            row["run_id"]: row
            for row in paired_comparisons((baseline, candidate))
        }["v1_tied_gelu"]
        self.assertEqual(comparison["seed_wins"], 2)
        self.assertTrue(comparison["median_beats_parent"])
        self.assertTrue(comparison["confirmation_pass"])
        self.assertEqual(comparison["comparison_status"], "passes-confirmation")

    def test_screen_ranks_require_promotable_comparable_evidence_and_valid_v0(self) -> None:
        def record(
            candidate_id: str,
            parent_id: str | None,
            *,
            score_class: str = "research-screen",
            comparable: bool = True,
            failure: dict | None = None,
        ) -> dict:
            return {
                "run_id": candidate_id,
                "candidate_id": candidate_id,
                "parent_id": parent_id,
                "eligibility": "candidate-safe",
                "score_class": score_class,
                "comparable": comparable,
                "accelerator": "cuda",
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "gpu_memory_bytes": 85_056_012_288,
                "python_version": "3.13.5",
                "torch_version": "2.12.1",
                "cuda_version": "13.0",
                "driver_version": "600.1",
                "bf16_supported": True,
                "git_commit": "abc",
                "manifest_sha256": "manifest",
                "dataset_config_sha256": "dataset",
                "finished_at_utc": "2026-08-07T00:00:00+00:00",
                "failure": failure,
                "result_json": None
                if failure
                else {"score": {"mean_exact_accuracy": 0.5}, "seeds": []},
            }

        contract_rows = {
            item["run_id"]: item
            for item in paired_comparisons(
                (
                    record(
                        "v0_baseline_adamw",
                        None,
                        score_class="contract-smoke",
                        comparable=False,
                    ),
                    record(
                        "v1_tied_gelu",
                        "v0_baseline_adamw",
                        score_class="contract-smoke",
                        comparable=False,
                    ),
                )
            )
        }
        self.assertIsNone(contract_rows["v1_tied_gelu"]["screen_rank"])
        self.assertFalse(contract_rows["v1_tied_gelu"]["screen_top_four"])
        self.assertEqual(
            contract_rows["v1_tied_gelu"]["comparison_status"],
            "non-promotable-score-class",
        )

        failed_baseline_rows = {
            item["run_id"]: item
            for item in paired_comparisons(
                (
                    record(
                        "v0_baseline_adamw",
                        None,
                        failure={"kind": "model-failure"},
                    ),
                    record("v1_tied_gelu", "v0_baseline_adamw"),
                )
            )
        }
        self.assertFalse(failed_baseline_rows["v1_tied_gelu"]["baseline_valid"])
        self.assertIsNone(failed_baseline_rows["v1_tied_gelu"]["screen_rank"])
        self.assertEqual(
            failed_baseline_rows["v1_tied_gelu"]["comparison_status"],
            "invalid-v0-baseline",
        )


class ColabPayloadTests(unittest.TestCase):
    def _build_sweep(self, output: Path, experiment_ids: tuple[str, ...]) -> dict:
        build_colab_job(
            mode="sweep",
            output_path=output,
            repo_root=ROOT,
            experiments=tuple(get_experiment(item) for item in experiment_ids),
            manifest_path=ROOT / "experiments" / "_infra" / "manifests" / "contract_cpu.json",
            requested_accelerator="A100",
            repository_sha="e32c2f985f8ed4107c96d00271448777954ecc0c",
        )
        return runpy.run_path(str(output), run_name="generated_colab_job_test")

    def test_preflight_payload_is_self_contained_and_checks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preflight.py"
            build_colab_job(
                mode="preflight",
                output_path=output,
                repo_root=ROOT,
                requested_accelerator="A100",
                repository_sha="e32c2f985f8ed4107c96d00271448777954ecc0c",
            )
            source = output.read_text(encoding="utf-8")
            ast.parse(source)
            self.assertIn("requested_accelerator", source)
            self.assertIn("verify_accelerator", source)
            self.assertIn("runtime_hardware", source)
            self.assertIn("driver_version", source)
            self.assertIn('"--device", device', source)
            self.assertNotIn("--keep", source)
            self.assertNotIn("os.environ", source)

    def test_a100_preflight_rejects_a_runtime_without_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            namespace = self._build_sweep(
                Path(temporary) / "sweep.py", ("v0_baseline_adamw",)
            )
            info = {
                "accelerator": "cuda",
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "bf16_supported": False,
            }
            with self.assertRaisesRegex(RuntimeError, "requires BF16"):
                namespace["verify_accelerator"](info)

    def test_cpu_sweep_cannot_bypass_a_bf16_manifest_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cpu-bf16-sweep.py"
            build_colab_job(
                mode="sweep",
                output_path=output,
                repo_root=ROOT,
                experiments=(get_experiment("v0_baseline_adamw"),),
                manifest_path=ROOT / "benchmark" / "manifests" / "h100_easy_e1.json",
                requested_accelerator="CPU",
                repository_sha="e32c2f985f8ed4107c96d00271448777954ecc0c",
            )
            namespace = runpy.run_path(
                str(output), run_name="generated_cpu_bf16_sweep_test"
            )
            self.assertTrue(namespace["PAYLOAD"]["requires_bf16"])
            with self.assertRaisesRegex(RuntimeError, "requires BF16"):
                namespace["verify_accelerator"](
                    {
                        "accelerator": "cpu",
                        "gpu_name": None,
                        "bf16_supported": False,
                    }
                )

    def test_cpu_environment_preflight_installs_but_never_trains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cpu-environment-preflight.py"
            build_colab_job(
                mode="environment-preflight",
                output_path=output,
                repo_root=ROOT,
                requested_accelerator="CPU",
                repository_sha="e32c2f985f8ed4107c96d00271448777954ecc0c",
            )
            source = output.read_text(encoding="utf-8")
            ast.parse(source)
            namespace = runpy.run_path(
                str(output), run_name="generated_cpu_environment_test"
            )
            self.assertEqual(namespace["PAYLOAD"]["mode"], "environment-preflight")
            self.assertEqual(namespace["PAYLOAD"]["experiments"], [])
            self.assertIn("setup_repo()", source)
            self.assertIn("environment_preflight_complete", source)

    def test_sweep_payload_selects_one_dataset_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "sweep.py"
            build_colab_job(
                mode="sweep",
                output_path=output,
                repo_root=ROOT,
                experiments=(get_experiment("v0_baseline_adamw"),),
                manifest_path=ROOT / "benchmark" / "manifests" / "h100_easy_e1.json",
                requested_accelerator="A100",
                repository_sha="e32c2f985f8ed4107c96d00271448777954ecc0c",
            )
            namespace = runpy.run_path(
                str(output), run_name="generated_colab_dataset_test"
            )
            payload = namespace["PAYLOAD"]
            self.assertEqual(
                payload["data_root"],
                "data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123",
            )
            self.assertIn(payload["data_root"], payload["dataset_command"])
            self.assertNotIn("fixed_n_899", payload["dataset_command"])
            self.assertEqual(payload["score_class"], "research-screen")
            self.assertTrue(payload["comparable"])

    def test_h100_label_requires_an_official_h100_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            derived = root / "derived.py"
            official = root / "official.py"
            common = {
                "mode": "sweep",
                "repo_root": ROOT,
                "experiments": (get_experiment("v0_baseline_adamw"),),
                "requested_accelerator": "H100",
                "repository_sha": "e32c2f985f8ed4107c96d00271448777954ecc0c",
            }
            build_colab_job(
                **common,
                output_path=derived,
                manifest_path=ROOT
                / "experiments"
                / "_infra"
                / "manifests"
                / "contract_cpu.json",
            )
            build_colab_job(
                **common,
                output_path=official,
                manifest_path=ROOT / "benchmark" / "manifests" / "h100_easy_e1.json",
            )
            derived_payload = runpy.run_path(
                str(derived), run_name="derived_h100_label_test"
            )["PAYLOAD"]
            official_payload = runpy.run_path(
                str(official), run_name="official_h100_label_test"
            )["PAYLOAD"]
            self.assertEqual(derived_payload["score_class"], "research-screen")
            self.assertEqual(official_payload["score_class"], "tier-faithful")

    def test_h100_manifest_must_match_the_pinned_git_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            manifest = repo / "benchmark" / "manifests" / "h100_test.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"name":"exact"}\n', encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=repo,
                check=True,
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertTrue(
                is_exact_public_h100_manifest(
                    manifest, repo_root=repo, repository_sha=revision
                )
            )
            manifest.write_text('{"name":"derived"}\n', encoding="utf-8")
            self.assertFalse(
                is_exact_public_h100_manifest(
                    manifest, repo_root=repo, repository_sha=revision
                )
            )

    def test_remote_hash_setup_failure_and_secret_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = self._build_sweep(
                root / "sweep.py", ("v0_baseline_adamw",)
            )
            item = dict(namespace["PAYLOAD"]["experiments"][0])
            item["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                namespace["write_embedded"](root, item, "candidates")

            secret = "top-secret-value"
            redacted = namespace["redact"](
                f"Authorization: Bearer {secret} api_key={secret} AIza{secret}"
            )
            self.assertNotIn(secret, redacted)
            self.assertIn("[REDACTED]", redacted)

            globals_dict = namespace["entrypoint"].__globals__
            original_main = globals_dict["main"]
            globals_dict["main"] = lambda: (_ for _ in ()).throw(
                RuntimeError(f"token={secret}")
            )
            stream = io.StringIO()
            try:
                with contextlib.redirect_stdout(stream), self.assertRaises(RuntimeError):
                    namespace["entrypoint"]()
            finally:
                globals_dict["main"] = original_main
            envelope = json.loads(stream.getvalue())
            self.assertEqual(envelope["type"], "setup_failure")
            self.assertNotIn(secret, envelope["message"])

    def test_remote_sweep_parses_results_and_continues_after_oom_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = self._build_sweep(
                root / "sweep.py",
                (
                    "v0_baseline_adamw",
                    "v1_tied_gelu",
                    "v1.2_persistent_context",
                ),
            )
            timeout = subprocess.TimeoutExpired(
                cmd=["python"], timeout=900, stderr="token=hidden"
            )
            outcomes = [
                SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="CUDA out of memory token=hidden",
                ),
                timeout,
                SimpleNamespace(
                    returncode=0,
                    stdout='RESULT_JSON={"score":{"mean_exact_accuracy":1.0},"seeds":[]}\n',
                    stderr="",
                ),
            ]
            runtime_info = {
                "accelerator": "cuda",
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "gpu_memory_bytes": 85_056_012_288,
                "python_version": "3.13.5",
                "torch_version": "2.12.1",
                "cuda_version": "13.0",
                "driver_version": "600.1",
                "bf16_supported": True,
            }
            stream = io.StringIO()
            with mock.patch.object(
                namespace["subprocess"], "run", side_effect=outcomes
            ), contextlib.redirect_stdout(stream):
                status = namespace["run_sweep"](root, runtime_info)
            self.assertEqual(status, 1)
            envelopes = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(
                [item["type"] for item in envelopes],
                [
                    "candidate_failure",
                    "candidate_timeout",
                    "candidate_result",
                    "sweep_summary",
                ],
            )
            self.assertEqual(envelopes[0]["record"]["failure"]["kind"], "oom")
            self.assertNotIn("hidden", json.dumps(envelopes[:2]))
            self.assertEqual(envelopes[2]["record"]["gpu_name"], runtime_info["gpu_name"])
            self.assertTrue(envelopes[2]["record"]["bf16_supported"])
            self.assertEqual(
                envelopes[2]["record"]["result_json"]["score"]["mean_exact_accuracy"], 1.0
            )
            self.assertEqual(
                envelopes[-1],
                {
                    "failed": 1,
                    "passed": 1,
                    "timed_out": 1,
                    "total": 3,
                    "type": "sweep_summary",
                },
            )

    def test_colab_jsonl_is_ingested_as_immutable_run_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = self._build_sweep(
                root / "sweep.py", ("v0_baseline_adamw",)
            )
            completed = SimpleNamespace(
                returncode=0,
                stdout=(
                    'RESULT_JSON={"score":{"mean_exact_accuracy":1.0},'
                    '"seeds":[{"seed":74,"completed_training_steps":2,'
                    '"training_seconds":1.0,"evaluation_seconds":0.1}]}\n'
                ),
                stderr="",
            )
            runtime_info = {
                "accelerator": "cuda",
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "gpu_memory_bytes": 85_056_012_288,
                "python_version": "3.13.5",
                "torch_version": "2.12.1",
                "cuda_version": "13.0",
                "driver_version": "600.1",
                "bf16_supported": True,
            }
            stream = io.StringIO()
            with mock.patch.object(
                namespace["subprocess"], "run", return_value=completed
            ), contextlib.redirect_stdout(stream):
                self.assertEqual(namespace["run_sweep"](root, runtime_info), 0)
            log = root / "colab-output.log"
            log.write_text("colab informational line\n" + stream.getvalue())
            artifact_root = root / "artifacts" / "one_layer_deeper"
            records = ingest_colab_log(log, artifact_root=artifact_root)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["completed_steps"], 2)
            self.assertEqual(record["seeds"], [74])
            self.assertEqual(record["source_sha256"], record["submission_sha256"])
            self.assertTrue(Path(record["stdout_path"]).is_file())
            destination = artifact_root / "runs" / record["run_id"] / "record.json"
            self.assertTrue(destination.is_file())
            self.assertEqual(ingest_colab_log(log, artifact_root=artifact_root), records)
            loaded = load_records(artifact_root / "runs")
            self.assertEqual(loaded, records)
            report_output = artifact_root / "reports" / "colab-check"
            with contextlib.redirect_stdout(io.StringIO()):
                status = experiments_main(
                    [
                        "report",
                        str(log),
                        "--artifact-root",
                        str(artifact_root),
                        "--output",
                        str(report_output),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue(report_output.with_suffix(".csv").is_file())
            self.assertTrue(report_output.with_suffix(".md").is_file())

    def test_manifest_backed_exact_t_research_uses_benchmark_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "exact-t.py"
            build_colab_job(
                mode="research",
                output_path=output,
                repo_root=ROOT,
                experiments=(get_experiment("v1.3_exact_t_diagnostic"),),
                manifest_path=ROOT
                / "experiments"
                / "_infra"
                / "manifests"
                / "contract_cpu.json",
                requested_accelerator="A100",
                repository_sha="e32c2f985f8ed4107c96d00271448777954ecc0c",
            )
            namespace = runpy.run_path(
                str(output), run_name="generated_exact_t_job_test"
            )
            completed = SimpleNamespace(
                returncode=0,
                stdout='RESULT_JSON={"score":{"mean_exact_accuracy":0.5},"seeds":[]}\n',
                stderr="",
            )
            runtime_info = {
                "accelerator": "cuda",
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "gpu_memory_bytes": 85_056_012_288,
                "python_version": "3.13.5",
                "torch_version": "2.12.1",
                "cuda_version": "13.0",
                "driver_version": "600.1",
                "bf16_supported": True,
            }
            stream = io.StringIO()
            with mock.patch.object(
                namespace["subprocess"], "run", return_value=completed
            ) as run_mock, contextlib.redirect_stdout(stream):
                status = namespace["run_research"](root, runtime_info)
            self.assertEqual(status, 0)
            command = run_mock.call_args.args[0]
            self.assertIn("benchmark.runner", command)
            self.assertNotIn("--device", command)
            result = json.loads(stream.getvalue().splitlines()[0])
            self.assertEqual(result["type"], "research_result")
            self.assertEqual(
                result["record"]["research_execution"], "benchmark-diagnostic"
            )
            self.assertEqual(
                result["record"]["result_json"]["score"]["mean_exact_accuracy"],
                0.5,
            )

    def test_exact_t_research_entrypoints_emit_json(self) -> None:
        for experiment_id in MATRICES["exact-t-diagnostics"]:
            with self.subTest(experiment_id=experiment_id):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(get_experiment(experiment_id).source_path()),
                        "--device",
                        "cpu",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                result = json.loads(completed.stdout)
                self.assertEqual(result["parsed_exact_t"], [1, 64])
                self.assertTrue(result["logits_finite"])


if __name__ == "__main__":
    unittest.main()
