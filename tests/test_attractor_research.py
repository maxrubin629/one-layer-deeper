from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RESEARCH_EXPERIMENTS = (
    "v4.2_attractor_implicit_gradients_research",
    "v4.3_equilibrium_internalization_research",
    "v4.4_discrete_cycle_solver_research",
    "v4.5_analytic_invariant_orbit_research",
)


def load_research_module(experiment_id: str):
    path = EXPERIMENTS / experiment_id / "research.py"
    module_name = f"test_{experiment_id.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AttractorResearchLayoutTests(unittest.TestCase):
    def test_every_research_experiment_is_complete_and_blocked(self) -> None:
        required = {
            "README.md",
            "experiment.json",
            "research.py",
            "BLOCKED_FROM_SUBMISSION",
        }
        for experiment_id in RESEARCH_EXPERIMENTS:
            with self.subTest(experiment_id=experiment_id):
                directory = EXPERIMENTS / experiment_id
                self.assertEqual(
                    {path.name for path in directory.iterdir() if path.is_file()},
                    required,
                )
                self.assertFalse((directory / "submission.py").exists())
                marker = (directory / "BLOCKED_FROM_SUBMISSION").read_text(
                    encoding="utf-8"
                )
                self.assertIn("RESEARCH ONLY", marker)
                self.assertNotIn(
                    "SUBMISSION =",
                    (directory / "research.py").read_text(encoding="utf-8"),
                )

    def test_manifests_have_stable_research_contract(self) -> None:
        for experiment_id in RESEARCH_EXPERIMENTS:
            with self.subTest(experiment_id=experiment_id):
                manifest = json.loads(
                    (EXPERIMENTS / experiment_id / "experiment.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(manifest["id"], experiment_id)
                self.assertEqual(manifest["eligibility"], "research-only")
                self.assertEqual(manifest["entrypoint"], "research.py")
                self.assertEqual(manifest["family"], "attractor-research")
                for required in (
                    "version",
                    "parent",
                    "title",
                    "description",
                    "metrics",
                ):
                    self.assertIn(required, manifest)


class AttractorGradientResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_research_module(
            "v4.2_attractor_implicit_gradients_research"
        )

    def test_root_solvers_agree_and_implicit_gradient_matches_unrolling(self) -> None:
        result = self.module.run_experiment(explicit_iterations=100)
        self.assertLess(
            result["solvers"]["picard"]["fixed_point_residual"], 1e-10
        )
        self.assertLess(
            result["solvers"]["anderson"]["fixed_point_residual"], 1e-10
        )
        self.assertLess(result["solvers"]["solution_max_distance"], 1e-10)
        self.assertLess(
            result["gradients"]["implicit"]["relative_error_to_explicit"],
            1e-9,
        )
        self.assertLess(
            result["gradients"]["phantom"]["relative_error_to_explicit"],
            result["gradients"]["one_step"]["relative_error_to_explicit"],
        )
        matrix = result["solver_matrix"]
        self.assertEqual(len(matrix), 3 * 2 * 2 * 2)
        self.assertEqual(
            {item["initialization"] for item in matrix.values()},
            {"zero", "gaussian", "learned_proposal"},
        )
        self.assertEqual(
            {item["conditioning"] for item in matrix.values()},
            {"initial_only", "persistent"},
        )
        self.assertEqual(
            {item["solver"] for item in matrix.values()}, {"picard", "anderson"}
        )
        self.assertEqual(
            {item["stopping"] for item in matrix.values()},
            {"fixed_12", "residual"},
        )
        self.assertTrue(
            all(
                item["fixed_point_residual"] < 1e-10
                for item in matrix.values()
                if item["stopping"] == "residual"
            )
        )
        self.assertLess(result["jacobian_estimate"]["spectral_radius"], 1.0)
        json.dumps(result)


class EquilibriumInternalizationResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_research_module(
            "v4.3_equilibrium_internalization_research"
        )

    def test_refinement_reduces_proposal_error_and_residual(self) -> None:
        result = self.module.run_experiment(training_steps=120)
        evaluations = result["evaluations"]
        ordered = [
            evaluations["k0_proposal_only"],
            evaluations["k1"],
            evaluations["k3_trained"],
            evaluations["k25_full_solve"],
        ]
        errors = [float(item["equilibrium_mse"]) for item in ordered]
        residuals = [float(item["fixed_point_residual"]) for item in ordered]
        self.assertTrue(all(left > right for left, right in zip(errors, errors[1:])))
        self.assertTrue(
            all(left > right for left, right in zip(residuals, residuals[1:]))
        )
        self.assertLess(
            result["training_loss_trace"][-1], result["training_loss_trace"][0]
        )
        self.assertEqual(
            evaluations["k64_extra_depth"]["equilibrium_mse"],
            evaluations["k25_full_solve"]["equilibrium_mse"],
        )
        json.dumps(result)


class DiscreteCycleResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_research_module("v4.4_discrete_cycle_solver_research")

    def test_rotation_and_permutation_cycles_close(self) -> None:
        result = self.module.run_experiment(
            cycle_lengths=(2, 4), max_iterations=80
        )
        self.assertLess(result["maximum_cycle_residual"], 1e-6)
        for task in result["tasks"].values():
            for metrics in task.values():
                self.assertLess(metrics["closure_error"], 1e-6)
                self.assertLess(metrics["anchor_error"], 1e-6)
                self.assertGreater(metrics["minimum_state_norm"], 0.99)
        json.dumps(result)


class AnalyticInvariantOrbitResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_research_module(
            "v4.5_analytic_invariant_orbit_research"
        )

    def test_fourier_curve_satisfies_invariance_and_transverse_stability(self) -> None:
        result = self.module.run_experiment(
            phase_count=48,
            max_iterations=80,
        )
        self.assertLess(result["invariance_residual_max"], 1e-6)
        self.assertLess(result["curve_rmse"], 1e-6)
        stability = result["transverse_stability"]
        self.assertAlmostEqual(
            stability["empirical_transverse_rate"],
            result["transverse_rate"],
            places=9,
        )
        self.assertLess(stability["final_distance"], stability["initial_distance"])
        self.assertEqual(result["phase_gauge"], "explicit theta coordinate")
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
