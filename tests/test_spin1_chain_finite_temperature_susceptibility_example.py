import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
MODULE_PATH = EXAMPLES_DIR / "spin1_chain_finite_temperature_susceptibility.py"
sys.path.insert(0, str(EXAMPLES_DIR))
SPEC = importlib.util.spec_from_file_location("spin1_chain_finite_temperature_susceptibility", MODULE_PATH)
spin1_chain_ft = importlib.util.module_from_spec(SPEC)
sys.modules["spin1_chain_finite_temperature_susceptibility"] = spin1_chain_ft
SPEC.loader.exec_module(spin1_chain_ft)


class Spin1ChainFiniteTemperatureSusceptibilityTest(unittest.TestCase):
    def test_temperature_beta_round_trip_uses_mev_and_kelvin(self):
        beta = spin1_chain_ft.beta_from_temperature_kelvin(500.0)

        self.assertAlmostEqual(beta, 1.0 / (spin1_chain_ft.K_B_MEV_PER_K * 500.0), places=14)
        self.assertAlmostEqual(spin1_chain_ft.temperature_kelvin_from_beta(beta), 500.0, places=12)

    def test_temperature_grid_is_logarithmic_and_descending(self):
        temperatures = spin1_chain_ft.temperature_grid_kelvin(2.0, 1200.0, 5)

        self.assertEqual(len(temperatures), 5)
        self.assertAlmostEqual(temperatures[0], 1200.0)
        self.assertAlmostEqual(temperatures[-1], 2.0)
        self.assertTrue(np.all(np.diff(temperatures) < 0.0))
        self.assertTrue(np.allclose(np.diff(np.log(temperatures)), np.diff(np.log(temperatures))[0]))

    def test_reduced_susceptibility_uses_total_magnetization_fluctuations(self):
        chi = spin1_chain_ft.reduced_susceptibility_from_moments(
            beta_mev_inv=2.0,
            n_sites=4,
            mz=1.0,
            mz2=5.0,
        )

        self.assertEqual(chi, 2.0)

    def test_molar_susceptibility_uses_si_prefactor_and_g_squared(self):
        chi_g2 = spin1_chain_ft.molar_susceptibility_from_reduced(1.0, g_factor=2.0)
        chi_g4 = spin1_chain_ft.molar_susceptibility_from_reduced(1.0, g_factor=4.0)
        expected = (
            spin1_chain_ft.VACUUM_PERMEABILITY
            * spin1_chain_ft.AVOGADRO_CONSTANT
            * (2.0 * spin1_chain_ft.BOHR_MAGNETON_J_PER_T) ** 2
            / spin1_chain_ft.MEV_TO_JOULE
        )

        self.assertAlmostEqual(chi_g2, expected, places=18)
        self.assertAlmostEqual(chi_g4 / chi_g2, 4.0, places=14)

    def test_physical_unit_helpers_reject_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "temperature_k must be positive"):
            spin1_chain_ft.beta_from_temperature_kelvin(0.0)
        with self.assertRaisesRegex(ValueError, "maximum_k must be greater than minimum_k"):
            spin1_chain_ft.temperature_grid_kelvin(10.0, 10.0, 5)
        with self.assertRaisesRegex(ValueError, "points must be at least 2"):
            spin1_chain_ft.temperature_grid_kelvin(2.0, 1200.0, 1)
        with self.assertRaisesRegex(ValueError, "g_factor must be positive"):
            spin1_chain_ft.molar_susceptibility_from_reduced(1.0, g_factor=0.0)

    def test_magnetization_moments_sum_sites_and_correlations(self):
        calls = []

        class FakePsi:
            def expectation_value(self, operator):
                calls.append(("expectation_value", operator))
                return np.array([0.25, -0.5, 0.75])

            def correlation_function(self, op1, op2):
                calls.append(("correlation_function", op1, op2))
                return np.array([
                    [1.0, 0.1, 0.2],
                    [0.1, 2.0, 0.3],
                    [0.2, 0.3, 3.0],
                ])

        mz, mz2 = spin1_chain_ft.magnetization_moments(FakePsi())

        self.assertEqual(calls, [("expectation_value", "Sz"), ("correlation_function", "Sz", "Sz")])
        self.assertEqual(mz, 0.5)
        self.assertEqual(mz2, 7.2)

    def test_rows_are_written_as_csv(self):
        rows = [{
            "case": "with",
            "beta": 0.5,
            "temperature": 2.0,
            "chi": 1.25,
            "mz": 0.0,
            "mz2": 10.0,
            "n_sites": 8,
        }]
        output = EXAMPLES_DIR.parent / "tests" / "tmp_spin1_susceptibility.csv"
        try:
            spin1_chain_ft.write_susceptibility_csv(rows, output)
            content = output.read_text(encoding="utf-8")
        finally:
            output.unlink(missing_ok=True)

        self.assertIn("case,beta,temperature,chi,mz,mz2,n_sites", content)
        self.assertIn("with,0.5,2.0,1.25,0.0,10.0,8", content)

    def test_default_cli_values(self):
        with patch.object(sys, "argv", ["spin1_chain_finite_temperature_susceptibility.py"]):
            args = spin1_chain_ft.parse_args()

        self.assertEqual(args.case, "with")
        self.assertEqual(args.csv, "spin1_chain_susceptibility_vs_temperature.csv")

    def test_run_case_scan_uses_mpo_purification_and_cools_by_two_dt(self):
        class FakeHMPO:
            def make_U(self, dt, approx):
                FakeHMPO.calls.append((dt, approx))
                return f"U({dt},{approx})"

        class FakeLat:
            bc_MPS = "finite"
            mps_unit_cell_width = 4

            def mps_sites(self):
                return ["site0", "site1", "site2", "site3"]

        class FakeBuiltModel:
            lat = FakeLat()
            H_MPO = FakeHMPO()

        class FakeModel:
            n_sites = 4

            def build_model(self):
                return FakeBuiltModel()

        class FakePsi:
            pass

        class FakePurificationMPS:
            @classmethod
            def from_infiniteT(cls, sites, **kwargs):
                FakePurificationMPS.calls.append((sites, kwargs))
                return FakePsi()

        class FakePurificationApplyMPO:
            def __init__(self, psi, mpo, options):
                self.psi = psi
                FakePurificationApplyMPO.options = options
                FakePurificationApplyMPO.initial_mpo = mpo
                FakePurificationApplyMPO.mpos = []
                FakePurificationApplyMPO.runs = 0

            def init_env(self, mpo):
                FakePurificationApplyMPO.mpos.append(mpo)

            def run(self):
                FakePurificationApplyMPO.runs += 1

        FakeHMPO.calls = []
        FakePurificationMPS.calls = []
        moments = [(0.0, 4.0), (0.5, 5.0), (1.0, 6.0)]

        with patch.object(
            spin1_chain_ft,
            "_require_tenpy_purification",
            return_value=(FakePurificationApplyMPO, FakePurificationMPS),
            create=True,
        ), patch.object(spin1_chain_ft, "magnetization_moments", side_effect=moments):
            rows = spin1_chain_ft.run_case_scan(
                "with",
                FakeModel(),
                beta_max=1.0,
                dt=0.25,
                chi_max=32,
                svd_min=1.0e-7,
            )

        self.assertEqual([row["beta"] for row in rows], [0.0, 0.5, 1.0])
        self.assertEqual([row["temperature"] for row in rows], [float("inf"), 2.0, 1.0])
        self.assertEqual([row["chi"] for row in rows], [0.0, 0.59375, 1.25])
        self.assertEqual(FakeHMPO.calls, [(-0.125 - 0.125j, "II"), (-0.125 + 0.125j, "II")])
        self.assertEqual(FakePurificationApplyMPO.initial_mpo, "U((-0.125-0.125j),II)")
        self.assertEqual(FakePurificationApplyMPO.mpos, [
            "U((-0.125-0.125j),II)",
            "U((-0.125+0.125j),II)",
            "U((-0.125-0.125j),II)",
            "U((-0.125+0.125j),II)",
        ])
        self.assertEqual(FakePurificationApplyMPO.runs, 4)
        self.assertEqual(FakePurificationApplyMPO.options["trunc_params"]["chi_max"], 32)
        self.assertEqual(FakePurificationMPS.calls, [
            (["site0", "site1", "site2", "site3"], {"bc": "finite", "unit_cell_width": 4})
        ])

    def test_run_susceptibility_scan_expands_both_cases_in_order(self):
        class FakeWithSIA:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeWithoutSIA:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        calls = []

        def fake_run_case_scan(case_label, model, beta_max, dt, chi_max, svd_min):
            calls.append((case_label, model.kwargs, beta_max, dt, chi_max, svd_min))
            return [{"case": case_label, "beta": 0.0, "temperature": float("inf"), "chi": 0.0, "mz": 0.0, "mz2": 0.0, "n_sites": 6}]

        with patch.object(spin1_chain_ft, "Spin1ChainWithSIA", FakeWithSIA, create=True), patch.object(
            spin1_chain_ft,
            "Spin1ChainWithoutSIA",
            FakeWithoutSIA,
            create=True,
        ), patch.object(spin1_chain_ft, "run_case_scan", side_effect=fake_run_case_scan, create=True):
            rows = spin1_chain_ft.run_susceptibility_scan(
                case="both",
                n_sites=6,
                j_iso=-16.1,
                dz=0.379,
                dpp=-0.017,
                bc="periodic",
                beta_max=2.0,
                dt=0.1,
                chi_max=64,
                svd_min=1.0e-8,
            )

        self.assertEqual([row["case"] for row in rows], ["with", "without"])
        self.assertEqual([call[0] for call in calls], ["with", "without"])
        self.assertEqual(calls[0][1]["dz"], 0.379)
        self.assertEqual(calls[0][1]["dpp"], -0.017)
        self.assertEqual(calls[1][1]["bc"], "periodic")


if __name__ == "__main__":
    unittest.main()
