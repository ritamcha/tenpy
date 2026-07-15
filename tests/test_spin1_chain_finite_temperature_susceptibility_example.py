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

    def test_run_case_scan_reaches_temperature_targets_with_bounded_mpo_steps(self):
        class FakeHMPO:
            calls = []

            def make_U(self, dt, approx):
                self.calls.append((dt, approx))
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
            calls = []

            @classmethod
            def from_infiniteT(cls, sites, **kwargs):
                cls.calls.append((sites, kwargs))
                return FakePsi()

        class FakePurificationApplyMPO:
            runs = 0
            applied_mpos = []

            def __init__(self, psi, mpo, options):
                self.psi = psi
                self.options = options
                self.initial_mpo = mpo

            def init_env(self, mpo):
                self.applied_mpos.append(mpo)

            def run(self):
                type(self).runs += 1

        temperature_max_k = spin1_chain_ft.temperature_kelvin_from_beta(0.75)
        temperature_min_k = spin1_chain_ft.temperature_kelvin_from_beta(1.25)
        moments = [(0.0, 4.0), (0.0, 8.0)]

        with patch.object(
            spin1_chain_ft,
            "_require_tenpy_purification",
            return_value=(FakePurificationApplyMPO, FakePurificationMPS),
        ), patch.object(spin1_chain_ft, "magnetization_moments", side_effect=moments):
            rows = spin1_chain_ft.run_case_scan(
                "with",
                FakeModel(),
                temperature_min_k=temperature_min_k,
                temperature_max_k=temperature_max_k,
                temperature_points=2,
                dt=0.25,
                chi_max=32,
                svd_min=1.0e-7,
                g_factor=2.0,
            )

        self.assertTrue(np.allclose([row["beta_meV_inv"] for row in rows], [1.25, 0.75]))
        self.assertTrue(rows[0]["temperature_K"] < rows[1]["temperature_K"])
        segment_dts = [-2.0 * call[0].real for call in FakeHMPO.calls[::2]]
        self.assertTrue(np.allclose(segment_dts, [0.25, 0.125, 0.25]))
        self.assertTrue(all(segment_dt <= 0.25 for segment_dt in segment_dts))
        self.assertTrue(all(call[1] == "II" for call in FakeHMPO.calls))
        self.assertEqual(FakePurificationApplyMPO.runs, 6)

    def test_run_case_scan_rejects_invalid_dt_and_g_before_tenpy(self):
        class FakeModel:
            n_sites = 4

        with patch.object(spin1_chain_ft, "_require_tenpy_purification") as require_tenpy:
            with self.assertRaisesRegex(ValueError, "dt must be positive"):
                spin1_chain_ft.run_case_scan(
                    "with", FakeModel(), 2.0, 1200.0, 10, 0.0, 32, 1.0e-7, 2.0
                )
            with self.assertRaisesRegex(ValueError, "g_factor must be positive"):
                spin1_chain_ft.run_case_scan(
                    "with", FakeModel(), 2.0, 1200.0, 10, 0.01, 32, 1.0e-7, 0.0
                )

        require_tenpy.assert_not_called()

    def test_run_susceptibility_scan_expands_both_cases_in_order(self):
        class FakeWithSIA:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeWithoutSIA:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        calls = []

        def fake_run_case_scan(
            case_label,
            model,
            temperature_min_k,
            temperature_max_k,
            temperature_points,
            dt,
            chi_max,
            svd_min,
            g_factor,
        ):
            calls.append((
                case_label,
                model.kwargs,
                temperature_min_k,
                temperature_max_k,
                temperature_points,
                dt,
                chi_max,
                svd_min,
                g_factor,
            ))
            return [{
                "case": case_label,
                "beta_meV_inv": 1.0,
                "temperature_K": spin1_chain_ft.temperature_kelvin_from_beta(1.0),
                "chi_reduced_meV_inv": 1.0,
                "chi_molar_m3_per_mol": spin1_chain_ft.molar_susceptibility_from_reduced(1.0, g_factor),
                "g_factor": g_factor,
                "mz": 0.0,
                "mz2": 4.0,
                "n_sites": 4,
            }]

        with patch.object(spin1_chain_ft, "Spin1ChainWithSIA", FakeWithSIA), patch.object(
            spin1_chain_ft,
            "Spin1ChainWithoutSIA",
            FakeWithoutSIA,
        ), patch.object(spin1_chain_ft, "run_case_scan", side_effect=fake_run_case_scan):
            rows = spin1_chain_ft.run_susceptibility_scan(
                case="both",
                n_sites=6,
                j_iso=-16.1,
                dz=0.379,
                dpp=-0.017,
                bc="periodic",
                temperature_min_k=2.0,
                temperature_max_k=1200.0,
                temperature_points=150,
                dt=0.01,
                chi_max=64,
                svd_min=1.0e-8,
                g_factor=2.1,
            )

        self.assertEqual([row["case"] for row in rows], ["with", "without"])
        self.assertEqual([call[0] for call in calls], ["with", "without"])
        self.assertEqual(calls[0][1]["dz"], 0.379)
        self.assertEqual(calls[0][1]["dpp"], -0.017)
        self.assertEqual(calls[1][1]["bc"], "periodic")
        self.assertEqual(calls[0][2:], (2.0, 1200.0, 150, 0.01, 64, 1.0e-8, 2.1))


if __name__ == "__main__":
    unittest.main()
