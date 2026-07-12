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
    def test_susceptibility_uses_total_magnetization_fluctuations(self):
        chi = spin1_chain_ft.susceptibility_from_moments(beta=2.0, n_sites=4, mz=1.0, mz2=5.0)

        self.assertEqual(chi, 2.0)

    def test_zero_beta_reports_zero_susceptibility(self):
        chi = spin1_chain_ft.susceptibility_from_moments(beta=0.0, n_sites=4, mz=1.0, mz2=5.0)

        self.assertEqual(chi, 0.0)

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

    def test_run_case_scan_starts_at_infinite_temperature_and_cools_by_two_dt(self):
        class FakeLat:
            bc_MPS = "finite"
            mps_unit_cell_width = 4

            def mps_sites(self):
                return ["site0", "site1", "site2", "site3"]

        class FakeBuiltModel:
            lat = FakeLat()

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

        class FakePurificationTEBD:
            def __init__(self, psi, model, options):
                self.psi = psi
                FakePurificationTEBD.options = options
                FakePurificationTEBD.steps = []

            def run_imaginary(self, dt):
                FakePurificationTEBD.steps.append(dt)

        FakePurificationMPS.calls = []
        moments = [(0.0, 4.0), (0.5, 5.0), (1.0, 6.0)]

        with patch.object(
            spin1_chain_ft,
            "_require_tenpy_purification",
            return_value=(FakePurificationTEBD, FakePurificationMPS),
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
        self.assertEqual(FakePurificationTEBD.steps, [0.25, 0.25])
        self.assertEqual(FakePurificationTEBD.options["trunc_params"]["chi_max"], 32)
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
