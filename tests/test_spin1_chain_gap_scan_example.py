import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
MODULE_PATH = EXAMPLES_DIR / "spin1_chain_gap_scan.py"
sys.path.insert(0, str(EXAMPLES_DIR))
SPEC = importlib.util.spec_from_file_location("spin1_chain_gap_scan", MODULE_PATH)
spin1_chain_gap_scan = importlib.util.module_from_spec(SPEC)
sys.modules["spin1_chain_gap_scan"] = spin1_chain_gap_scan
SPEC.loader.exec_module(spin1_chain_gap_scan)


class Spin1ChainGapScanTest(unittest.TestCase):
    def test_parse_n_values_accepts_commas_and_spaces(self):
        self.assertEqual(spin1_chain_gap_scan.parse_n_values("6, 8 10,12"), [6, 8, 10, 12])

    def test_parse_n_values_rejects_non_positive_values(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            spin1_chain_gap_scan.parse_n_values("6,0,8")

    def test_product_state_has_requested_total_sz(self):
        sz0_state = spin1_chain_gap_scan.product_state_for_total_sz(6, 0)
        sz1_state = spin1_chain_gap_scan.product_state_for_total_sz(6, 1)

        self.assertEqual(spin1_chain_gap_scan.total_sz(sz0_state), 0)
        self.assertEqual(spin1_chain_gap_scan.total_sz(sz1_state), 1)
        self.assertNotIn("0", sz1_state)
        self.assertIn("0.0", sz1_state)

    def test_run_gap_scan_computes_triplet_minus_singlet_for_both_hamiltonians(self):
        class FakeWithSIA:
            def __init__(self, **kwargs):
                self.n_sites = kwargs["n_sites"]

            def run_dmrg(self, product_state):
                sector = spin1_chain_gap_scan.total_sz(product_state)
                return {"info": {"E": -10.0 * self.n_sites + sector * 1.25}}

        class FakeWithoutSIA:
            def __init__(self, **kwargs):
                self.n_sites = kwargs["n_sites"]

            def run_dmrg(self, product_state):
                sector = spin1_chain_gap_scan.total_sz(product_state)
                return {"info": {"E": -9.0 * self.n_sites + sector * 1.75}}

        with patch.object(spin1_chain_gap_scan, "Spin1ChainWithSIA", FakeWithSIA), patch.object(
            spin1_chain_gap_scan,
            "Spin1ChainWithoutSIA",
            FakeWithoutSIA,
        ):
            rows = spin1_chain_gap_scan.run_gap_scan(
                n_values=[6, 8],
                j_iso=-16.1,
                dz=0.379,
                dpp=-0.017,
                bc="periodic",
                settings=spin1_chain_gap_scan.DMRGSettings(chi_max=8, max_sweeps=2),
            )

        self.assertEqual([row["n_sites"] for row in rows], [6, 8])
        self.assertEqual([row["gap_with_sia"] for row in rows], [1.25, 1.25])
        self.assertEqual([row["gap_without_sia"] for row in rows], [1.75, 1.75])

    def test_rows_are_written_as_csv(self):
        rows = [{
            "n_sites": 6,
            "energy_with_sia_sz0": -12.0,
            "energy_with_sia_sz1": -10.5,
            "gap_with_sia": 1.5,
            "energy_without_sia_sz0": -9.0,
            "energy_without_sia_sz1": -7.0,
            "gap_without_sia": 2.0,
            "gap_difference": -0.5,
        }]

        output = EXAMPLES_DIR.parent / "tests" / "tmp_gap_scan.csv"
        try:
            spin1_chain_gap_scan.write_gap_csv(rows, output)
            content = output.read_text(encoding="utf-8")
        finally:
            output.unlink(missing_ok=True)

        self.assertIn("n_sites,energy_with_sia_sz0,energy_with_sia_sz1,gap_with_sia", content)
        self.assertIn("energy_without_sia_sz0,energy_without_sia_sz1,gap_without_sia,gap_difference", content)
        self.assertIn("6,-12.0,-10.5,1.5,-9.0,-7.0,2.0,-0.5", content)

    def test_default_plot_output_is_png(self):
        with patch.object(sys, "argv", ["spin1_chain_gap_scan.py"]):
            args = spin1_chain_gap_scan.parse_args()

        self.assertEqual(args.plot, "spin1_chain_gap_vs_n.png")

    def test_plot_gap_writes_png_with_matplotlib(self):
        rows = [
            {"n_sites": 6, "gap_with_sia": 1.5, "gap_without_sia": 2.0},
            {"n_sites": 8, "gap_with_sia": 1.2, "gap_without_sia": 1.8},
        ]
        output = EXAMPLES_DIR.parent / "tests" / "tmp_gap_plot.png"
        fake_matplotlib = types.ModuleType("matplotlib")
        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        calls = {"backend": None, "plots": [], "legend": False, "closed": False}

        def use(backend):
            calls["backend"] = backend

        class FakeFigure:
            def tight_layout(self):
                pass

            def savefig(self, path, dpi):
                Path(path).write_bytes(b"PNG")
                calls["dpi"] = dpi

        class FakeAxes:
            def plot(self, n_values, gaps, **kwargs):
                calls["plots"].append((list(n_values), list(gaps), kwargs["label"]))

            def set_xlabel(self, _label):
                pass

            def set_ylabel(self, _label):
                pass

            def set_title(self, _title):
                pass

            def legend(self):
                calls["legend"] = True

            def grid(self, *_args, **_kwargs):
                pass

        def subplots(figsize):
            calls["figsize"] = figsize
            return FakeFigure(), FakeAxes()

        fake_matplotlib.use = use
        fake_pyplot.subplots = subplots
        fake_pyplot.close = lambda _fig: calls.__setitem__("closed", True)

        try:
            with patch.dict(sys.modules, {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot}):
                spin1_chain_gap_scan.plot_gap(rows, output)
            content = output.read_bytes()
        finally:
            output.unlink(missing_ok=True)

        self.assertEqual(content, b"PNG")
        self.assertEqual(calls["backend"], "Agg")
        self.assertEqual([plot[2] for plot in calls["plots"]], ["with SIA", "D=0"])
        self.assertTrue(calls["legend"])
        self.assertTrue(calls["closed"])


if __name__ == "__main__":
    unittest.main()
