import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "spin1_chain_dmrg.py"
SPEC = importlib.util.spec_from_file_location("spin1_chain_dmrg", MODULE_PATH)
spin1_chain_dmrg = importlib.util.module_from_spec(SPEC)
sys.modules["spin1_chain_dmrg"] = spin1_chain_dmrg
SPEC.loader.exec_module(spin1_chain_dmrg)


class ImmediateFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class RecordingExecutor:
    calls = []
    max_workers = None

    def __init__(self, max_workers):
        RecordingExecutor.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args):
        RecordingExecutor.calls.append((fn.__name__, args))
        case = args[0]
        return ImmediateFuture({
            "case": case,
            "label": "with SIA" if case == "with" else "without SIA",
            "energy": 1.0 if case == "with" else 2.0,
            "info": {"E": 1.0 if case == "with" else 2.0},
        })


class Spin1ChainMultiprocessingTest(unittest.TestCase):
    def setUp(self):
        RecordingExecutor.calls = []
        RecordingExecutor.max_workers = None

    def test_selected_cases_expands_both_in_stable_order(self):
        self.assertEqual(spin1_chain_dmrg._selected_cases("both"), ("with", "without"))
        self.assertEqual(spin1_chain_dmrg._selected_cases("with"), ("with",))

    def test_default_model_parameters_use_open_dmrg_boundary_and_full_sia_terms(self):
        model = spin1_chain_dmrg.Spin1ChainWithSIA(n_sites=6)

        self.assertNotIn("bc", model.tenpy_model_params())
        self.assertEqual(model.tenpy_model_params()["bc_x"], "open")
        self.assertEqual(model.tenpy_model_params()["bc_MPS"], "finite")
        self.assertEqual(model.tenpy_model_params()["D"], 0.379)
        self.assertEqual(model.tenpy_model_params()["E"], -0.034)
        self.assertEqual(model.tenpy_model_params()["conserve"], "parity")

    def test_periodic_lattice_boundary_can_be_requested_explicitly(self):
        model = spin1_chain_dmrg.Spin1ChainWithoutSIA(n_sites=6, bc="periodic")

        self.assertEqual(model.tenpy_model_params()["bc_x"], "periodic")

    def test_tenpy_requirement_uses_spin_model_for_periodic_dmrg_mpo(self):
        fake_dmrg = object()
        fake_spin_model = type("FakeSpinModel", (), {})
        fake_spin_chain = type("FakeSpinChain", (), {})
        fake_mps = type("FakeMPS", (), {})
        modules = {
            "tenpy": types.ModuleType("tenpy"),
            "tenpy.algorithms": types.ModuleType("tenpy.algorithms"),
            "tenpy.models": types.ModuleType("tenpy.models"),
            "tenpy.models.spins": types.ModuleType("tenpy.models.spins"),
            "tenpy.networks": types.ModuleType("tenpy.networks"),
            "tenpy.networks.mps": types.ModuleType("tenpy.networks.mps"),
        }
        modules["tenpy.algorithms"].dmrg = fake_dmrg
        modules["tenpy.models.spins"].SpinModel = fake_spin_model
        modules["tenpy.models.spins"].SpinChain = fake_spin_chain
        modules["tenpy.networks.mps"].MPS = fake_mps

        with patch.dict(sys.modules, modules):
            model_class, mps_class, dmrg_module = spin1_chain_dmrg._require_tenpy()

        self.assertIs(model_class, fake_spin_model)
        self.assertIsNot(model_class, fake_spin_chain)
        self.assertIs(mps_class, fake_mps)
        self.assertIs(dmrg_module, fake_dmrg)

    def test_without_sia_reference_has_no_single_ion_terms(self):
        model = spin1_chain_dmrg.Spin1ChainWithoutSIA(n_sites=6)

        self.assertEqual(model.tenpy_model_params()["D"], 0.0)
        self.assertEqual(model.tenpy_model_params()["E"], 0.0)
        self.assertEqual(model.tenpy_model_params()["conserve"], "Sz")

    def test_cli_accepts_converted_single_ion_parameters(self):
        with patch.object(sys, "argv", ["spin1_chain_dmrg.py", "--dz", "0.379", "--dpp", "-0.017"]):
            args = spin1_chain_dmrg.parse_args()

        self.assertEqual(args.dz, 0.379)
        self.assertEqual(args.dpp, -0.017)

    def test_parallel_run_dispatches_both_cases_to_process_pool(self):
        settings = spin1_chain_dmrg.DMRGSettings(chi_max=8, max_sweeps=2)

        with patch.object(spin1_chain_dmrg, "ProcessPoolExecutor", RecordingExecutor):
            results = spin1_chain_dmrg.run_cases(
                "both",
                n_sites=6,
                j_iso=-1.5,
                dz=0.25,
                dpp=-0.01,
                bc="periodic",
                settings=settings,
                parallel=True,
                workers=2,
            )

        self.assertEqual(RecordingExecutor.max_workers, 2)
        self.assertEqual([call[1][0] for call in RecordingExecutor.calls], ["with", "without"])
        self.assertEqual([result["case"] for result in results], ["with", "without"])
        self.assertEqual([result["energy"] for result in results], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
