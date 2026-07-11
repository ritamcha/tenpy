"""Run TenPy DMRG for a spin-1 chain with and without single-ion anisotropy.

Hamiltonians:

    with SIA:    H = -2 J sum_i S_i . S_{i+1} + D sum_i (S_i^z)^2
    without SIA: H = -2 J sum_i S_i . S_{i+1}

Run from the root of a TenPy checkout, for example:

    python examples/spin1_chain_dmrg.py --n-sites 12 --j -16.1 --d 0.252 --case both
    python examples/spin1_chain_dmrg.py --n-sites 12 --bc periodic --case both
    python examples/spin1_chain_dmrg.py --n-sites 12 --case both --parallel
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np


ZERO_STATE = "0.0"
STATE_SZ = {"up": 1, ZERO_STATE: 0, "down": -1}


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _validate_non_negative_real(value: object, name: str) -> float:
    value = _validate_finite_real(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_choice(value: object, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {choices}")
    return value


def total_sz(product_state: list[str] | tuple[str, ...]) -> int:
    try:
        return sum(STATE_SZ[state] for state in product_state)
    except KeyError as exc:
        raise ValueError(f"unknown spin-1 product-state label {exc.args[0]!r}") from exc


def product_state_for_total_sz(n_sites: int, target_sz: int) -> list[str]:
    n_sites = _validate_positive_integer(n_sites, "n_sites")
    if not isinstance(target_sz, Integral):
        raise ValueError("target_sz must be an integer")
    target_sz = int(target_sz)
    if abs(target_sz) > n_sites:
        raise ValueError("abs(target_sz) cannot exceed n_sites for spin-1 sites")

    state = [ZERO_STATE] * n_sites
    offset = 0
    if target_sz > 0:
        state[0] = "up"
        offset = 1
    elif target_sz < 0:
        state[0] = "down"
        offset = 1

    for site in range(offset, n_sites - 1, 2):
        state[site] = "up"
        state[site + 1] = "down"

    while total_sz(state) < target_sz:
        index = state.index("down") if "down" in state else state.index(ZERO_STATE)
        state[index] = ZERO_STATE if state[index] == "down" else "up"
    while total_sz(state) > target_sz:
        index = state.index("up") if "up" in state else state.index(ZERO_STATE)
        state[index] = ZERO_STATE if state[index] == "up" else "down"
    return state


def _copy_mps_for_orthogonalization(psi):
    return psi.copy() if hasattr(psi, "copy") else psi


def _require_tenpy():
    """Return TenPy constructors or explain how to run this script."""
    try:
        from tenpy.algorithms import dmrg
        from tenpy.models.spins import SpinModel
        from tenpy.networks.mps import MPS
    except Exception as exc:
        raise RuntimeError(
            "TenPy is unavailable. Run this script from an installed TenPy environment, "
            "for example after `python -m pip install -e .` in the cloned repository."
        ) from exc
    return SpinModel, MPS, dmrg


@dataclass(frozen=True)
class DMRGSettings:
    """Default DMRG controls for a finite-chain starting run."""

    chi_max: int = 100
    max_sweeps: int = 20
    svd_min: float = 1.0e-10
    trunc_cut: float = 1.0e-10
    mixer: bool = True
    active_sites: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "chi_max", _validate_positive_integer(self.chi_max, "chi_max"))
        object.__setattr__(self, "max_sweeps", _validate_positive_integer(self.max_sweeps, "max_sweeps"))
        object.__setattr__(self, "svd_min", _validate_non_negative_real(self.svd_min, "svd_min"))
        object.__setattr__(self, "trunc_cut", _validate_non_negative_real(self.trunc_cut, "trunc_cut"))
        if not isinstance(self.mixer, bool):
            raise ValueError("mixer must be a boolean")
        if self.active_sites not in (1, 2):
            raise ValueError("active_sites must be 1 or 2")

    def to_options(self) -> dict[str, object]:
        return {
            "mixer": self.mixer,
            "active_sites": self.active_sites,
            "max_sweeps": self.max_sweeps,
            "trunc_params": {
                "chi_max": self.chi_max,
                "svd_min": self.svd_min,
                "trunc_cut": self.trunc_cut,
            },
        }


def _merged_options(base: dict[str, object], updates: dict[str, object] | None) -> dict[str, object]:
    if updates is None:
        return base
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


class _Spin1ChainDMRG:
    """Shared TenPy construction for spin-1 DMRG runs."""

    n_sites: int
    j_iso: float
    bc: str
    bc_MPS: str
    conserve: str
    dmrg_settings: DMRGSettings

    @property
    def sia_d(self) -> float:
        raise NotImplementedError

    def tenpy_model_params(self) -> dict[str, object]:
        coupling = -2.0 * self.j_iso
        return {
            "L": self.n_sites,
            "S": 1,
            "Jx": coupling,
            "Jy": coupling,
            "Jz": coupling,
            "D": self.sia_d,
            "E": 0.0,
            "bc_x": self.bc,
            "bc_MPS": self.bc_MPS,
            "conserve": self.conserve,
        }

    def dmrg_options(self, updates: dict[str, object] | None = None) -> dict[str, object]:
        return _merged_options(self.dmrg_settings.to_options(), updates)

    def initial_product_state(self, pattern: tuple[str, ...] = ("up", "down")) -> list[str]:
        if len(pattern) == 0:
            raise ValueError("pattern must contain at least one local state label")
        return [pattern[site % len(pattern)] for site in range(self.n_sites)]

    def build_model(self):
        SpinModel, _, _ = _require_tenpy()
        return SpinModel(self.tenpy_model_params())

    def build_initial_mps(self, model=None, product_state: list[str] | tuple[str, ...] | None = None):
        _, MPS, _ = _require_tenpy()
        if model is None:
            model = self.build_model()
        if product_state is None:
            product_state = self.initial_product_state()
        if len(product_state) != self.n_sites:
            raise ValueError("product_state must have one entry per site")
        return MPS.from_product_state(model.lat.mps_sites(), list(product_state), bc=model.lat.bc_MPS)

    def run_dmrg(
        self,
        dmrg_options: dict[str, object] | None = None,
        product_state: list[str] | tuple[str, ...] | None = None,
        orthogonal_to: list[object] | None = None,
    ) -> dict[str, object]:
        _, _, dmrg = _require_tenpy()
        model = self.build_model()
        psi = self.build_initial_mps(model=model, product_state=product_state)
        info = dmrg.run(psi, model, self.dmrg_options(dmrg_options), orthogonal_to=orthogonal_to or [])
        return {"info": info, "psi": psi, "model": model}

    def run_lowest_states(
        self,
        num_states: int = 20,
        target_sz: int = 0,
        dmrg_options: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        num_states = _validate_positive_integer(num_states, "num_states")
        product_state = product_state_for_total_sz(self.n_sites, target_sz)
        _, _, dmrg = _require_tenpy()
        model = self.build_model()
        previous_states = []
        results = []

        for state_index in range(num_states):
            psi = self.build_initial_mps(model=model, product_state=product_state)
            info = dmrg.run(
                psi,
                model,
                self.dmrg_options(dmrg_options),
                orthogonal_to=list(previous_states),
            )
            energy = float(info["E"])
            results.append({
                "state_index": state_index,
                "target_sz": int(target_sz),
                "energy": energy,
                "info": info,
                "psi": psi,
            })
            previous_states.append(_copy_mps_for_orthogonalization(psi))
        return results


@dataclass(frozen=True)
class Spin1ChainWithSIA(_Spin1ChainDMRG):
    """TenPy DMRG model for H = -2 J sum_i S_i.S_{i+1} + D sum_i (Sz_i)^2."""

    n_sites: int = 12
    j_iso: float = -16.1
    d: float = 0.252
    bc: str = "periodic"
    bc_MPS: str = "finite"
    conserve: str = "Sz"
    dmrg_settings: DMRGSettings = DMRGSettings()

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_sites", _validate_positive_integer(self.n_sites, "n_sites"))
        object.__setattr__(self, "j_iso", _validate_finite_real(self.j_iso, "j_iso"))
        object.__setattr__(self, "d", _validate_finite_real(self.d, "d"))
        object.__setattr__(self, "bc", _validate_choice(self.bc, "bc", ("open", "periodic")))
        object.__setattr__(self, "bc_MPS", _validate_choice(self.bc_MPS, "bc_MPS", ("finite", "infinite")))
        object.__setattr__(self, "conserve", _validate_choice(self.conserve, "conserve", ("Sz", "parity", "None")))
        if not isinstance(self.dmrg_settings, DMRGSettings):
            raise ValueError("dmrg_settings must be a DMRGSettings instance")

    @property
    def sia_d(self) -> float:
        return self.d


@dataclass(frozen=True)
class Spin1ChainWithoutSIA(_Spin1ChainDMRG):
    """TenPy DMRG model for H = -2 J sum_i S_i.S_{i+1} with D fixed to zero."""

    n_sites: int = 12
    j_iso: float = -16.1
    bc: str = "periodic"
    bc_MPS: str = "finite"
    conserve: str = "Sz"
    dmrg_settings: DMRGSettings = DMRGSettings()

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_sites", _validate_positive_integer(self.n_sites, "n_sites"))
        object.__setattr__(self, "j_iso", _validate_finite_real(self.j_iso, "j_iso"))
        object.__setattr__(self, "bc", _validate_choice(self.bc, "bc", ("open", "periodic")))
        object.__setattr__(self, "bc_MPS", _validate_choice(self.bc_MPS, "bc_MPS", ("finite", "infinite")))
        object.__setattr__(self, "conserve", _validate_choice(self.conserve, "conserve", ("Sz", "parity", "None")))
        if not isinstance(self.dmrg_settings, DMRGSettings):
            raise ValueError("dmrg_settings must be a DMRGSettings instance")

    @property
    def sia_d(self) -> float:
        return 0.0


def run_case(model: _Spin1ChainDMRG, label: str) -> dict[str, object]:
    result = model.run_dmrg()
    energy = result["info"]["E"]
    print(f"{label}: E = {energy:.16f}")
    return result


def _selected_cases(case: str) -> tuple[str, ...]:
    if case == "both":
        return ("with", "without")
    return (case,)


def _build_case_model(
    case: str,
    n_sites: int,
    j_iso: float,
    d: float,
    bc: str,
    settings: DMRGSettings,
) -> tuple[_Spin1ChainDMRG, str]:
    if case == "with":
        return (
            Spin1ChainWithSIA(n_sites=n_sites, j_iso=j_iso, d=d, bc=bc, dmrg_settings=settings),
            "with SIA",
        )
    if case == "without":
        return (
            Spin1ChainWithoutSIA(n_sites=n_sites, j_iso=j_iso, bc=bc, dmrg_settings=settings),
            "without SIA",
        )
    raise ValueError("case must be one of ('with', 'without')")


def _run_case_worker(
    case: str,
    n_sites: int,
    j_iso: float,
    d: float,
    bc: str,
    settings: DMRGSettings,
) -> dict[str, object]:
    model, label = _build_case_model(case, n_sites, j_iso, d, bc, settings)
    result = model.run_dmrg()
    return {
        "case": case,
        "label": label,
        "energy": float(result["info"]["E"]),
        "info": result["info"],
    }


def run_cases(
    case: str,
    n_sites: int,
    j_iso: float,
    d: float,
    bc: str,
    settings: DMRGSettings,
    *,
    parallel: bool = False,
    workers: int | None = None,
) -> list[dict[str, object]]:
    cases = _selected_cases(case)
    if parallel and len(cases) > 1:
        max_workers = workers or len(cases)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_run_case_worker, selected_case, n_sites, j_iso, d, bc, settings)
                for selected_case in cases
            ]
            results = [future.result() for future in futures]
        for result in results:
            print(f"{result['label']}: E = {result['energy']:.16f}")
        return results

    results = []
    for selected_case in cases:
        model, label = _build_case_model(selected_case, n_sites, j_iso, d, bc, settings)
        result = run_case(model, label)
        results.append({
            "case": selected_case,
            "label": label,
            "energy": float(result["info"]["E"]),
            "info": result["info"],
        })
    return results


def write_spectrum_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    fieldnames = ["case", "label", "state_index", "target_sz", "energy"]
    with Path(output_path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def run_lowest_state_cases(
    case: str,
    n_sites: int,
    j_iso: float,
    d: float,
    bc: str,
    settings: DMRGSettings,
    *,
    num_states: int,
    target_sz: int,
) -> list[dict[str, object]]:
    rows = []
    for selected_case in _selected_cases(case):
        model, label = _build_case_model(selected_case, n_sites, j_iso, d, bc, settings)
        results = model.run_lowest_states(num_states=num_states, target_sz=target_sz)
        for result in results:
            row = {
                "case": selected_case,
                "label": label,
                "state_index": result["state_index"],
                "target_sz": result["target_sz"],
                "energy": result["energy"],
            }
            rows.append(row)
            print(
                f"{label}, state {row['state_index']}, Sz={target_sz}: "
                f"E = {row['energy']:.16f}"
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sites", type=int, default=12, help="Number of spin-1 sites.")
    parser.add_argument("--j", type=float, default=-16.1, help="J in H = -2 J sum_i S_i.S_{i+1}.")
    parser.add_argument("--d", type=float, default=0.252, help="Single-ion anisotropy D.")
    parser.add_argument("--bc", choices=("open", "periodic"), default="periodic", help="Lattice boundary condition.")
    parser.add_argument("--chi-max", type=int, default=100, help="Maximum MPS bond dimension.")
    parser.add_argument("--max-sweeps", type=int, default=20, help="Maximum number of DMRG sweeps.")
    parser.add_argument("--case", choices=("with", "without", "both"), default="both")
    parser.add_argument("--parallel", action="store_true", help="Run with/without SIA cases in separate processes.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes for --parallel.")
    parser.add_argument("--num-states", type=int, default=1, help="Number of lowest states to calculate per case.")
    parser.add_argument("--target-sz", type=int, default=0, help="Total Sz sector for --num-states excited-state DMRG.")
    parser.add_argument(
        "--spectrum-csv",
        default="spin1_chain_lowest_states.csv",
        help="CSV output path used when --num-states is greater than 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = DMRGSettings(chi_max=args.chi_max, max_sweeps=args.max_sweeps)
    if args.workers is not None:
        _validate_positive_integer(args.workers, "workers")
    _validate_positive_integer(args.num_states, "num_states")
    if args.num_states > 1:
        rows = run_lowest_state_cases(
            args.case,
            args.n_sites,
            args.j,
            args.d,
            args.bc,
            settings,
            num_states=args.num_states,
            target_sz=args.target_sz,
        )
        write_spectrum_csv(rows, args.spectrum_csv)
        print(f"Wrote {args.spectrum_csv}")
        return
    run_cases(
        args.case,
        args.n_sites,
        args.j,
        args.d,
        args.bc,
        settings,
        parallel=args.parallel,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
