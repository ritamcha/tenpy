"""Scan spin-chain sizes and plot the SIA energy gap from TenPy DMRG.

This script compares the two Hamiltonians implemented in spin1_chain_dmrg.py:

    with SIA:    H = -2 J sum_i S_i . S_{i+1} + D sum_i (S_i^z)^2
    without SIA: H = -2 J sum_i S_i . S_{i+1}

The plotted gap is

    gap(N) = E_without_SIA(N) - E_with_SIA(N)

Run from the root of a TenPy checkout, for example:

    python examples/spin1_chain_gap_scan.py --n-sites 6,8,10,12,14,16
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

from spin1_chain_dmrg import DMRGSettings, run_cases


def parse_n_values(raw_values: str) -> list[int]:
    values = []
    for token in re.split(r"[\s,]+", raw_values.strip()):
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"n-sites values must be integers; got {token!r}") from exc
        if value <= 0:
            raise ValueError("n-sites values must be positive integers")
        values.append(value)
    if not values:
        raise ValueError("at least one n-sites value is required")
    return values


def _energy_by_case(results: Iterable[dict[str, object]]) -> dict[str, float]:
    energies = {}
    for result in results:
        energies[str(result["case"])] = float(result["energy"])
    missing = {"with", "without"} - set(energies)
    if missing:
        raise RuntimeError(f"DMRG result is missing cases: {sorted(missing)}")
    return energies


def run_gap_scan(
    n_values: list[int],
    j_iso: float,
    d: float,
    bc: str,
    settings: DMRGSettings,
    *,
    parallel_cases: bool = False,
    workers: int | None = None,
) -> list[dict[str, float | int]]:
    rows = []
    for n_sites in n_values:
        results = run_cases(
            "both",
            n_sites=n_sites,
            j_iso=j_iso,
            d=d,
            bc=bc,
            settings=settings,
            parallel=parallel_cases,
            workers=workers,
        )
        energies = _energy_by_case(results)
        gap = energies["without"] - energies["with"]
        row = {
            "n_sites": n_sites,
            "energy_with_sia": energies["with"],
            "energy_without_sia": energies["without"],
            "gap": gap,
        }
        rows.append(row)
        print(
            f"N={n_sites}: E_with_SIA={energies['with']:.16f}, "
            f"E_without_SIA={energies['without']:.16f}, gap={gap:.16f}"
        )
    return rows


def write_gap_csv(rows: list[dict[str, float | int]], output_path: str | Path) -> None:
    fieldnames = ["n_sites", "energy_with_sia", "energy_without_sia", "gap"]
    with Path(output_path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_gap(rows: list[dict[str, float | int]], output_path: str | Path) -> None:
    if not rows:
        raise ValueError("cannot plot an empty gap scan")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to save the plot. Install it with "
            "`python -m pip install matplotlib`."
        ) from exc

    n_values = [int(row["n_sites"]) for row in rows]
    gaps = [float(row["gap"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(n_values, gaps, marker="o", linewidth=1.8)
    ax.set_xlabel("Number of spin-1 sites, N")
    ax.set_ylabel("Gap: E(D=0) - E(D)")
    ax.set_title("SIA energy gap vs chain length")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sites", default="6,8,10,12,14,16", help="Comma/space separated N values.")
    parser.add_argument("--j", type=float, default=-16.1, help="J in H = -2 J sum_i S_i.S_{i+1}.")
    parser.add_argument("--d", type=float, default=0.252, help="Single-ion anisotropy D.")
    parser.add_argument("--bc", choices=("open", "periodic"), default="periodic", help="Lattice boundary condition.")
    parser.add_argument("--chi-max", type=int, default=100, help="Maximum MPS bond dimension.")
    parser.add_argument("--max-sweeps", type=int, default=20, help="Maximum number of DMRG sweeps.")
    parser.add_argument(
        "--parallel-cases",
        action="store_true",
        help="Run the with-SIA and without-SIA DMRG jobs for each N in separate processes.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Worker processes for --parallel-cases.")
    parser.add_argument("--csv", default="spin1_chain_gap_vs_n.csv", help="Output CSV path.")
    parser.add_argument("--plot", default="spin1_chain_gap_vs_n.png", help="Output PNG plot path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = parse_n_values(args.n_sites)
    settings = DMRGSettings(chi_max=args.chi_max, max_sweeps=args.max_sweeps)
    rows = run_gap_scan(
        n_values=n_values,
        j_iso=args.j,
        d=args.d,
        bc=args.bc,
        settings=settings,
        parallel_cases=args.parallel_cases,
        workers=args.workers,
    )
    write_gap_csv(rows, args.csv)
    plot_gap(rows, args.plot)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
