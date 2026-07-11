"""Scan spin-chain sizes and plot triplet-singlet gaps from TenPy DMRG.

This script compares the two Hamiltonians implemented in spin1_chain_dmrg.py:

    with SIA:    H = -2 J sum_i S_i.S_{i+1}
                    + Dz sum_i (Sz_i)^2
                    + Dpp sum_i [(Sp_i)^2 + (Sm_i)^2]
    without SIA: H = -2 J sum_i S_i . S_{i+1}

For each Hamiltonian, the plotted spin gap is

    gap(N) = E(Sz_total=1, N) - E(Sz_total=0, N)

This gives one curve with the SIA term and one curve with D=0. With a
single-ion anisotropy, this targets the Sz=+1 triplet branch; the Sz=0 triplet
component requires an excited-state DMRG calculation in the Sz_total=0 sector.

Run from the root of a TenPy checkout, for example:

    python examples/spin1_chain_gap_scan.py --n-sites 6,8,10,12,14,16

The default plot output is a matplotlib PNG file.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path

from spin1_chain_dmrg import (
    DMRGSettings,
    Spin1ChainWithSIA,
    Spin1ChainWithoutSIA,
    product_state_for_total_sz,
    total_sz,
)


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


def run_sector_energy(model: object, target_sz: int) -> float:
    product_state = product_state_for_total_sz(model.n_sites, target_sz)
    result = model.run_dmrg(product_state=product_state)
    return float(result["info"]["E"])


def run_gap_scan(
    n_values: list[int],
    j_iso: float,
    dz: float,
    dpp: float,
    bc: str,
    settings: DMRGSettings,
) -> list[dict[str, float | int]]:
    rows = []
    for n_sites in n_values:
        model_with_sia = Spin1ChainWithSIA(
            n_sites=n_sites,
            j_iso=j_iso,
            dz=dz,
            dpp=dpp,
            bc=bc,
            dmrg_settings=settings,
        )
        model_without_sia = Spin1ChainWithoutSIA(
            n_sites=n_sites,
            j_iso=j_iso,
            bc=bc,
            dmrg_settings=settings,
        )
        energy_with_sia_sz0 = run_sector_energy(model_with_sia, 0)
        energy_with_sia_sz1 = run_sector_energy(model_with_sia, 1)
        energy_without_sia_sz0 = run_sector_energy(model_without_sia, 0)
        energy_without_sia_sz1 = run_sector_energy(model_without_sia, 1)
        gap_with_sia = energy_with_sia_sz1 - energy_with_sia_sz0
        gap_without_sia = energy_without_sia_sz1 - energy_without_sia_sz0
        row = {
            "n_sites": n_sites,
            "energy_with_sia_sz0": energy_with_sia_sz0,
            "energy_with_sia_sz1": energy_with_sia_sz1,
            "gap_with_sia": gap_with_sia,
            "energy_without_sia_sz0": energy_without_sia_sz0,
            "energy_without_sia_sz1": energy_without_sia_sz1,
            "gap_without_sia": gap_without_sia,
            "gap_difference": gap_with_sia - gap_without_sia,
        }
        rows.append(row)
        print(
            f"N={n_sites}: gap_with_SIA={gap_with_sia:.16f}, "
            f"gap_D0={gap_without_sia:.16f}, "
            f"delta={gap_with_sia - gap_without_sia:.16f}"
        )
    return rows


def write_gap_csv(rows: list[dict[str, float | int]], output_path: str | Path) -> None:
    fieldnames = [
        "n_sites",
        "energy_with_sia_sz0",
        "energy_with_sia_sz1",
        "gap_with_sia",
        "energy_without_sia_sz0",
        "energy_without_sia_sz1",
        "gap_without_sia",
        "gap_difference",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return 0.5 * (out_min + out_max)
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)


def _plot_gap_svg(rows: list[dict[str, float | int]], output_path: Path) -> None:
    width = 760
    height = 480
    left = 82
    right = 24
    top = 54
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    n_values = [int(row["n_sites"]) for row in rows]
    series = [
        ("gap_with_sia", "with SIA", "#0f766e"),
        ("gap_without_sia", "D=0", "#b91c1c"),
    ]
    all_gaps = [float(row[key]) for key, _, _ in series for row in rows]
    x_min = min(n_values)
    x_max = max(n_values)
    y_min = min(0.0, min(all_gaps))
    y_max = max(0.0, max(all_gaps))
    if y_min == y_max:
        padding = abs(y_min) * 0.1 or 1.0
        y_min -= padding
        y_max += padding
    else:
        padding = 0.08 * (y_max - y_min)
        y_min -= padding
        y_max += padding

    points_by_key = {}
    for key, _, _ in series:
        points = []
        for row in rows:
            x = _scale(int(row["n_sites"]), x_min, x_max, left, left + plot_width)
            y = _scale(float(row[key]), y_min, y_max, top + plot_height, top)
            points.append((x, y))
        points_by_key[key] = points

    y_ticks = [y_min + index * (y_max - y_min) / 4.0 for index in range(5)]
    curve_markup = []
    for key, _, color in series:
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points_by_key[key])
        circles = "\n".join(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" />'
            for x, y in points_by_key[key]
        )
        curve_markup.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5" />\n{circles}'
        )
    x_labels = "\n".join(
        (
            f'<text x="{x:.2f}" y="{height - 34}" text-anchor="middle" '
            f'font-size="13">{n_sites}</text>'
        )
        for (x, _), n_sites in zip(points_by_key["gap_with_sia"], n_values)
    )
    legend = "\n".join(
        (
            f'<line x1="{left + index * 120}" x2="{left + index * 120 + 28}" y1="48" y2="48" '
            f'stroke="{color}" stroke-width="2.5" />'
            f'<text x="{left + index * 120 + 36}" y="52" font-size="13">{label}</text>'
        )
        for index, (_, label, color) in enumerate(series)
    )
    y_labels = []
    grid_lines = []
    for tick in y_ticks:
        y = _scale(tick, y_min, y_max, top + plot_height, top)
        y_labels.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{tick:.6g}</text>'
        )
        grid_lines.append(
            f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.2f}" y2="{y:.2f}" '
            'stroke="#d1d5db" stroke-width="1" />'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="30" text-anchor="middle" font-size="20" font-family="Arial, sans-serif">{html.escape("Triplet-singlet gap vs chain length")}</text>
  <g font-family="Arial, sans-serif" fill="#111827">
    {legend}
    {"".join(grid_lines)}
    <line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5" />
    <line x1="{left}" x2="{left + plot_width}" y1="{top + plot_height}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5" />
    {"".join(curve_markup)}
    {"".join(y_labels)}
    {x_labels}
    <text x="{left + plot_width / 2}" y="{height - 10}" text-anchor="middle" font-size="15">Number of spin-1 sites, N</text>
    <text transform="translate(22 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-size="15">Gap: E(Sz=1) - E(Sz=0)</text>
  </g>
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")


def _plot_gap_with_matplotlib(rows: list[dict[str, float | int]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for PNG plots. Install it with "
            "`python -m pip install matplotlib`."
        ) from exc

    n_values = [int(row["n_sites"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(n_values, [float(row["gap_with_sia"]) for row in rows], marker="o", linewidth=1.8, label="with SIA")
    ax.plot(n_values, [float(row["gap_without_sia"]) for row in rows], marker="s", linewidth=1.8, label="D=0")
    ax.set_xlabel("Number of spin-1 sites, N")
    ax.set_ylabel("Gap: E(Sz=1) - E(Sz=0)")
    ax.set_title("Triplet-singlet gap vs chain length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gap(rows: list[dict[str, float | int]], output_path: str | Path) -> None:
    if not rows:
        raise ValueError("cannot plot an empty gap scan")
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".svg":
        _plot_gap_svg(rows, output_path)
        return
    _plot_gap_with_matplotlib(rows, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sites", default="6,8,10,12,14,16", help="Comma/space separated N values.")
    parser.add_argument("--j", type=float, default=-16.1, help="J in H = -2 J sum_i S_i.S_{i+1}.")
    parser.add_argument("--dz", type=float, default=0.379, help="Coefficient Dz for sum_i (Sz_i)^2.")
    parser.add_argument(
        "--dpp",
        type=float,
        default=-0.017,
        help="Coefficient Dpp for sum_i [(Sp_i)^2 + (Sm_i)^2].",
    )
    parser.add_argument("--bc", choices=("open", "periodic"), default="open", help="Lattice boundary condition.")
    parser.add_argument("--chi-max", type=int, default=100, help="Maximum MPS bond dimension.")
    parser.add_argument("--max-sweeps", type=int, default=20, help="Maximum number of DMRG sweeps.")
    parser.add_argument("--csv", default="spin1_chain_gap_vs_n.csv", help="Output CSV path.")
    parser.add_argument("--plot", default="spin1_chain_gap_vs_n.png", help="Output PNG plot path. Requires matplotlib.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = parse_n_values(args.n_sites)
    settings = DMRGSettings(chi_max=args.chi_max, max_sweeps=args.max_sweeps)
    rows = run_gap_scan(
        n_values=n_values,
        j_iso=args.j,
        dz=args.dz,
        dpp=args.dpp,
        bc=args.bc,
        settings=settings,
    )
    write_gap_csv(rows, args.csv)
    plot_gap(rows, args.plot)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
