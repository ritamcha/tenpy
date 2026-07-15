# Finite-Temperature Susceptibility SI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce finite-temperature spin-1 susceptibility data from 2 K through 1200 K and save SI molar susceptibility to both CSV and PNG outputs.

**Architecture:** Keep TeNPy's Hamiltonian and purification evolution in meV, then convert target temperatures to `meV^-1` beta values and reduced susceptibility to `m^3/mol` at explicit helper boundaries. Drive one purification state through a logarithmic finite-temperature target grid with bounded imaginary-time substeps, then feed unit-bearing rows to isolated CSV and headless Matplotlib writers.

**Tech Stack:** Python 3, NumPy, TeNPy `PurificationMPS`/`PurificationApplyMPO`, optional Matplotlib `Agg`, `unittest`, `unittest.mock`.

## Global Constraints

- Interpret `J`, `Dz`, and `Dpp` as meV without rescaling the TeNPy Hamiltonian.
- Default to 2 K through 1200 K with 150 logarithmically spaced calculated points.
- Report primary susceptibility as SI molar susceptibility in `m^3/mol` and retain reduced susceptibility in `meV^-1`.
- Use a positive z-axis Lande factor from `--g-factor`, default 2.0.
- Treat `--dt`, default `0.01 meV^-1`, only as the maximum purification half-step.
- Save CSV and a 300 dpi PNG automatically; use Matplotlib's non-interactive `Agg` backend.
- Preserve `--beta-max` as a deprecated low-temperature-bound alternative and reject it when `--temperature-min-k` is also supplied.
- Do not modify or commit the pre-existing worktree changes in `examples/spin1_chain_dmrg.py` or `examples/spin1_chain_gap_scan.py`.

---

## File Map

- Modify `examples/spin1_chain_finite_temperature_susceptibility.py`: physical constants, conversions, temperature grid, target-driven purification, unit-bearing rows, CSV/PNG writers, CLI integration.
- Modify `tests/test_spin1_chain_finite_temperature_susceptibility_example.py`: pure conversion, evolution, output, CLI, and regression coverage using fakes at TeNPy and Matplotlib boundaries.
- Keep `examples/spin1_chain_dmrg.py` unchanged: existing Hamiltonian/model construction remains the source for both SIA cases.

### Task 1: Physical Units and Temperature Grid

**Files:**
- Modify: `examples/spin1_chain_finite_temperature_susceptibility.py:20-76`
- Test: `tests/test_spin1_chain_finite_temperature_susceptibility_example.py`

**Interfaces:**
- Consumes: total magnetization moments `mz`, `mz2`, site count, input beta in `meV^-1`, and positive g-factor.
- Produces: `beta_from_temperature_kelvin(temperature_k) -> float`, `temperature_kelvin_from_beta(beta_mev_inv) -> float`, `temperature_grid_kelvin(minimum_k, maximum_k, points) -> list[float]`, `reduced_susceptibility_from_moments(beta_mev_inv, n_sites, mz, mz2) -> float`, and `molar_susceptibility_from_reduced(chi_reduced_mev_inv, g_factor) -> float`.

- [ ] **Step 1: Write failing conversion and validation tests**

Add these methods to `Spin1ChainFiniteTemperatureSusceptibilityTest` and replace the two old generic susceptibility tests with the reduced-susceptibility names:

```python
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
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m unittest tests.test_spin1_chain_finite_temperature_susceptibility_example -v
```

Expected: FAIL with missing attributes such as `beta_from_temperature_kelvin` and `molar_susceptibility_from_reduced`.

- [ ] **Step 3: Add constants and pure conversion helpers**

Replace the generic CSV constant and susceptibility helper area with these named constants and functions while retaining the existing scalar validators:

```python
K_B_MEV_PER_K = 0.08617333262
MEV_TO_JOULE = 1.602176634e-22
VACUUM_PERMEABILITY = 1.25663706127e-6
AVOGADRO_CONSTANT = 6.02214076e23
BOHR_MAGNETON_J_PER_T = 9.2740100657e-24


def beta_from_temperature_kelvin(temperature_k: float) -> float:
    temperature_k = _validate_positive_real(temperature_k, "temperature_k")
    return 1.0 / (K_B_MEV_PER_K * temperature_k)


def temperature_kelvin_from_beta(beta_mev_inv: float) -> float:
    beta_mev_inv = _validate_positive_real(beta_mev_inv, "beta_mev_inv")
    return 1.0 / (K_B_MEV_PER_K * beta_mev_inv)


def temperature_grid_kelvin(minimum_k: float, maximum_k: float, points: int) -> list[float]:
    minimum_k = _validate_positive_real(minimum_k, "minimum_k")
    maximum_k = _validate_positive_real(maximum_k, "maximum_k")
    points = _validate_positive_integer(points, "points")
    if maximum_k <= minimum_k:
        raise ValueError("maximum_k must be greater than minimum_k")
    if points < 2:
        raise ValueError("points must be at least 2")
    return [float(value) for value in np.geomspace(maximum_k, minimum_k, points)]


def reduced_susceptibility_from_moments(
    beta_mev_inv: float,
    n_sites: int,
    mz: float,
    mz2: float,
) -> float:
    beta_mev_inv = _validate_positive_real(beta_mev_inv, "beta_mev_inv")
    n_sites = _validate_positive_integer(n_sites, "n_sites")
    mz = _validate_finite_real(mz, "mz")
    mz2 = _validate_finite_real(mz2, "mz2")
    return beta_mev_inv * (mz2 - mz * mz) / n_sites


def molar_susceptibility_from_reduced(chi_reduced_mev_inv: float, g_factor: float) -> float:
    chi_reduced_mev_inv = _validate_finite_real(chi_reduced_mev_inv, "chi_reduced_mev_inv")
    g_factor = _validate_positive_real(g_factor, "g_factor")
    prefactor = (
        VACUUM_PERMEABILITY
        * AVOGADRO_CONSTANT
        * (g_factor * BOHR_MAGNETON_J_PER_T) ** 2
        / MEV_TO_JOULE
    )
    return prefactor * chi_reduced_mev_inv
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: all conversion/grid tests PASS; evolution and CSV tests may still fail only where they assert the old schema.

- [ ] **Step 5: Commit Task 1**

```bash
git add examples/spin1_chain_finite_temperature_susceptibility.py tests/test_spin1_chain_finite_temperature_susceptibility_example.py
git commit -m "add physical susceptibility unit conversions"
```

### Task 2: Target-Driven Purification Evolution

**Files:**
- Modify: `examples/spin1_chain_finite_temperature_susceptibility.py:99-194`
- Test: `tests/test_spin1_chain_finite_temperature_susceptibility_example.py`

**Interfaces:**
- Consumes: temperature bounds and count in K, maximum `dt` in `meV^-1`, model object, g-factor, and truncation controls.
- Produces: `run_case_scan(case_label, model, temperature_min_k, temperature_max_k, temperature_points, dt, chi_max, svd_min, g_factor) -> list[dict[str, float | int | str]]` and matching multi-case `run_susceptibility_scan` rows in ascending temperature order.

- [ ] **Step 1: Replace the fixed-step evolution test with a failing target-beta test**

Use this test to prove exact endpoint beta values, a shortened final segment, bounded `dt`, and ascending returned temperatures:

```python
def test_run_case_scan_reaches_temperature_targets_with_bounded_mpo_steps(self):
    class FakeHMPO:
        calls = []

        def make_U(self, dt, approximation):
            self.calls.append((dt, approximation))
            return f"U({dt},{approximation})"

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
```

Replace the existing multi-case test with this complete method:

```python
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
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run the Task 1 test command. Expected: FAIL because `run_case_scan` does not accept temperature bounds or emit unit-bearing rows.

- [ ] **Step 3: Implement segmented target evolution and unit-bearing measurements**

Use these row and propagator helpers:

```python
def _measurement_row(
    case_label: str,
    beta_mev_inv: float,
    temperature_k: float,
    n_sites: int,
    psi,
    g_factor: float,
) -> dict[str, float | int | str]:
    mz, mz2 = magnetization_moments(psi)
    chi_reduced = reduced_susceptibility_from_moments(beta_mev_inv, n_sites, mz, mz2)
    return {
        "case": case_label,
        "beta_meV_inv": beta_mev_inv,
        "temperature_K": temperature_k,
        "chi_reduced_meV_inv": chi_reduced,
        "chi_molar_m3_per_mol": molar_susceptibility_from_reduced(chi_reduced, g_factor),
        "g_factor": g_factor,
        "mz": mz,
        "mz2": mz2,
        "n_sites": n_sites,
    }


def _mpo_propagators(tenpy_model, segment_dt: float) -> list[object]:
    return [
        tenpy_model.H_MPO.make_U(-step * segment_dt, "II")
        for step in (0.5 + 0.5j, 0.5 - 0.5j)
    ]
```

Replace `run_case_scan` with target-driven evolution:

```python
def run_case_scan(
    case_label: str,
    model,
    temperature_min_k: float,
    temperature_max_k: float,
    temperature_points: int,
    dt: float,
    chi_max: int,
    svd_min: float,
    g_factor: float,
) -> list[dict[str, float | int | str]]:
    _validate_choice(case_label, "case_label", ("with", "without"))
    temperatures = temperature_grid_kelvin(temperature_min_k, temperature_max_k, temperature_points)
    dt = _validate_positive_real(dt, "dt")
    chi_max = _validate_positive_integer(chi_max, "chi_max")
    svd_min = _validate_non_negative_real(svd_min, "svd_min")
    g_factor = _validate_positive_real(g_factor, "g_factor")

    PurificationApplyMPO, PurificationMPS = _require_tenpy_purification()
    tenpy_model = model.build_model()
    purification_kwargs = {"bc": tenpy_model.lat.bc_MPS}
    unit_cell_width = getattr(tenpy_model.lat, "mps_unit_cell_width", None)
    if unit_cell_width is not None:
        purification_kwargs["unit_cell_width"] = unit_cell_width
    psi = PurificationMPS.from_infiniteT(tenpy_model.lat.mps_sites(), **purification_kwargs)
    options = {"trunc_params": {"chi_max": chi_max, "svd_min": svd_min}}

    rows = []
    engine = None
    current_beta = 0.0
    for temperature_k in temperatures:
        target_beta = beta_from_temperature_kelvin(temperature_k)
        tolerance = 1.0e-12 * max(1.0, target_beta)
        while target_beta - current_beta > tolerance:
            beta_increment = min(2.0 * dt, target_beta - current_beta)
            segment_dt = 0.5 * beta_increment
            propagators = _mpo_propagators(tenpy_model, segment_dt)
            if engine is None:
                engine = PurificationApplyMPO(psi, propagators[0], options)
            for propagator in propagators:
                engine.init_env(propagator)
                engine.run()
            current_beta += beta_increment
        current_beta = target_beta
        rows.append(_measurement_row(
            case_label,
            target_beta,
            temperature_k,
            model.n_sites,
            psi,
            g_factor,
        ))

    rows.reverse()
    return rows
```

Replace `run_susceptibility_scan` with the complete multi-case boundary:

```python
def run_susceptibility_scan(
    case: str,
    n_sites: int,
    j_iso: float,
    dz: float,
    dpp: float,
    bc: str,
    temperature_min_k: float,
    temperature_max_k: float,
    temperature_points: int,
    dt: float,
    chi_max: int,
    svd_min: float,
    g_factor: float,
) -> list[dict[str, float | int | str]]:
    n_sites = _validate_positive_integer(n_sites, "n_sites")
    j_iso = _validate_finite_real(j_iso, "j_iso")
    dz = _validate_finite_real(dz, "dz")
    dpp = _validate_finite_real(dpp, "dpp")
    bc = _validate_choice(bc, "bc", ("open", "periodic"))
    temperature_grid_kelvin(temperature_min_k, temperature_max_k, temperature_points)
    dt = _validate_positive_real(dt, "dt")
    chi_max = _validate_positive_integer(chi_max, "chi_max")
    svd_min = _validate_non_negative_real(svd_min, "svd_min")
    g_factor = _validate_positive_real(g_factor, "g_factor")

    rows = []
    for selected_case in _selected_cases(case):
        model = _build_case_model(selected_case, n_sites, j_iso, dz, dpp, bc)
        rows.extend(run_case_scan(
            selected_case,
            model,
            temperature_min_k,
            temperature_max_k,
            temperature_points,
            dt,
            chi_max,
            svd_min,
            g_factor,
        ))
    return rows
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Task 1 test command. Expected: target evolution and multi-case tests PASS with exactly six fake MPO applications in the segmented test.

- [ ] **Step 5: Commit Task 2**

```bash
git add examples/spin1_chain_finite_temperature_susceptibility.py tests/test_spin1_chain_finite_temperature_susceptibility_example.py
git commit -m "drive purification over a kelvin temperature grid"
```

### Task 3: Unit-Bearing CSV and Headless PNG

**Files:**
- Modify: `examples/spin1_chain_finite_temperature_susceptibility.py:20-98`
- Test: `tests/test_spin1_chain_finite_temperature_susceptibility_example.py`

**Interfaces:**
- Consumes: rows from Task 2 and output filesystem paths.
- Produces: `write_susceptibility_csv(rows, output_path) -> None`, `_require_matplotlib_pyplot()`, and `write_susceptibility_plot(rows, output_path) -> None`.

- [ ] **Step 1: Write failing CSV schema and plotting-boundary tests**

Replace the existing CSV test row and assertions with:

```python
def test_rows_are_written_with_unit_bearing_csv_columns(self):
    rows = [{
        "case": "with",
        "beta_meV_inv": 0.5,
        "temperature_K": 23.209036243100163,
        "chi_reduced_meV_inv": 1.25,
        "chi_molar_m3_per_mol": 2.0e-6,
        "g_factor": 2.0,
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

    self.assertIn(
        "case,beta_meV_inv,temperature_K,chi_reduced_meV_inv,"
        "chi_molar_m3_per_mol,g_factor,mz,mz2,n_sites",
        content,
    )
    self.assertIn("with,0.5,23.209036243100163,1.25,2e-06,2.0,0.0,10.0,8", content)
```

Add a fake Matplotlib boundary test:

```python
def test_plot_groups_cases_sorts_temperature_and_saves_png(self):
    class FakeAxes:
        def __init__(self):
            self.plot_calls = []
            self.xlabel = None
            self.ylabel = None

        def plot(self, x, y, **kwargs):
            self.plot_calls.append((list(x), list(y), kwargs))

        def set_xlabel(self, label):
            self.xlabel = label

        def set_ylabel(self, label):
            self.ylabel = label

        def ticklabel_format(self, **kwargs):
            self.tick_format = kwargs

        def legend(self):
            self.legend_called = True

        def grid(self, *args, **kwargs):
            self.grid_call = (args, kwargs)

    class FakeFigure:
        def __init__(self):
            self.saved = None

        def tight_layout(self):
            self.tight_layout_called = True

        def savefig(self, output_path, **kwargs):
            self.saved = (Path(output_path), kwargs)

    class FakePyplot:
        def __init__(self):
            self.figure = FakeFigure()
            self.axes = FakeAxes()

        def subplots(self):
            return self.figure, self.axes

        def close(self, figure):
            self.closed = figure

    rows = [
        {"case": "with", "temperature_K": 1200.0, "chi_molar_m3_per_mol": 1.0e-7},
        {"case": "with", "temperature_K": 2.0, "chi_molar_m3_per_mol": 4.0e-6},
        {"case": "without", "temperature_K": 1200.0, "chi_molar_m3_per_mol": 1.2e-7},
        {"case": "without", "temperature_K": 2.0, "chi_molar_m3_per_mol": 3.8e-6},
    ]
    pyplot = FakePyplot()
    output = Path("susceptibility.png")

    with patch.object(spin1_chain_ft, "_require_matplotlib_pyplot", return_value=pyplot):
        spin1_chain_ft.write_susceptibility_plot(rows, output)

    self.assertEqual(pyplot.axes.plot_calls[0][0], [2.0, 1200.0])
    self.assertEqual(pyplot.axes.plot_calls[1][0], [2.0, 1200.0])
    self.assertEqual(pyplot.axes.xlabel, "Temperature (K)")
    self.assertEqual(pyplot.axes.ylabel, "Molar susceptibility (m^3 mol^-1)")
    self.assertEqual(pyplot.figure.saved, (output, {"dpi": 300, "bbox_inches": "tight"}))
    self.assertIs(pyplot.closed, pyplot.figure)

def test_missing_matplotlib_error_includes_install_command(self):
    with patch.dict(sys.modules, {"matplotlib": None}):
        with self.assertRaisesRegex(RuntimeError, "python -m pip install matplotlib"):
            spin1_chain_ft._require_matplotlib_pyplot()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run the Task 1 test command. Expected: CSV header assertion FAIL and plot test FAIL because `write_susceptibility_plot` is missing.

- [ ] **Step 3: Implement explicit CSV fields and lazy headless plotting**

Set the schema and case labels:

```python
CSV_FIELDNAMES = [
    "case",
    "beta_meV_inv",
    "temperature_K",
    "chi_reduced_meV_inv",
    "chi_molar_m3_per_mol",
    "g_factor",
    "mz",
    "mz2",
    "n_sites",
]
CASE_PLOT_LABELS = {"with": "with SIA", "without": "without SIA"}
```

Add the lazy dependency and plot writer:

```python
def _require_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Matplotlib is required to write the susceptibility plot. "
            "Install it with `python -m pip install matplotlib`."
        ) from exc
    return plt


def write_susceptibility_plot(
    rows: list[dict[str, float | int | str]],
    output_path: str | Path,
) -> None:
    plt = _require_matplotlib_pyplot()
    figure, axes = plt.subplots()
    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    for case in cases:
        points = sorted(
            (
                (float(row["temperature_K"]), float(row["chi_molar_m3_per_mol"]))
                for row in rows
                if row["case"] == case and np.isfinite(float(row["temperature_K"]))
            ),
            key=lambda point: point[0],
        )
        if not points:
            continue
        axes.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            markersize=3,
            label=CASE_PLOT_LABELS.get(case, case),
        )
    axes.set_xlabel("Temperature (K)")
    axes.set_ylabel("Molar susceptibility (m^3 mol^-1)")
    axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes.grid(True, alpha=0.25)
    axes.legend()
    figure.tight_layout()
    figure.savefig(Path(output_path), dpi=300, bbox_inches="tight")
    plt.close(figure)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Task 1 test command. Expected: CSV and fake plotting tests PASS without requiring Matplotlib to be installed in the test interpreter.

- [ ] **Step 5: Commit Task 3**

```bash
git add examples/spin1_chain_finite_temperature_susceptibility.py tests/test_spin1_chain_finite_temperature_susceptibility_example.py
git commit -m "write SI susceptibility CSV and plot outputs"
```

### Task 4: CLI Integration and Legacy Bound Compatibility

**Files:**
- Modify: `examples/spin1_chain_finite_temperature_susceptibility.py:196-240`
- Test: `tests/test_spin1_chain_finite_temperature_susceptibility_example.py`

**Interfaces:**
- Consumes: optional CLI argument list and Task 1-3 helpers.
- Produces: `parse_args(argv: list[str] | None = None) -> argparse.Namespace` and `main(argv: list[str] | None = None) -> None` that write both configured outputs.

- [ ] **Step 1: Write failing CLI default, legacy, conflict, and main tests**

Replace the existing default test and add compatibility coverage:

```python
def test_default_cli_values_use_kelvin_si_grid_and_plot(self):
    args = spin1_chain_ft.parse_args([])

    self.assertEqual(args.case, "with")
    self.assertEqual(args.temperature_min_k, 2.0)
    self.assertEqual(args.temperature_max_k, 1200.0)
    self.assertEqual(args.temperature_points, 150)
    self.assertEqual(args.dt, 0.01)
    self.assertEqual(args.g_factor, 2.0)
    self.assertEqual(args.csv, "spin1_chain_susceptibility_vs_temperature.csv")
    self.assertEqual(args.plot, "spin1_chain_susceptibility_vs_temperature.png")

def test_legacy_beta_max_sets_low_temperature_endpoint(self):
    args = spin1_chain_ft.parse_args(["--beta-max", "5.0"])

    self.assertAlmostEqual(
        args.temperature_min_k,
        spin1_chain_ft.temperature_kelvin_from_beta(5.0),
        places=12,
    )

def test_cli_rejects_conflicting_low_temperature_bounds(self):
    with self.assertRaises(SystemExit):
        spin1_chain_ft.parse_args([
            "--temperature-min-k",
            "2.0",
            "--beta-max",
            "5.0",
        ])

def test_main_writes_csv_and_png(self):
    rows = [{
        "case": "with",
        "beta_meV_inv": 1.0,
        "temperature_K": 11.604518121550082,
        "chi_reduced_meV_inv": 1.0,
        "chi_molar_m3_per_mol": 1.0e-6,
        "g_factor": 2.0,
        "mz": 0.0,
        "mz2": 12.0,
        "n_sites": 12,
    }]

    with patch.object(spin1_chain_ft, "run_susceptibility_scan", return_value=rows) as run_scan, patch.object(
        spin1_chain_ft,
        "write_susceptibility_csv",
    ) as write_csv, patch.object(spin1_chain_ft, "write_susceptibility_plot") as write_plot:
        spin1_chain_ft.main([
            "--temperature-min-k",
            "2",
            "--temperature-max-k",
            "1200",
            "--csv",
            "result.csv",
            "--plot",
            "result.png",
        ])

    run_scan.assert_called_once_with(
        case="with",
        n_sites=12,
        j_iso=-16.1,
        dz=0.379,
        dpp=-0.017,
        bc="open",
        temperature_min_k=2.0,
        temperature_max_k=1200.0,
        temperature_points=150,
        dt=0.01,
        chi_max=100,
        svd_min=1.0e-8,
        g_factor=2.0,
    )
    write_csv.assert_called_once_with(rows, "result.csv")
    write_plot.assert_called_once_with(rows, "result.png")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run the Task 1 test command. Expected: `parse_args([])` raises a signature error or lacks the new properties, and `main([...])` lacks the argument-list interface.

- [ ] **Step 3: Add temperature/SI/output CLI arguments and resolve bounds**

Change the parser signature to `def parse_args(argv: list[str] | None = None) -> argparse.Namespace:`. Replace the old finite-temperature arguments with:

```python
parser.add_argument("--temperature-min-k", type=float, default=None, help="Minimum temperature in K.")
parser.add_argument("--temperature-max-k", type=float, default=1200.0, help="Maximum temperature in K.")
parser.add_argument("--temperature-points", type=int, default=150, help="Number of logarithmic temperatures.")
parser.add_argument(
    "--beta-max",
    type=float,
    default=None,
    help="Deprecated alternative to --temperature-min-k, in meV^-1.",
)
parser.add_argument(
    "--dt",
    type=float,
    default=0.01,
    help="Maximum purification half-step in meV^-1.",
)
parser.add_argument("--g-factor", type=float, default=2.0, help="Lande g-factor along z.")
parser.add_argument(
    "--csv",
    default="spin1_chain_susceptibility_vs_temperature.csv",
    help="Output CSV path.",
)
parser.add_argument(
    "--plot",
    default="spin1_chain_susceptibility_vs_temperature.png",
    help="Output PNG path.",
)
args = parser.parse_args(argv)
if args.temperature_min_k is not None and args.beta_max is not None:
    parser.error("--temperature-min-k cannot be combined with --beta-max")
if args.temperature_min_k is None:
    if args.beta_max is None:
        args.temperature_min_k = 2.0
    else:
        try:
            args.temperature_min_k = temperature_kelvin_from_beta(args.beta_max)
        except ValueError as exc:
            parser.error(str(exc))
return args
```

Change `main` to accept `argv`, pass every physical/grid argument to `run_susceptibility_scan`, write both outputs, and print both paths:

```python
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = run_susceptibility_scan(
        case=args.case,
        n_sites=args.n_sites,
        j_iso=args.j,
        dz=args.dz,
        dpp=args.dpp,
        bc=args.bc,
        temperature_min_k=args.temperature_min_k,
        temperature_max_k=args.temperature_max_k,
        temperature_points=args.temperature_points,
        dt=args.dt,
        chi_max=args.chi_max,
        svd_min=args.svd_min,
        g_factor=args.g_factor,
    )
    write_susceptibility_csv(rows, args.csv)
    write_susceptibility_plot(rows, args.plot)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.plot}")
```

- [ ] **Step 4: Run focused and complete example tests**

Run:

```bash
python -m unittest tests.test_spin1_chain_finite_temperature_susceptibility_example -v
python -m unittest tests.test_spin1_chain_dmrg_example tests.test_spin1_chain_gap_scan_example tests.test_spin1_chain_finite_temperature_susceptibility_example -v
python examples/spin1_chain_finite_temperature_susceptibility.py --help
```

Expected: all unit tests PASS, and `--help` lists K bounds, 150 points, `dt`, g-factor, CSV, PNG, and deprecated beta bound without importing TeNPy purification.

- [ ] **Step 5: Commit Task 4**

```bash
git add examples/spin1_chain_finite_temperature_susceptibility.py tests/test_spin1_chain_finite_temperature_susceptibility_example.py
git commit -m "integrate kelvin susceptibility CLI outputs"
```

### Task 5: Final Verification and Server Handoff

**Files:**
- Verify: `examples/spin1_chain_finite_temperature_susceptibility.py`
- Verify: `tests/test_spin1_chain_finite_temperature_susceptibility_example.py`

**Interfaces:**
- Consumes: the completed example and tests.
- Produces: verified current-branch commits and a server command that installs plotting support and writes both outputs.

- [ ] **Step 1: Run final whitespace and scope checks**

```bash
git diff --check origin/main...HEAD
git status --short
git diff --stat origin/main...HEAD
```

Expected: no whitespace errors; intended commits contain the specification, plan, finite-temperature script, and its test. The two known unstaged files `examples/spin1_chain_dmrg.py` and `examples/spin1_chain_gap_scan.py` remain unstaged and uncommitted.

- [ ] **Step 2: Run the complete example regression suite again**

```bash
python -m unittest tests.test_spin1_chain_dmrg_example tests.test_spin1_chain_gap_scan_example tests.test_spin1_chain_finite_temperature_susceptibility_example -v
```

Expected: all tests PASS with no traceback or warning from the changed script.

- [ ] **Step 3: Verify the documented fresh-server workflow**

Use this exact handoff command after the branch is published:

```bash
python3 -m pip install -e ".[plot]"
python3 examples/spin1_chain_finite_temperature_susceptibility.py \
  --n-sites 16 \
  --case both \
  --temperature-min-k 2 \
  --temperature-max-k 1200 \
  --temperature-points 150 \
  --dt 0.01 \
  --g-factor 2.0 \
  --csv susceptibility.csv \
  --plot susceptibility_vs_temperature.png
```

Expected artifacts: `susceptibility.csv` with unit-bearing columns and `susceptibility_vs_temperature.png` with K and `m^3 mol^-1` axes.

- [ ] **Step 4: Publish only the current branch**

```bash
git push origin main
```

Expected: `origin/main` advances to the verified implementation commit; no additional branch is created.
