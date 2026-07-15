# Finite-Temperature Susceptibility SI Output Design

## Goal

Update `examples/spin1_chain_finite_temperature_susceptibility.py` so that a
single run calculates and plots the finite-temperature susceptibility using
explicit physical units:

- Hamiltonian parameters `J`, `Dz`, and `Dpp` are interpreted in meV.
- Temperature is requested and reported in K.
- Susceptibility is reported as SI molar susceptibility in `m^3/mol`.
- The default finite-temperature range is 2 K through 500 K.
- The script writes both a CSV data file and a headless PNG plot.

The existing purification calculation and spin-1 Hamiltonian definitions stay
unchanged. Unit conversion is applied at the input/output boundary.

## Physical Conventions

The internal TeNPy Hamiltonian uses meV, so inverse temperature has units of
`meV^-1`. A requested temperature is converted to an internal beta using

```text
beta_meV_inv = 1 / (k_B_meV_per_K * temperature_K).
```

Use named module constants containing CODATA values:

```text
k_B                 = 0.08617333262 meV/K
1 meV               = 1.602176634e-22 J
mu_0                = 1.25663706127e-6 N/A^2
N_A                 = 6.02214076e23 mol^-1
mu_B                = 9.2740100657e-24 J/T
```

For dimensionless spin operators, retain the reduced per-site fluctuation
susceptibility for diagnostics:

```text
chi_reduced_meV_inv = beta_meV_inv / N
                      * (<Mz^2> - <Mz>^2).
```

Convert that value to SI molar susceptibility with

```text
chi_molar_m3_per_mol = mu_0 * N_A * (g * mu_B)^2
                       / MEV_TO_J
                       * chi_reduced_meV_inv.
```

The Lande factor is supplied by `--g-factor`, defaults to 2.0, and must be
positive. This is the z-axis factor because the calculation uses total `Sz`.

## Temperature Grid and Evolution

The user-facing grid is independent of the imaginary-time integration step:

- `--temperature-min-k`: default 2.0 K.
- `--temperature-max-k`: default 500.0 K.
- `--temperature-points`: default 100.
- `--dt`: default 0.01 `meV^-1`; this is the maximum half-step used by the
  purification evolution, not a temperature-axis bound.

Generate `temperature_points` logarithmically spaced target temperatures from
the maximum down to the minimum. Convert these targets to increasing beta and
cool the same purification state sequentially. For each target beta:

1. Find the remaining beta increment from the current state.
2. Split it into increments no larger than `2 * dt`.
3. For each increment, set `segment_dt = increment / 2`, build the two complex
   MPO propagators, and apply them with `PurificationApplyMPO`.
4. Use a final shorter segment when needed so the measurement is taken at the
   requested temperature rather than at an interpolated value.

Only the requested finite-temperature points are emitted. Rows are returned in
ascending temperature order for convenient plotting even though the state is
evolved from high to low temperature.

Validate that both temperature bounds are positive, the maximum is greater
than the minimum, `temperature_points` is at least two, and `dt` is positive.

For compatibility with the previously documented server command,
`--beta-max` remains accepted as a deprecated alternative for specifying the
low-temperature endpoint. If both `--beta-max` and `--temperature-min-k` are
given explicitly, fail with a clear argument error instead of choosing one
silently.

## Data Output

The CSV uses unit-bearing column names:

```text
case
beta_meV_inv
temperature_K
chi_reduced_meV_inv
chi_molar_m3_per_mol
g_factor
mz
mz2
n_sites
```

The reduced value remains available for diagnostics, while
`chi_molar_m3_per_mol` is the primary physical result. Both SIA cases are
grouped by `case`, with ascending temperatures inside each group.

## Plot Output

Add `--plot`, defaulting to
`spin1_chain_susceptibility_vs_temperature.png`. After writing the CSV, the
script automatically writes the PNG.

The plotting helper will:

- import Matplotlib lazily and select the non-interactive `Agg` backend;
- draw one curve for each requested Hamiltonian case;
- sort each curve by `temperature_K`;
- label the axes `Temperature (K)` and
  `Molar susceptibility (m^3 mol^-1)`;
- use scientific notation for the susceptibility axis;
- save at 300 dpi with a tight layout; and
- fail with a concise installation instruction if Matplotlib is unavailable.

The plot contains calculated points only. It does not extrapolate or
interpolate beyond the requested 2-500 K default range.

## Code Boundaries

Keep the change scoped to the finite-temperature example and its test module.
Introduce small pure helpers for:

- beta/temperature conversion;
- the logarithmic target grid;
- reduced-to-molar susceptibility conversion; and
- grouping and plotting rows.

Keep TeNPy-dependent evolution separate from these pure conversion helpers so
unit tests can run without constructing a real tensor-network state.

## Error Handling

Reject non-finite or invalid physical inputs before starting an expensive
purification run. Error messages must identify the offending argument and its
required range. A missing Matplotlib installation should mention
`python -m pip install matplotlib`. CSV and PNG filesystem errors should retain
their native path context.

## Tests

Extend `tests/test_spin1_chain_finite_temperature_susceptibility_example.py`
with focused tests that verify:

- meV inverse temperature converts to the expected kelvin value;
- the target grid has exactly the requested endpoints, count, and ordering;
- segmented evolution reaches each target beta without exceeding `dt`;
- molar susceptibility follows the SI conversion and scales as `g^2`;
- invalid temperature bounds, point counts, `dt`, and `g` are rejected;
- CSV headers and values use the explicit unit-bearing names;
- the headless plotting path groups cases, sorts temperatures, labels axes,
  and saves the requested file;
- CLI defaults are 2 K, 500 K, 100 points, `dt=0.01`, and `g=2.0`; and
- the deprecated `--beta-max` path works while conflicting bounds fail.

Use fakes for TeNPy purification classes and the plotting boundary where
appropriate. Run the focused example tests and the complete example-test suite
before publishing the implementation.

## Acceptance Criteria

A default run produces a CSV and PNG containing 100 calculated temperatures
from 2 K through 500 K. The plotted y-axis and primary CSV susceptibility are
SI molar susceptibility in `m^3/mol`, calculated with the requested g-factor.
Changing the temperature range does not require manually deriving `dt`, and
changing `dt` changes the maximum integration step, numerical accuracy, and
cost without changing the requested temperature grid.
