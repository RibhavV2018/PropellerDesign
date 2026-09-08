# `uiuc_hover.csv` — data dictionary

Phase 1a output. One row per **propeller operating point** measured under
static (hover) conditions.

| | |
|---|---|
| Rows | 2,121 |
| Propellers | 119 |
| Families | 24 |
| Columns | 50 |
| Source | [UIUC Propeller Database](https://m-selig.ae.illinois.edu/props/propDB.html), volumes 1–2 |
| Built by | `src/build_dataset.py` (reproduce raw data with `src/fetch_data.py`) |

## The row-count caveat

**2,121 rows is not n = 2,121.** Each propeller contributes 11–55 rows (one per
measured RPM), and every one of those rows carries an **identical copy of the
36 geometry columns** — the thing a model predicts. The effective sample size
is **119 propellers**.

Any train/test split must therefore group on `prop_name`
(`GroupKFold(groups=df.prop_name)`), never split rows at random. A random split
puts the same blade in train and test and inflates R² to near-meaningless
levels. Report both numbers — "2,121 operating points across 119 propellers".

## Columns

### Identity / grouping keys

| column | type | notes |
|---|---|---|
| `prop_name` | str | Base propeller, matching its geometry file. **The grouping key for cross-validation.** |
| `family` | str | Manufacturer/series prefix (`apcsf`, `gwsdd`, …). 24 distinct. A stricter CV grouping than `prop_name` — see below. |
| `blade_count` | int | 2, 3, or 4. From a `_3b_`/`_4b_` filename infix; 2 is the unmarked default. |
| `diameter_in` | float | Inches, parsed from the filename. 2.24–19.00. |
| `pitch_in` | float | Inches, parsed from the filename. |
| `source_file` | str | Originating `*_static_*.txt`. Six propellers were tested more than once; both runs are kept. |

### Measured operating point

| column | type | notes |
|---|---|---|
| `rpm` | float | 1,261–27,050. As measured. |
| `CT` | float | Thrust coefficient, dimensionless. |
| `CP` | float | Power coefficient, dimensionless. |

### Derived (see `src/physics.py`)

$$T = C_T \rho n^2 D^4 \qquad P = C_P \rho n^3 D^5$$

with $n$ in rev/s, $D$ in metres, $\rho = 1.225$ kg/m³ (ISA sea level).

| column | units | notes |
|---|---|---|
| `thrust_N` | N | 0.003–16.2 |
| `thrust_gf` | gram-force | 0.26–1,654 |
| `shaft_power_W` | W | 0.004–161 |
| `thrust_per_watt_N_W` | N/W | The success metric. |
| `thrust_per_watt_gf_W` | gf/W | Same, in the unit the RC/drone world quotes. Median 13.1. |

**Power is mechanical shaft power at the propeller, not electrical draw.**
Motor and ESC losses are out of scope for v1 (see `SCOPE.md`). This is also
what the source data supports, since it derives from measured thrust and torque.

### Labels — blade geometry (36 columns)

`c_R_r015` … `c_R_r100` and `beta_r015` … `beta_r100`

18 radial stations at `r/R` = 0.15 to 1.00 in 0.05 steps. The suffix encodes
the station: `r015` is `r/R` = 0.15.

- `c_R_*` — chord at that station, **normalized by tip radius R** (not diameter).
  Physical chord in inches = `c_R * diameter_in / 2`.
- `beta_*` — blade angle (twist) at that station, **degrees**.

All 18 stations are stored, though `SCOPE.md` specifies 6–10. That requirement
describes what the *tool* outputs to a user; storing full fidelity keeps the
Phase 2 representation choice open (raw 18, a 9-station subsample, or PCA
scores — PCA reaches ~94% variance in 5 components).

The station grid is verified identical for every propeller at build time.

## Exclusions

Starting from 264 static files across the four volumes:

| excluded | count | reason |
|---|---|---|
| No geometry file | 124 propellers | Volumes 3–4 ship performance data with no geometry at all, so they cannot supply labels. |
| Unparseable name | 8 propellers | No diameter encoded → no `CT` → thrust conversion possible. |

The 8 dropped by name:
`cfnq_45p1`, `cfnq_45p2`, `cfnq_45t1`, `cfnq_45t2`,
`nr640_5_15deg`, `nr640_9_15deg`, `pl_triturbo`, `union_u80`

`nr640_*_15deg` state a blade angle rather than a pitch length; the `cfnq`,
`pl_triturbo` and `union_u80` names encode no size at all.

## Known quirks

**Five propellers are dimensioned in millimetres, not inches** — `vp_140x45`,
`pl_100x80`, `kpf_96x70`, `ef_130x70`, `pl_57x20`. Nothing in the filename says
so. Confirmed by computing implied thrust: read as inches they give 12–45
*tonnes* on a wind-tunnel bench; read as mm, 29–108 gram-force. `parse_names.py`
converts them. Left uncorrected, $D^4$ scaling makes these the dominant outliers.

**Efficiency diverges at negligible thrust.** `thrust_per_watt_gf_W` reaches
84.8 on `gwsdd_2.5x1`, which produces 0.26 gf — since $T/P \propto 1/(nD)$,
efficiency grows without bound as thrust goes to zero. These are degenerate
operating points, not errors. Any optimizer maximizing efficiency needs a
minimum-thrust constraint or it will chase them.

**Blade count is heavily imbalanced**: 2,000 rows at 2 blades, 75 at 3, 46 at 4
— and the non-2-blade rows come from just four base propellers in two airfoil
families (`da4022`, `da4052`). Enough to represent the feature, not enough to
learn its effect. Phase 1b synthetic BEMT data has to supply that variation.

**`family` is the stricter CV grouping.** Splitting on `prop_name` still allows
`apce_11x7` to train while `apce_11x8` tests — same manufacturer, same airfoil
family, one inch of pitch apart. Reporting `GroupKFold` on `family` as well
tests generalization to an unseen manufacturer, which is closer to what the
tool actually faces.

**`gwsdd_9x5` has duplicate geometry files** in volumes 1 and 2. They are
numerically identical; `readers.py` verifies this rather than assuming it.
