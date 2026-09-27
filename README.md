# Propeller Designer

Machine learning plus blade-element physics that turns a hover thrust requirement
into a printable propeller for small electric UAS and eVTOL aircraft.

**[See the showcase site →](https://ribhavv2018.github.io/PropellerDesign/)**
Example designs with 3D viewers and downloadable STL and STEP files, plus the validation
results and known limitations.

Give it a target thrust, a diameter and a motor speed. A model trained on simulated
propellers proposes blade geometry, a blade-element momentum (BEMT) solver checks what
that blade actually does, and CadQuery turns it into STEP and STL files. All performance
figures are **simulated**, not measured.

This rebuilds my high-school AIAA paper, *Using Machine Learning to Expedite Production
of Aircraft Propellers*, retargeted from piston aircraft to 2–30 inch electric hover
rotors. See [SCOPE.md](SCOPE.md) for the input/output contract and success criteria.

## How it works

1. **Requirement.** Thrust, diameter and rpm, plus an optional blade count, become two
   dimensionless inputs: a thrust coefficient and a Reynolds number.
2. **Model.** A linear regression predicts chord and twist at 18 radial stations for each
   blade count from 2 to 6. It's trained on the efficient frontier (29,639 blades) of
   200,000 synthetic propellers generated with the BEMT solver.
3. **BEMT check.** Each candidate is simulated. The most efficient one landing between
   0% and +20% of the thrust target wins.
4. **CAD.** Cambered NACA sections are lofted into blades and patterned around a hub
   with a 6 mm round bore.

## Results in brief

- **Solver vs 119 real UIUC propellers (2,121 operating points).** The median thrust error
  is −12.6%, so the solver under-predicts. The spread is 17% between propellers but only
  4% within one propeller across its rpm sweep.
- **Model.** Linear regression is deployed. XGBoost was benchmarked on the same data and
  tied it.
- **Closed-loop test (Phase 3).** In the 6–11 in range, 8 of 10 designs were no more than
  15% less efficient than the best real propeller for the same requirement. The two
  failures, `ma_11x4` and `grcp_11x4`, auto-selected 5 blades where the real propellers
  use 2, and came out 38–42% worse. Both sides of this comparison are simulated in the
  same solver, so it measures design quality rather than agreement with measured data.
- **Outside 6–11 in** the model extrapolates below its training Reynolds range and does
  markedly worse. The tool warns when a requirement falls there.

The [site](https://ribhavv2018.github.io/PropellerDesign/#validation) has the full table and
the [limitations](https://ribhavv2018.github.io/PropellerDesign/#limits).

## Running it locally

Needs Python 3.9+. CadQuery is a large install (about 2 GB with its VTK dependency).

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python src/patch_xgboost_openmp.py   # macOS without Homebrew only
```

Rebuild the data. The raw UIUC files and the 200k-row synthetic set are not checked in:

```bash
./.venv/bin/python src/fetch_data.py
./.venv/bin/python src/build_dataset.py
./.venv/bin/python src/generate_synthetic.py
```

Then design a propeller:

```bash
./.venv/bin/python src/app.py                       # web UI at http://127.0.0.1:5000
./.venv/bin/python src/cad.py --thrust 500 --diameter 10 --rpm 6000   # CLI, writes STEP + STL
```

Generating one propeller peaks at about 760 MB of memory, which is why the public site is
static rather than a hosted copy of the app.

## Rebuilding the showcase site

```bash
./.venv/bin/python src/export_showcase.py   # regenerates docs/designs/ and docs/data/
python3 -m http.server -d docs 8000         # preview at http://localhost:8000
```

GitHub Pages serves the `docs/` folder.

## Repository layout

| Path | What it is |
|---|---|
| `src/bemt.py` | Blade-element momentum solver for hover |
| `src/validate_bemt.py` | Solver vs the 119 measured UIUC propellers |
| `src/generate_synthetic.py` | Synthetic training propellers, labelled by BEMT |
| `src/models.py`, `src/evaluate.py` | Inverse-design models and the benchmark harness |
| `src/design.py` | Requirement → geometry → simulated performance, with blade-count selection |
| `src/phase3_validation.py` | Closed-loop test against real propellers |
| `src/cad.py` | CadQuery geometry, STEP/STL export and renders |
| `src/app.py` | Local Flask web app |
| `src/export_showcase.py` | Pre-generates the static site's example designs |
| `docs/` | The GitHub Pages site |
| `data/processed/` | Built datasets and validation results ([data dictionary](data/processed/DATA_DICTIONARY.md)) |

## Data

Measured propeller geometry and performance come from the
[UIUC Propeller Data Site](https://m-selig.ae.illinois.edu/props/propDB.html)
(volumes 1–2).
