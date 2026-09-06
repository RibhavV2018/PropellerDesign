# ML + CAD eVTOL/UAS Propeller Design — Project Plan

**Owner:** Ribhav Vallishayee
**Origin:** High school AIAA paper, "Using Machine Learning to Expedite Production of Aircraft Propellers" (XGBoost, 17-aircraft dataset)
**Goal of this revision:** Turn the original proof-of-concept into a project that (1) fixes the scientific weaknesses of the original paper, (2) retargets it from legacy piston aircraft to small UAS/eVTOL-scale propellers — the segment of aerospace actually growing right now — (3) outputs an actual 3D-printable/manufacturable propeller design instead of three numbers, and (4) is strong enough to anchor a mechanical engineering portfolio, résumé line, and a possible AIAA follow-up.

**Why UAS/eVTOL instead of general aviation:** the eVTOL/drone market is scaling fast (roughly $0.76B in 2024 projected to $17.3B by 2035), and every one of those aircraft needs multiple custom, rapidly-iterated propellers rather than one off-the-shelf design — which is exactly the "propeller design takes too long" problem your original paper was built around. It also happens to fit your existing data sources better: the UIUC and APC datasets are overwhelmingly propellers in the 2–30 inch range, which is UAS/eVTOL scale, not WWII-bomber scale. And because the parts are small, this version of the project is bench-testable on a desktop 3D printer and a cheap thrust stand — something a full-size aircraft propeller project never could be.

This plan is organized in phases, not fixed calendar dates, because your available time will vary with coursework and internships. Each phase lists what "done" looks like, roughly how long it takes, and a fallback if you're short on time. Treat Phases 1–4 as the required core — that alone fixes everything wrong with the original paper. Phases 5–7 are what make it stand out.

---

## Phase 0 — Scope and success criteria (~1 week)

Before touching code, pin down what "finished" means, or this will sprawl indefinitely.

- **Pick a target propeller class.** Scope this version to small electric propellers for multirotor/eVTOL and fixed-wing UAS use — roughly 2–30 inch diameter, driven by brushless motors, the same scale the UIUC and APC datasets actually cover. This also means picking which mission profile you're designing for: hover-optimized (multirotor/eVTOL lift rotors, judged mostly on thrust per watt at a given RPM) vs. cruise-optimized (fixed-wing UAS pusher/tractor props, judged more like the original paper's efficiency framing). Pick one as your primary case — hover-optimized is the more distinctly "eVTOL" story and the simpler physics to start with — and treat the other as a stretch extension.
- **Write down the final input/output contract**, e.g.:
  - Inputs (hover case): target thrust, motor max RPM/Kv and voltage, propeller diameter constraint, number of blades (or let the model choose), number of rotors (for weight/thrust budgeting).
  - Inputs (cruise case, if you extend to it): aircraft/airframe weight, cruise speed, max RPM, motor power, propeller diameter constraint.
  - Outputs: a full blade geometry — chord and twist at ~6–10 radial stations — not three scalars.
- **Write down your success bar**, e.g.: "predicted geometry, when simulated in a physics solver, produces thrust/efficiency within X% of a known reference propeller for 3 held-out test cases, and the geometry exports as a valid, non-self-intersecting solid model sized to mount on a standard brushless motor shaft."
- **Decide your time budget.** If you only have a few weekends, do Phases 1–4 and stop — that's already a legitimate v2. If you have a full semester or a break, push through Phase 6–7.

**Deliverable:** a one-page scope doc (input/output spec, target UAS/eVTOL class, success metric). Keep it in your repo as `SCOPE.md` — it's also useful if you ever write this up for AIAA again, since reviewers will ask exactly these questions.

---

## Phase 1 — Real data, replacing the 17-row table (~2–3 weeks)

This is the single highest-leverage fix. Two complementary sources:

**1a. Real measured performance data (ground truth from the world, not from your own formula):**
- [UIUC Propeller Data Site](https://m-selig.ae.illinois.edu/props/propDB.html) — geometry (chord/twist distributions) and wind-tunnel-measured thrust/torque/efficiency curves for hundreds of small propellers, almost entirely in the exact 2–30 inch UAS/eVTOL range you're now targeting. A pre-cleaned version exists on [Kaggle](https://www.kaggle.com/datasets/heitornunes/uiuc-propeller-database).
- [APC Propellers performance data](https://www.apcprop.com/technical-information/performance-data/) — another real, larger, freely available performance dataset, also squarely at RC/drone scale, with thrust/torque/RPM curves that map directly onto the hover-case inputs above.
- Action: write a data-ingestion script that pulls these into one clean table (geometry + performance), instead of hand-typing a 1973 military report. This alone takes you from n=17 to n in the hundreds, and unlike the original dataset, it's now data that actually matches your target use case rather than being repurposed from it.

**1b. Physics-generated synthetic data (to fix the "circular ground truth" problem):**
Your original blade angle was derived from a *constant* assumed efficiency (0.8) and *constant* angle of attack (8.5°) for every aircraft — that's not real, it's an approximation baked into the labels. Replace it with an actual blade-element-momentum-theory (BEMT) solver so your "ground truth" reflects real aerodynamics per-case, not one fixed assumption:
- Easiest path: use an existing open-source BEMT tool as your data generator. [QPROP](https://web.mit.edu/drela/Public/web/qprop/) (Mark Drela, MIT) is a particularly good fit here — it was written specifically for propeller/motor-driven small electric aircraft and UAVs, i.e. exactly this use case — or [OpenProp](https://transistor-man.com/files/cnc_prop/Epps_OpenProp_SNAME_090420.pdf) (MATLAB, MIT). Run it thousands of times across a randomized sweep of target thrust/RPM/diameter/blade-count to generate a large, physically consistent synthetic dataset.
- Harder but more "yours": write your own simple BEMT solver in Python (a few hundred lines — it's a converging iterative loop over blade stations). This is a legitimate mechanical-engineering exercise on its own and would make a strong appendix/section in a rewritten paper.
- Recommendation: do the easy path first (wrap an existing tool), and only write your own solver if you have time left over — don't let this become the whole project.

**Deliverable:** a combined dataset (real UIUC/APC data + synthetic BEMT-generated data) of at least several hundred rows, each with full input specs and full output geometry (not single scalars), stored as a clean CSV/Parquet with a data dictionary.

---

## Phase 2 — Reframe the ML problem (~2–3 weeks)

- **Change the output from 3 scalars to a geometry vector.** Predict chord and twist at N radial stations (e.g., 8 stations from root to tip), or predict a small number of CST/Bézier parameters that define the whole blade shape smoothly (this is what the current published work — e.g. the CST + deep-learning + genetic-algorithm paper — does, and it avoids jagged, physically-invalid blade shapes that a per-station scalar predictor can produce).
- **Multi-output regression:** XGBoost supports multi-output natively now (or wrap it with scikit-learn's `MultiOutputRegressor`); also try a small neural network (a few dense layers) since you now have enough data for one, and compare.
- **Do this properly, statistically:**
  - k-fold cross-validation (5-fold minimum) instead of a single 80/20 split — with only a few hundred rows, a single split is still noisy.
  - Report R² and per-output error (not one blended "accuracy %" — your original paper's "91.95% accuracy" number was ad hoc; use standard regression metrics reviewers will recognize: MAE, RMSE, R²).
  - Compare against two baselines: (a) a simple linear regression, and (b) your original hand-derived formula. If XGBoost doesn't clearly beat both, that's an important, honest finding to report, not something to hide.
- **Feature importance:** XGBoost gives you this for free (`feature_importances_` / SHAP values) — use it to say something real about which inputs actually drive propeller geometry. This is easy to add and makes the ML section much more substantive.

**Deliverable:** a trained multi-output model with cross-validated metrics, a baseline comparison, and a feature-importance analysis, all in a reproducible notebook/script.

---

## Phase 3 — Closed-loop physics validation (~1–2 weeks)

This is what separates "the model output a number" from "the model output something that actually works." For your held-out test cases:
1. Predict the geometry with your ML model.
2. Feed that predicted geometry back into the BEMT solver from Phase 1b.
3. Check whether the *simulated performance* (thrust at target RPM, efficiency) of your predicted propeller actually meets the target case's requirements — not just whether the geometry numbers are numerically close to some reference blade.

This is a much stronger validation story than "predicted radius was within 8% of real radius," because it tests whether the design actually *works*, which is the point of the whole project.

**Deliverable:** a validation table/plot: for each test case, target performance vs. simulated performance of the ML-predicted propeller.

---

## Phase 4 — CAD generation (~2–4 weeks)

This is the piece your original paper explicitly flagged as future work and never did — closing this loop is probably the single biggest differentiator for a mechanical engineering portfolio, since it turns a data-science script into an actual manufacturable part.

- Take the predicted chord/twist-per-station output and loft it into a 3D solid:
  - **Recommended:** [CadQuery](https://cadquery.readthedocs.io/) or [build123d](https://build123d.readthedocs.io/) — both are Python CAD kernels that plug directly into your existing pipeline (no context-switch out of Python), and both can loft a series of airfoil cross-sections (scaled by chord, rotated by twist, placed along the radius) into a solid blade.
  - **Alternative:** since you already know Onshape/Fusion 360 from your NASA internship, both have scripting APIs (Onshape's FeatureScript/REST API, Fusion's Python API) if you'd rather build the model there.
- Use a standard airfoil profile (an Eppler or Clark-Y section, consistent with the assumption in your original paper) scaled and twisted per station, then loft root-to-tip, then pattern around the hub for the number of blades.
- Export as STEP (for CAD interchange) and STL (for 3D printing/viewing).
- **Because you're now at UAS scale, don't treat the hub/mount as a stretch goal — build it in.** Parametrize a standard brushless-motor mount (a set-screw prop adapter or a bolt-pattern hub sized to a common motor shaft diameter) so the output is a drop-in part, not just a shape. This is the step that makes the project genuinely bench-testable, which a full-size aircraft version of this project never could be.

**Deliverable:** a script that takes ML-predicted geometry in and produces a STEP/STL propeller model out, mountable on a real small brushless motor, with at least one physically printed example.

**Phase 4.5 — Physical validation (optional but high-value, ~1 week):** 3D print a predicted propeller, mount it on a brushless motor/ESC, and measure actual thrust and current draw with a cheap thrust stand (e.g. an RCbenchmark-style load cell rig, or even a DIY load-cell-on-a-scale setup). Comparing real bench data against your BEMT prediction and your original ML prediction is a genuinely rare thing for a student project to include — most of the published ML-for-propeller-design papers you'll be compared against stop at simulation.

---

## Phase 5 — Optimization loop (stretch, ~2–3 weeks)

Once Phases 1–4 work, the natural next step (and what the more advanced published papers do) is to stop treating this as single-shot prediction and start treating it as design search:
- Wrap your ML model as a fast surrogate inside a genetic algorithm or Bayesian optimizer (e.g. `scikit-optimize`, `DEAP`, or `pymoo`).
- Search geometry space for the design that maximizes efficiency subject to thrust/RPM/diameter constraints, using the ML surrogate for speed.
- Verify the top 1–3 candidates with the full BEMT solver (since the surrogate can be wrong at the edges of its training distribution).

**Deliverable:** given a target aircraft spec, the pipeline outputs not just "a" propeller but the best propeller found under your constraints, with a verified performance estimate.

---

## Phase 6 — Package it as a tool (stretch, ~2–3 weeks)

You already built a web-based mission-planning tool at NASA and shipped a full iOS app with a content pipeline at Treebeard — this phase is mostly assembling skills you already have:
- A small web frontend (even a simple form) where a user enters rotor/mission specs (target thrust, motor Kv, diameter constraint) and gets back a downloadable CAD file plus a performance summary/plot.
- Doesn't need to be fancy — a working local demo you can screen-record is enough for a portfolio; a hosted version is a bonus.

**Deliverable:** a short demo video or hosted tool link you can put in your portfolio/résumé.

---

## Phase 7 — Write-up (~1–2 weeks, do this regardless of how far you get)

- Rewrite the paper (or write a new one) with the corrected methodology: real/physics-generated data, cross-validated multi-output regression, baseline comparisons, closed-loop physics validation, and (if you get there) CAD generation and optimization.
- Put the whole thing on GitHub with a clear README, the data pipeline, the model, and the CAD generation script — this is the artifact you'll actually link from your résumé/portfolio, more than the PDF.
- If you want to chase it: this revised version, especially with CAD output and (if you get to it) bench validation, is a legitimate re-submission or extension for a student conference — AIAA Region IV again (its scope covers UAS/eVTOL work fine), an AIAA SciTech student paper, or a broader undergraduate research symposium at UMD. You already have one accepted abstract on this exact topic, which is a real head start, and the eVTOL/UAS reframing arguably makes it a *more* current fit for those venues than the original piston-aircraft framing was.

---

## Suggested pacing

You're a sophomore with limited free time, so don't try to do this all in one sprint:

- **Now → end of this semester:** Phase 0 + Phase 1 (scope + real/physics data). This is the part that most fixes the paper's credibility and can be done in scattered evenings/weekends.
- **Winter break:** Phase 2 + Phase 3 (proper multi-output modeling + closed-loop validation). This is the part that benefits from a few uninterrupted days.
- **Spring semester (light effort, ongoing):** Phase 4 (CAD generation) — this is the differentiator, worth prioritizing over Phases 5–6 if time is tight.
- **Next summer / next break:** Phases 5–6 (optimization loop, tool packaging) if you want to push it further — these are genuinely stretch goals, not required for a strong portfolio piece.
- **Whenever you stop:** Phase 7 (write-up) — always do this last step regardless of which phase you stopped at, so the work is legible to anyone looking at your portfolio.

## Minimum viable "done" for portfolio purposes

If you only get through Phase 1–4, that is already a complete, legitimate project: real/physics-based data, a properly validated multi-output ML model with baseline comparisons, closed-loop physics validation, and an automatically generated 3D-printable propeller model. That alone is a significant step up from the original paper and is genuinely portfolio-worthy on its own — everything past that is upside, not requirement.

## Tech stack summary

| Purpose | Tool |
|---|---|
| Data | UIUC Propeller Database, APC performance data |
| Physics ground truth / synthetic data | QPROP (electric/UAV-focused BEMT) or OpenProp/XROTOR (or a hand-written BEMT solver) |
| ML | Python, XGBoost (multi-output), scikit-learn, optionally a small neural net |
| Validation | Same BEMT solver, run on ML-predicted geometry |
| CAD generation | CadQuery or build123d (Python), or Onshape/Fusion 360 API |
| Optimization (stretch) | `pymoo` or `scikit-optimize` for GA/Bayesian search |
| Packaging (stretch) | Simple web frontend, reusing your NASA/Treebeard experience |
