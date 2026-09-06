# UAS/eVTOL Hover Propeller Design — Scope (Phase 0)

## Target class
- Propellers: 2–30 inch diameter, fixed-pitch, brushless-motor-driven, electric.
- Primary mission: hover-optimized (multirotor/eVTOL lift rotors). Cruise-optimized fixed-wing UAS props are an explicit stretch extension, not part of v1.

## Input / output contract

**Inputs:**
- Target thrust (grams-force or Newtons) at the design point.
- Diameter constraint (max allowable diameter — e.g. from frame arm length or ground clearance).
- Target operating RPM (assumed derived externally from motor Kv × battery voltage — the motor itself is not modeled electromechanically in v1; see Assumptions below).
- Blade count — **optional.** If specified, it's a fixed categorical input and the model designs geometry for exactly that count. If not specified, the tool internally evaluates a small candidate set (2 through 6 blades), predicts geometry + performance for each, and returns whichever candidate maximizes thrust per shaft watt while meeting the thrust and diameter constraints.

**Outputs:**
- Chord and twist at ~6–10 radial stations (the blade geometry).
- Number of blades actually used (echoed back, since it may have been auto-selected).
- Predicted thrust and predicted mechanical shaft power at the design point, from which thrust per shaft watt is computed.

## Success metric
- At the single design point specified: predicted thrust per shaft watt from the ML pipeline, when the resulting geometry is simulated in the BEMT solver (Phase 1b/3), is within X% of the best available real reference propeller at that same thrust/RPM/diameter from the UIUC/APC dataset. (Pick a firm number — 10–15% is a reasonable starting target — once you have baseline error rates from Phase 2.)
- Geometry exports as a valid, non-self-intersecting solid, correctly sized to mount on a standard brushless motor shaft (Phase 4).
- When blade count is left unspecified, the tool's auto-selected count matches or plausibly beats the blade count of the closest real reference propeller for that design point, on a majority of held-out test cases.

## Explicit assumptions / out of scope for v1
- Motor is represented only as a target RPM, not a full torque-speed/Kv/resistance model. Solving for the equilibrium RPM from raw motor specs and battery voltage is a documented stretch extension, not v1.
- Power means mechanical shaft power at the propeller, not battery/electrical draw — motor and ESC losses are not modeled. This keeps the metric consistent with the UIUC and APC datasets, which report measured thrust and torque (hence shaft power) rather than electrical input. Converting to thrust per electrical watt would require a motor efficiency model, and is a documented stretch extension that pairs naturally with the Kv/voltage RPM solver above.
- Efficiency (thrust per shaft watt) is judged at one specified design point, not across a full throttle sweep. A full efficiency curve is a documented stretch extension.
- Standard sea-level air density; no altitude/temperature correction.
- Isolated single-rotor performance only — no multirotor prop-wash interference between arms, no ground effect.
- Fixed-pitch propellers only (no variable-pitch/collective mechanisms).

## Why the optional blade-count design is safe to build now
It doesn't require redesigning the model — blade count is a categorical input feature the model needs to see regardless, and the UIUC/APC training data already spans many blade counts, so there's no new data burden. "Auto-select best count" is a thin wrapper that loops model inference over a small candidate set and keeps the best result. It's effectively a preview of the fuller Phase 5 optimization loop, scoped down to one discrete variable instead of a continuous geometry search.
