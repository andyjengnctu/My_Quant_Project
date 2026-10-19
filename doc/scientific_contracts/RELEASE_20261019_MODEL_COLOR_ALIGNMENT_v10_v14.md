# Model Standard SOP Color Alignment — 2026-10-19

## Authorization and scope

User request: Research `[1]` model reports do not apply their established color
rules consistently. This release normalizes **only presentation coloring** for
`model.standard_sop@v10` and `model.standard_comparison@v14`. It does not change
metric formulas, evidence or contract columns/section order, model fitting,
PIT/target/cohort or OOS/seed/fold semantics. Original v9/v13 identities remain
historical and must never be rewritten to claim they were v10/v14.

## Approved display rules

The canonical metric column's `preference` (higher/lower/neutral) is the sole
owner of performance direction. The existing `core.report_style` palette is
reused, not duplicated:

- **M1 Learnability** and **M3 Upside/Downside Validation**: for each comparable
  model set, table column and Global/Breakout scope separately, use green for
  best, red for worst; other rows are plain. Lower-preferred metrics reverse
  this ranking. Neutral columns, nonnumeric values and complete ties are plain.
- **M2 Generalization**: changes are measured from the labeled source scope.
  Use signed-delta improvement/deterioration relative to the column preference:
  green for improvement, red for deterioration, neutral for zero/missing. This
  is not a cross-model best/worst competition.
- **M4 Joint MFE/Adverse Quality**, **Rolling Stability** and
  **Across-seed Stability**: preserve the existing preference-based best/worst
  scheme. Adverse magnitude and variability (`lower`) favor smaller values.
- **Single-model report**: no fabricated best/worst rank with one model.
  Signed generalization still uses direction-specific colors.
- **Console/Markdown**: use the same semantic signal and central palette;
  headings/scope colors and all source values retain their prior meaning.

## Artifact and version transition

- Append `model.standard_sop@v10` and `model.standard_comparison@v14` to
  `REPORT_CONTRACT_VERSIONS.json`, retaining byte-identical previous definitions.
  v10/v14 have the same metric schemas as v9/v13; the new fingerprint arises
  from report version identity and their presentation authorization.
- Append a new `tools/validate/baselines/ssot_regression_v4.json` protection
  record with SHA in `ssot_regression_guard.py`; prior v1–v3 Golden files and
  their historical report contracts stay frozen. This protects the new
  user-authorized display version without loosening historical guards.
- When old Standard report output is encountered, only **REPORT DERIVE** is
  permitted with verified canonical source evidence. Never retrain the model,
  rescore OOS or reinterpret old scientific results due to color changes.

## Independent validation boundary

Isolated Report renderer tests cover Console and Markdown across Forward,
Rolling, single/multi model and robustness, including ties and negative rho.
Report-contract freeze, historical SSOT protection and stage-readiness tests
are separate checks. Actual Windows `apps/run_bundle.py` remains user-run.
