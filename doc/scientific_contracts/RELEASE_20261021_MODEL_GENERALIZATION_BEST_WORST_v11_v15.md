# Model Standard SOP Generalization — Color Revision (2026-10-21)

## User authorization

User explicitly requests that `MODEL SOP | 2. Generalization` adopt the same
**best green / worst red** rule as the other model-comparison sections. This
is a **presentation-only** authorization, without changes to formulas, sign,
labels, relative evaluation modes, eligibility, historical evidence, training,
model inputs, target, PIT, folds, seeds or artifact identity.

## Approved display behavior

- Each **Generalization transition** is compared independently:
  `Validation → OOS` and `Global OOS → Breakout OOS` are distinct tables.
- For each numeric metric column within one transition, compare only different
  models displayed in that table using the column's canonical `preference`.
  Best receives shared semantic **green**, worst receives **red**.
  These columns currently all prefer **higher** deltas. Thus `-0.01` can be
  greener than `-0.03`, and `+0.01` can be redder than `+0.03`.
- Middle values are plain. Exact/near ties at the extrema share the corresponding
  highlight; a complete tie, missing/non-finite values, neutral columns, and a
  single-model table have no best/worst highlight.
- Console and Markdown use `core.report_style.best_worst_signals` and
  `styled_signal`, with existing palette, no additional judgement vocabulary.
  All M1/M3/M4 and Rolling/Across-seed colors remain unchanged.

## Version and evidence preservation

- Append `model.standard_sop@v11` and `model.standard_comparison@v15` to the
  append-only report registry with new immutable definition JSON files.
  Scientific metric columns, preferences, values and ordering are unchanged.
  The report version identity changes, not the training/evidence producer's
  existing `standard_model_sop_v10` scientific schema.
- Historical v7–v10 Model SOP and v11–v14 Comparison definitions remain
  unchanged. Prior report evidence is retained under its original identity.
- Existing verified FIT / SCORE / EVIDENCE is eligible for **REPORT DERIVE**
  only; a color-only revision does not authorize retraining, rescoring or
  backfilling historical research artifacts.
- Append SSOT regression v5 protection for the new versions. Earlier v1–v4
  Golden files and their hashes must remain byte-identical.

## Review boundary

Independent renderer/color-contract tests verify separately ranked transitions,
positive/negative delta counterexamples to sign-based colors, single-model
neutrality, ties/missing values and both Console/Markdown targets. User's
Windows formal `apps/run_bundle.py` remains the only full integration gate.
