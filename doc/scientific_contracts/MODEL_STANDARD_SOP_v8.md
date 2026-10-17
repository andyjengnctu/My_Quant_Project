# Model Standard SOP — Version 8 (2026-10-17)

Status: **USER-AUTHORIZED REPORT REVISION; USER-SIDE PROTECTED APPROVAL NOT INSTALLED.**

## Identity and history

| Report ID | Current version | Current schema fingerprint | Previous version / fingerprint |
|---|---|---|---|
| `model.standard_sop` | v8 | `0daefa4ac6e22ca4` | v7 / `56e5fb1173404d7f` |
| `model.standard_comparison` | v12 | `f64c192d57961e76` | v11 / `0626f6512f145d23` |

Historical result files remain under their original identities. A new report may not be
misrepresented as having old v7 evidence. The in-Python fingerprint is a compatibility
check **not** cryptographically independent user authorization. To make release
authorization tamper-resistant, pin this snapshot in a user-controlled protected Git
reference or another approval authority outside GPT-modifiable code.

## Scope and mandatory human report sections

M1. **Learnability** — existing Groups, Daily rho, Global rho, Pair, Top 10% Target,
Bottom 10% Target, Top-Bottom Target. Global OOS and Breakout OOS are separate tables;
Validation used in M2, not another model row in comparison output.

M2. **Generalization** — existing Delta Daily rho, Delta Pair, Delta Top-Bottom:
Validation -> OOS and Global OOS -> Breakout OOS. The second transition changes
cohort; it is not a time generalization estimate.

M3. **Upside / Downside Validation** — existing Target->MFE, Target->Safety,
Score->MFE, Score->Safety Daily Spearman rho. Safety = -Adverse-to-MFE.

M4. **Joint MFE / Adverse Quality** — each model (and each scope) gets one row:
`Model | Top 10% N | Top 10% MFE (%) | Top 10% Adverse (%) | Top 10% Joint Quality (0..100)`.
No Raw/All comparison rows, no distribution matrix, no hard 2R/1R hit threshold,
no MFE-Adverse as a substitute for true joint selection ability.

## Exact evaluation semantics

- Outcome: **Full-window** `target_favorable_return` (future high relative to
  signal-date anchor), `target_adverse_return_to_peak` (nonnegative drawdown from
  that anchor to the first/selected maximum favorable excursion). Do NOT confuse
  this Adverse with full-window MAE or future realized trade PnL.
- Score selection: for each ranked date/fold group, include exactly
  `ceil(10% * eligible_group_size)`, minimum 1, sorted Score descending;
  tie-break ticker ascending and source row index ascending.
- Percentile truth: canonical `build_same_date_percentile_targets` on actual
  price-return `target_favorable_return` (MFE%) and negative
  `target_adverse_return_to_peak` (Safety = -Adverse%) among all mature valid
  stocks on the underlying date. **Do not** use R-normalized labels as
  percentiles for the price-percent M4 table; stock-specific risk R can reorder
  two stocks compared with actual price-return outcomes.
  The Global and Breakout result rows consume the SAME underlying truth
  percentiles, so the Breakout analysis remains directly comparable with Global.
- Per-stock Joint Quality = `100 * min(Percentile(MFE), Percentile(-Adverse))`.
  `Top 10% Joint Quality` = arithmetic mean across selected valid stock-days.
  This is a relative joint ranking score, **not** calibrated success probability,
  odds of realizing MFE, or broker-net trade performance.
- MFE / Adverse main cells = 100 times arithmetic mean of canonical
  `target_favorable_return` and `target_adverse_return_to_peak` on the SAME
  selected stock-days. No second price/outcome engine.
- Seed aggregation: evaluate every benchmark seed independently and only then
  take the seed-wise arithmetic mean (and report existing required variability).
  Never pool seed rows, pick best seed, or forge old evidence using mean values.
- Universe and PIT: daily-universal scores must not use Breakout qualification as
  fitting input; Breakout is a read-only application slice. Dates, score lineage,
  maturity cutoff and fit identity must pass existing canonical gates.
- Mode / status: Forward, Rolling, and Robustness share the same four chapter
  definitions. Completeness evidence, rolling folds and seeds remain required but
  no longer become numbered display sections. Old v9 payloads lack joint
  per-stock truth and MUST be re-evaluated from valid score/target evidence.
  Missing evidence means BLOCKED, **never** zero or NA.

## Change-control decision

Use this evaluation/report version first; **do not** immediately add a Strategy
RoMD-correlation report. In the next separately authorized read-only Audit,
quantify how M1/M3/M4 per-model **OOS** quality indicators correlate with
strategy RoMD, using matched strategy arms, periods and seeds; preserve matched
PIT/folds, confounders and multiple-testing limits. Any subsequent permanent
SOP revision requires a NEW explicit version and user approval.
