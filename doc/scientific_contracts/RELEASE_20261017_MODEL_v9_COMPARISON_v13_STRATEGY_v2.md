# Approved Scientific Report Contract Release — 2026-10-17

Authorized by the user request: finish all project governance and SOP work items 1–4 in one iteration; explicitly exclude item 5 (DL Metric → RoMD Audit). This release is **not** an OOS result or permission to train any model.

## Immutable identities

- `model.standard_sop@v9` — M1–M4 calculations, maturity, PIT, scoring and Top 10% semantics identical to v8. Only machine-readable `sop_view` schema now explicitly allowlists `learnability`, `generalization`, `upside_downside_alignment`, `joint_mfe_adverse_quality`, `evaluation_mode`, `rolling_specific`, `robustness_specific`. Source raw evidence and explicit historic Audit sidecars remain available separately; `extensions`, `ranking`, `evidence`, `split_names`, `comparison_extension_aggregation` no longer appear in v9.
- `model.standard_comparison@v13` — exactly the same view for each model and no root `model_specific_extension_ids`; no historical Audit computations during Standard SOP robustness. Same legal matched stock/day, fold, horizon and benchmark seeds remain mandatory.
- `strategy.standard_sop@v2` — four sections below, shared by Forward, Rolling and Robustness. Trade execution, realized PnL, initial-risk Research EV(A), fee rebate timing, annualization, and RoMD are unchanged. Trading EV(R) untouched.

## Strategy S1–S4 definitions

1. **S1 Core Performance:** `Arm`, `Strategy`, `Total Return` (net broker-cash total return %), `Annualized Return` (canonical annual %), `MDD` (positive % magnitude), `RoMD` (existing total return / MDD), `Worst Full Year` (lowest **complete** calendar-year return %; partial years excluded).
2. **S2 Trade Quality:** `Arm`, `Strategy`, `Trades` (completed round trips), `Win Rate` (%), `Payoff` (average win / abs average loss), `EV` (original Research EV(A) in R). This does not redefine Trading's EV(R).
3. **S3 Capital / Execution:** `Arm`, `Strategy`, `Avg Exposure` (%), `Fill Rate` (% of canonical reserved orders filled), `Avg Orderable` (eligible daily candidates), `Supply Gap Days`, `Underfilled Days`, `Position Gap` (slot days). Reuse existing portfolio/capacity producers; do not infer defaults.
4. **S4 Yearly Results:** `Year`, `0050`, one column for each Compare Suite arm in enabled order. Entries are signed percentages of canonical source annual/partial-year return. `YYYY*` indicates partial year; S1's worst full year uses only full years. The 0050 reference is the **existing canonical adjusted-price 0050 benchmark return**, NOT a fee-adjusted buy-and-hold portfolio. This distinction is mandatory in every human report. The strategy columns remain broker-cash after applicable fees/taxes. Do not misrepresent them as same-cost returns or fabricate a broker-cash 0050. A future cost-matched benchmark would be a new scientific contract.

Canonical benchmark 0050 price evidence is carried from the existing `core.portfolio_benchmark._get_benchmark_period_stats` producer into a **new, read-only report sidecar** (`bm_report_yearly_return_rows`) on the portfolio profile, through `strategy_compare_engine` to `benchmark_yearly`, and through the fixed baseline into multi-seed reports. This new sidecar preserves both full and partial calendar years. The historical `bm_yearly_return_rows` remains full-year-only and is unchanged. No portfolio return, risk, fee, or 0050 price formula is replaced. Missing, inconsistent, nonfinite or date-less values BLOCK complete report publication. Historical pre-v2 reports may lack this evidence; they are preserved as historic and must not be relabelled as v2. Missing new *report* evidence must not silently invalidate a fitted DL checkpoint or retrain strategy parameters.

## Audit and historical scopes

The seven chapter v1 strategy rendering is historic; no standard v2 section may auto-display v1 Selection Quality, MFE×Safety, Upside Survival or First-Passage. The standard Strategy v2 runner no longer auto-materializes a `strategy_diagnostics.md` file; historical diagnostic code and raw pair evidence remain available only for an explicitly requested read-only Audit. Model mode Robustness retains seed-statistic measures but removes the obsolete "Model Extension / aggregation / Status" table. Historical source `strategy_comparison.json` pair diagnostics can still be examined in **explicit read-only Audit**. Robustness maintains same-seed contrasts, across-seed RoMD distribution, and per-fold/per-year sources as independently approved mode evidence. Runtime execution duration is operational metadata, not a fifth strategy scientific chapter.

## Change and approval

These three new versions are append-only additions to `REPORT_CONTRACT_VERSIONS.json`; previous v7/v8, v11/v12, and Strategy v1 records and definition bytes are unmodified. The separate SSOT Golden is **version-bumped to v3**; it protects frozen v1 and v2 Golden SHA256s and the entire historical contract prefix. The v3 protected-test AST snapshot is independently recorded in its immutable Golden. Prior Golden files, contract version records and earlier schema definitions are preserved byte-for-byte. Neither local Git tags nor an extra command are required. The in-repository hash/validator mechanism detects ordinary drift but is not independently tamper-proof if all code and version files can be modified by the same actor.

## Release gates

- Forward, Rolling and Robustness complete evidence and PIT/maturity/seed identities still mandatory.
- RUN/REUSE share the same canonical schema; v1, v8, v12 historical JSON may be read as historical truth but never published under the new version IDs without validated transformation and required source evidence.
- Any new permanent indicator requires a **new** `report_id@vN`, user review and an independently evaluated reproducible scientific rationale.
- Item 5, DL metric relationship to Strategy RoMD, is deferred; results of a future Audit are not silently promoted into this release.

## Local targeted verification (not the Windows formal suite)

- Research report contract synthetic: 48/48 PASS.
- SSOT structural/immutable-Golden regression: 16/16 PASS.
- Strategy Compare contract checks: 19/19 PASS.
- Strategy readable SOP v2 contract checks: 76/76 PASS, including hard failure when 0050, a full-year cohort, or Research EV evidence is missing.
- Independent renderer smoke: Plain/Console/Markdown S1–S4 all render; partial year `*` appears and benchmark fee semantics are disclosed.
- No official `apps/run_bundle.py` or `apps/test_suite.py` execution was performed by ChatGPT. Local formal integration remains the user's acceptance step.
