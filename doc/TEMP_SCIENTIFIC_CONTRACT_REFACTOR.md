# TEMP — Strategy Scientific Contract / Composition Refactor

> **Temporary implementation checklist only.**
>
> This file is not a scientific/config/runtime truth owner.  Every completed rule
> must be moved into its canonical code/config/Registry/Architecture/test owner.
> Delete this file after Round 6 closes and all items below have permanent owners.

## 0. Fixed refactor principles

1. **Shared implementation, frozen scientific semantics.**  Algorithms/formulas stay single-source; a named historical experiment/artifact must not change because a mutable common config later changes.
2. **Scientific vs execution/display separation.**  Any value that can change generated trials, accepted/rejected candidates, selected params, portfolio economics, PIT/data scope, score, or final result belongs to the scientific contract.  Worker count, cache sizing, profiling, console refresh and display scaling do not.
3. **Resolve once, consume everywhere.**  A strategy-parameter job resolves one immutable canonical scientific contract at the boundary.  Inner Optimizer/runtime consumers must eventually stop re-reading mutable scientific globals.
4. **IDs describe different layers.**  Existing Registry namespaces remain authoritative (`DL-*` = DL runtime source, `SR-C*` = strategy runtime/comparison arm, `PARAM-P*` = strategy-parameter stage).  Future rule-component and deployable-composition namespaces must be explicitly allocated in the Experiment Registry during Round 6; `RULE-*` / `STR-*` below are provisional architecture labels only, not pre-allocated scientific IDs.  DL and rules may be chained; they are not mutually exclusive alternatives.
5. **Versioned shared contracts, not copied per experiment.**  Common scientific dimensions (search, score, gate, execution economics, schedule, fitting, validation, selector, data lineage) are referenced by experiments/compositions.  Used contract versions are immutable; new semantics create a new version/identity rather than mutating historical meaning.
6. **Validation is a first-class scientific contract.**  Current no-validation and fixed-checkpoint validation remain historical/control semantics.  A later plateau early-stop protocol will use max budget + warmup + patience + min-delta + metric/direction, sharing the state-machine abstraction with DL where appropriate while keeping DL/non-DL evaluators separate.
7. **Behavior-preserving architecture work must not alter existing scientific/artifact identity.**  Identity-changing migrations happen only in an explicit later round with equivalence/migration evidence.

## 1. Scientific boundary classification

### Must be scientific / frozen for a used identity

- Dataset profile **and canonical dataset-generation lineage**.
- PIT/schedule scope: selection start, coverage end, train window, OOS horizon.
- Search-space definition, categorical choices, `enabled_by`, inactive/fixed defaults and fixed TP policy.
- Strategy/rule parameter schema values that affect trial materialization.
- Score/objective semantics, numerator, MDD treatment and score amplifiers.
- Hard eligibility gates.
- Selector/finalist semantics, including `base-finalists-agree` agreement rule.
- Local-min review and finalist-review budget if enabled.
- Inner-validation and dominant-year anti-overfit semantics.
- Portfolio execution economics and sizing that affect backtest/score: capital, fees, tax, minimum fee/notional, fixed risk, position cap, max positions, rotation, compounding.
- Fitting stochasticity: trial budget, fitting seed protocol, resolved fixed seed bundle / reproducible generator semantics.
- Validation protocol.
- Composition topology once `STR-*` is introduced.

### May stay mutable execution/display config

- CPU/GPU/process worker counts **only when deterministic scientific semantics are unchanged**.
- Cache/prefetch/storage/profiling/resource sampling.
- Console refresh / verbosity / progress rendering.
- Display-only score scaling/precision/colors.
- Workspace/path choices that do not change scientific data identity.
- RUN/REUSE/RESUME/FORCE-REBUILD execution action.

### Special caution

Adaptive/parallel search knobs are execution-only **only after** proving they do not alter suggestion/update order or other scientific stochastic semantics.

## 2. Known pre-refactor defects to close

- `production_multiseed_current` / `PARAM-P6` freezes only part of its semantics; train/OOS window and several Optimizer policies are still read from mutable common config.
- Scientific semantics currently missing or incompletely represented in P6/build identity include hard gates, score details, local-min, inner validation, dominant-year dependency, finalist agreement, fixed TP, schema inactive defaults, full execution economics, and dataset-generation lineage.
- Existing Optimizer effective-policy fingerprint includes display-only policy but omits some result-changing hard-gate semantics.
- The same logical scientific boundary is represented differently by profile/build-contract/effective-policy fingerprint.
- Some validators hard-code the current C85 / P6 / 4-seed / 300-trial values instead of validating registered-profile relationships.
- Human progress output has exposed stale generic fields (`best_local_min`, `seed_min_agree`) in contexts where the active selector does not use those semantics; later display cleanup must consume the resolved contract rather than add formatter exceptions.

## 3. Six-round implementation plan

### Round 1 — Canonical scientific-contract boundary **[DONE]**

Goal: define one resolver/fingerprint boundary without changing current Optimizer behavior or existing artifact fingerprints.

- [x] Add canonical strategy-parameter scientific contract resolver.
- [x] Include scientific sections: profile semantics, dataset lineage, schedule, search/parameter-generation, score, hard gates, selector/anti-overfit, execution economics, fitting, validation.
- [x] Explicitly exclude runtime/display policy from the scientific fingerprint.
- [x] Add generic contract tests proving representative scientific mutations change the new fingerprint while display/runtime-only mutations do not.
- [x] Document the stable architecture boundary in `doc/ARCHITECTURE.md`.
- [x] Do **not** inject the new contract into existing artifact/fingerprint/runtime paths yet.

Exit: one canonical resolved contract exists and current scientific behavior/output identities remain unchanged.

Round-1 evidence: the existing P6 build contract and legacy Optimizer effective-policy fingerprint remain byte-equivalent to the input ZIP; disabled-switch `build_trial_params()` materialization is also byte-equivalent.  The new scientific fingerprint is observation-only and is **not yet** an artifact reuse key.

### Round 2 — Runtime injection / eliminate scientific global re-reads **[DONE]**

- [x] Resolve scientific contract once at Strategy Parameter producer boundary.
- [x] Inject relevant resolved sections into Optimizer session/runtime.
- [x] Remove direct mutable-scientific-config reads from result-changing inner Optimizer paths: parameter generation/inactive defaults, execution economics, buy-sort/EV, score, hard gates, local-min neighborhood/review, inner validation, dominant-year thresholds and finalist-agree.
- [x] Preserve exact current results/seed/trial/order/fingerprints during the behavior-preserving migration.
- [x] Keep performance/display config dynamic.

Exit: runtime scientific semantics come from the resolved contract. Generic/non-Strategy-Parameter Optimizer sessions retain the legacy config-backed fallback; artifact identity binding is intentionally deferred to Round 3.

Round-2 evidence: latest-baseline vs refactored legacy fixed-multiseed build-contract hash and Optimizer effective-policy fingerprint are identical; generic no-contract score/selector/search and trial-param materialization are also byte/value-equivalent. Capability tests (95/95) additionally mutate mutable training config *after* contract resolution and verify the injected session/context continues to use the frozen trial generation, score, buy-sort/EV, local-min, fixed-TP, finalist-agree and dominant-year semantics.

### Round 3 — C85 / PARAM-P6 first full identity consumer **[DONE]**

- [x] Freeze/register current P6/C85 scientific fingerprint without silently changing historical semantics.
- [x] Bind 120M/12M, 4 fixed fitting seeds, 300 trials/seed, `base-finalists-agree`, data lineage, score/gates/search/economics/anti-overfit/validation-disabled semantics.
- [x] Fail fast on scientific contract drift instead of rebuilding a different meaning under the same P6/C85 identity.
- [x] Make progress/report fields contract-driven; remove inapplicable `best_local_min` / `seed_min_agree` leakage from this selector path through the canonical renderer owner.

Exit: C85/P6 has one reproducible scientific meaning.

Round-3 evidence: `production_multiseed_current` is source-registered against the canonical resolved scientific fingerprint; P6 inspect returns `BLOCKED` on common scientific drift, existing manifests without scientific identity are `LEGACY_SCIENTIFIC_IDENTITY_UNBOUND`, and bound manifests must carry the exact resolved contract/fingerprint. Progress visibility is derived from the same contract so finalist-agree/local-min-disabled runs no longer leak `seed_min_agree` / `best_local_min`; no optimizer result semantics changed. Permanent formal ownership is registered as B422/T560 in `doc/TEST_SUITE_CHECKLIST.md`; B253 now explicitly distinguishes ordinary artifact/workspace staleness from scientific-identity drift.

### Round 4 — Migrate existing strategy-param / Strategy Compare identities **[NEXT]**

- [ ] Migrate C58/C61/C62/C84/C85 and relevant `PARAM-P*` lineage to the same contract system.
- [ ] Keep completed historical evidence immutable.
- [ ] Replace validators that hard-code current C/P/seed/trial values with registry/profile relational checks.
- [ ] Confirm equivalent existing artifacts remain reusable only when their resolved scientific fingerprint matches.

Exit: old and current Research arms share one identity mechanism.

### Round 5 — Validation redesign **[PENDING]**

- [ ] Preserve no-validation baseline as an immutable validation contract.
- [ ] Preserve fixed-checkpoint P5/C84 validation as historical/control semantics.
- [ ] Add plateau early-stop validation contract: max budget, warmup/min observations, patience, min delta, metric, direction, PIT-safe validation window.
- [ ] Share a generic early-stop state-machine abstraction with DL where valid; keep DL/non-DL evaluation computation separate.
- [ ] First validate single-seed behavior/cost, then extend to multi-seed.
- [ ] Produce a new `PARAM-P*`; never overwrite P6/P5 meaning.

Exit: validation budget is adaptive and versioned, with no OOS feedback leakage.

### Round 6 — Strategy component/composition identity + closure **[PENDING]**

- [ ] Define the permanent component/composition identity model through the canonical Experiment Registry before allocating any new prefix.  Preserve existing meanings (`DL-*` source, `SR-C*` runtime/comparison arm, `PARAM-P*` parameter stage); provisional `RULE-*` / `STR-*` labels may be adopted only after explicit Registry allocation.
- [ ] Represent fitted rule-parameter artifacts with the existing `PARAM-P*` namespace unless the Registry explicitly defines a different layer.
- [ ] Keep Research evaluation/comparison arms in the existing `SR-C*` namespace unless the Registry explicitly changes that contract.
- [ ] Support rule-only, DL-only, and chained/hybrid topologies (`RULE -> DL`, `DL -> RULE confirmation`, multi-component compositions) without separate engines.
- [ ] Make composition identity reference shared versioned scientific contracts rather than duplicate their values.
- [ ] Complete drift/invariance regression coverage and permanent docs/Registry owners.
- [ ] Mark every item in this file migrated to a permanent owner.
- [ ] Delete `doc/TEMP_SCIENTIFIC_CONTRACT_REFACTOR.md` in the final closure patch.

Exit: component identity, scientific contracts, runtime and historical experiment lineage are all single-source and the temporary checklist no longer exists.

## 4. Round status

| Round | Status | Scientific behavior intentionally changed? |
| --- | --- | --- |
| 1 | DONE | No |
| 2 | DONE | No |
| 3 | DONE | No; identity binding/migration only |
| 4 | NEXT | No; historical migration only |
| 5 | PENDING | **Yes — explicit validation research change** |
| 6 | PENDING | No for identity refactor; new compositions later require explicit research identity |

## 5. Deletion gate for this temporary file

Delete this file only when all are true:

- [ ] All six rounds are complete or explicitly closed by user decision.
- [ ] Every surviving rule has a permanent code/config/Registry/Architecture/test owner.
- [ ] No production/runtime module imports or parses this file.
- [ ] No formal/synthetic validator treats this file as a truth source.
- [ ] Historical experiment/artifact identities remain reproducible.
- [ ] Current production/Research flows pass local formal double-check from `apps/run_bundle.py` after the final patch.
