# TEMP — Strategy Scientific Contract / Composition Refactor

> **Temporary implementation checklist only.**
>
> This file is not a scientific/config/runtime truth owner.  Every completed rule
> must be moved into its canonical code/config/Registry/Architecture/test owner.
> Delete this file after Round 7 closes and all items below have permanent owners.

## 0. Fixed refactor principles

1. **Shared implementation, frozen scientific semantics.**  Algorithms/formulas stay single-source; a named historical experiment/artifact must not change because a mutable common config later changes.
2. **Scientific vs execution/display separation.**  Any value that can change generated trials, accepted/rejected candidates, selected params, portfolio economics, PIT/data scope, score, or final result belongs to the scientific contract.  Worker count, cache sizing, profiling, console refresh and display scaling do not.
3. **Resolve once, consume everywhere.**  A strategy-parameter job resolves one immutable canonical scientific contract at the boundary.  Inner Optimizer/runtime consumers must eventually stop re-reading mutable scientific globals.
4. **IDs describe different layers.**  Existing Registry namespaces remain authoritative (`DL-*` = DL runtime source, `SR-C*` = strategy runtime/comparison arm, `PARAM-P*` = strategy-parameter stage).  Future rule-component and deployable-composition namespaces must be explicitly allocated in the Experiment Registry during Round 7; `RULE-*` / `STR-*` below are provisional architecture labels only, not pre-allocated scientific IDs.  DL and rules may be chained; they are not mutually exclusive alternatives.
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

## 3. Seven-round implementation plan

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

Round-3 evidence (Round-5 stabilization correction): `production_multiseed_current` is source-registered against the canonical resolved scientific fingerprint; P6 inspect returns `BLOCKED` on true common scientific drift, while existing manifests without scientific identity are `LEGACY_SCIENTIFIC_IDENTITY_UNBOUND` and must route to canonical `REBUILD` (never REUSE/direct-adopt) when the registered source still exact-matches. Bound manifests must carry the exact resolved contract/fingerprint. Progress visibility is derived from the same contract so finalist-agree/local-min-disabled runs no longer leak `seed_min_agree` / `best_local_min`; no optimizer result semantics changed. Permanent formal ownership is registered as B422/T560 in `doc/TEST_SUITE_CHECKLIST.md`; B253 now explicitly distinguishes ordinary artifact/workspace staleness from scientific-identity drift.

### Round 4 — Non-validation identity convergence **[DONE]**

- [x] Register canonical non-validation policy-output profiles for Min `base-finalist-best`, Full `base-finalist-best`, and Full `base-finalists-agree`.
- [x] Preserve one physical canonical schedule per family; policy outputs emitted from the same search may share that physical truth while carrying distinct scientific identities.
- [x] Bind each canonical policy source record to its exact registered scientific contract/fingerprint.
- [x] Make source-registration drift fail fast as `BLOCKED`; legacy-unbound/mismatched concrete artifacts may rebuild only under an unchanged registered source identity and may never be silently REUSED.
- [x] Make current Strategy Compare canonical/fixed-multiseed arms validate their strategy-param profile relationally from config/registry ownership rather than hard-coded C/P/seed/trial literals.
- [x] Keep `PARAM-P5 / C84` fixed-checkpoint validation completely unchanged and outside this round.
- [x] Preserve `PARAM-P6 / C85` Round-3 binding unchanged.

Exit: C58/C61/C62/C85 non-validation paths share one scientific-identity mechanism and can be regression-tested without mixing in validation redesign.

Round-4 evidence: current Min/Full canonical build contracts, P6 build contract/scientific fingerprint, and Strategy Compare runtime payload remain byte/value-equivalent to the input ZIP.  Registered fingerprints are `783eb920…428d` (Min/base-best), `77fa547f…0ad9` (Full/base-best), and `7f8d7c77…7d440` (Full/base-agree).  Full base-best/base-agree continue to share the same physical canonical schedule; identity is attached at policy-output source-record level.

### Round 5 — Non-validation stabilization / regression **[DONE]**

- [x] Run focused non-validation regression across C58/C61/C62/C85, canonical OOS/Rolling preparation, REUSE/BUILD/REBUILD/BLOCKED states, and relevant Robustness seed semantics.
- [x] Close the unsafe identity under-binding found by regression: both single-seed benchmark params and P6 replicated multi-seed robustness artifacts now bind a concrete scientific identity derived from the registered source profile plus the benchmark fitting override.
- [x] Make the versioned fitting protocol the owner of Strategy Parameter seed/trial semantics; generic Optimizer seed/trial defaults no longer mutate a registered strategy fitting identity.
- [x] Limit physical-schedule manifest comparison to schedule-owned fields; selector/score/gate scientific semantics remain owned by each registered policy-output scientific identity.
- [x] Verify execution/display-only mutations do not invalidate the registered scientific identity and scientific mutations fail fast on the registered source contract.
- [x] Verify historical evidence and physical truth reuse remain unchanged where fingerprints match.
- [x] Keep C84/P5 fixed-checkpoint validation untouched.

Exit: all non-validation strategy-parameter/compare paths are stable enough for user-side formal double-check before validation behavior changes.

Round-5 evidence: C58/C61/C62 single-seed benchmark artifacts and C85/P6 replicated multi-seed benchmark artifacts now persist derived scientific identity and follow `BUILD → REUSE`, while removing that binding returns `REBUILD` rather than unsafe reuse; registered source scientific drift is `BLOCKED`. Baseline-vs-Round-5 canonical Min/Full/P6 build contracts, registered fingerprints and current Strategy Compare/Robustness runtime payloads are byte/value-equivalent. C84/P5 validation profile/runtime payload hashes are identical. A conservative pre-existing fingerprint over-binding of some inactive/non-applicable selector review knobs was observed; changing it would alter already-registered fingerprints, so it is explicitly deferred to an identity-version migration rather than silently mixed into this behavior-preserving round.

### Round 6 — Validation redesign **[NEXT]**

- [ ] Preserve no-validation baseline as an immutable validation contract.
- [ ] Preserve fixed-checkpoint P5/C84 validation as historical/control semantics.
- [ ] Add plateau early-stop validation contract: max budget, warmup/min observations, patience, min delta, metric, direction, PIT-safe validation window.
- [ ] Share a generic early-stop state-machine abstraction with DL where valid; keep DL/non-DL evaluation computation separate.
- [ ] First validate single-seed behavior/cost, then extend to multi-seed only after the single-seed protocol is accepted.
- [ ] Produce a new `PARAM-P*`; never overwrite P6/P5 meaning.

Exit: validation budget is adaptive and versioned, with no OOS feedback leakage.

### Round 7 — Strategy component/composition identity + closure **[PENDING]**

- [ ] Define the permanent component/composition identity model through the canonical Experiment Registry before allocating any new prefix. Preserve existing meanings (`DL-*` source, `SR-C*` runtime/comparison arm, `PARAM-P*` parameter stage); provisional `RULE-*` / `STR-*` labels may be adopted only after explicit Registry allocation.
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
| 4 | DONE | No; non-validation identity convergence only |
| 5 | DONE | No; stabilization/regression only |
| 6 | NEXT | **Yes — explicit validation research change** |
| 7 | PENDING | No for identity refactor; new compositions later require explicit research identity |

## 5. Deletion gate for this temporary file

Delete this file only when all are true:

- [ ] All seven rounds are complete or explicitly closed by user decision.
- [ ] Every surviving rule has a permanent code/config/Registry/Architecture/test owner.
- [ ] No production/runtime module imports or parses this file.
- [ ] No formal/synthetic validator treats this file as a truth source.
- [ ] Historical experiment/artifact identities remain reproducible.
- [ ] Current production/Research flows pass local formal double-check from `apps/run_bundle.py` after the final patch.
