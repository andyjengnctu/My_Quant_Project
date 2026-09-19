# Trading／Research 生命週期的唯一責任歸屬

本文件說明程式責任與資料契約，不建立第二份策略規格。最高治理文件仍為 `doc/PROJECT_SETTINGS.md`；帳務、價格／tick、投組選擇與 Optimizer 參數真理維持既有 owner。本輪不變更 BQ 科學身分或正式報表契約。

## 規則與使用者決策權

策略參考方案、使用者確認的帳戶意圖、實際成交是三種不同資料。使用者可以在策略與帳戶資源上限內調降股數，也可以先選較低排名的合法候選，而不必先買較高排名候選。選擇順序是輸入；驗證、資金預留與計算規則不是 UI 自行實作的例外。

`core/entry_plans.py::accept_entry_quantity_decision` 持有股數接受規則。Research 沿用既有自動選擇政策；Trading 提交使用者的明確決策。自動同步不能把使用者股數改回 Scanner 建議，也不能放寬日期、tick、資金預留、禁止當日買賣或禁止盤中資金重用的既有約束。本輪沒有新增任意修改 frozen 策略規則的權限。

## 唯一 owner 與依賴方向

| 責任 | 唯一 owner | 使用者 |
| --- | --- | --- |
| 管理時序、出場優先序與成交日封存 | `core/position_management.py` | Research 執行 adapter、確認成交重播、Shadow |
| 建倉工廠與完整 Shadow 承接 | `core/entry_plans.py` | Research 成交、Trading 帳戶異動及重建 |
| 確認股數事件重播與下一交易日投影 | `core/position_replay.py` | 持股同步與單股查詢 |
| 真實建倉／事件正規化、出場義務讀模型 | `core/trading_position_projection.py` | 持股重播與查詢 |
| 成交前資訊邊界的 Shadow 重建 | `core/trading_lifecycle_plans.py` | 掛單成交、Scanner 直接成交、OMS 建倉 |
| frozen 參數的成交前指標與時間軸 | `core/trade_lifecycle.py` | 掛單同步與圖表讀模型 |
| 固定 finalized execution-consumer 脈絡 | `services/trading/lifecycle_context.py` | 共用生命週期操作及持股查詢 |
| 首次觀察訊號的持久化 binding | `services/trading/signal_lineage.py` | 每日 Scanner 工作流程 |
| 真實券商事實、hash chain、確認異動 | 既有 account／order／fill owners | 明確使用者／券商操作 |

Core 不得反向引用 services、apps、tools 或 UI。`core/position_step.py` 只負責 Research 模擬成交與共用帳務，不再另有 STOP／TP／trailing 執行迴圈。Trading adapter 提供驗證過的資料脈絡與確認事件，呼叫共用重播，不自行排列管理 primitive。相容匯出指向相同函式物件，不保留第二份演算法。

## 管理時序與成交事實

建倉承接 Shadow 管理狀態，但保留實際成交價、股數與成本。成交日記錄最高價與 STOP／TP 觸及，STOP 優先；不產生同日賣出。沿用既有 Research 語意：記錄成交日 high-water，不會立即以同一個已消耗的 high 再提高追蹤停損。

後續交易日先使用前一完成日資訊，再依唯一核心的順序判斷。策略決策不等於成交。Research 可確認模擬成交；Trading 重播只承接對應日期的真實股數事件。同日先確認的成交必須先影響剩餘股數，再產生後一個決策，因此半倉成交後的指標全出場不會重複使用原股數。

延後半倉義務不同於尚未觸及的 TP 限價單。`TP_HALF_DEFERRED_OPEN` 表示下一合法交易時點的市價義務，延後全出場使用 `STOP_EXIT_MARKET`。兩者仍需明確建立券商單並確認成交；觸及本身不會扣減真實庫存或現金。部分成交後保留尚未履行的股數。

重播分開輸出「執行流程使用的狀態」與「下一交易日管理投影」。後者由副本計算，不能回灌成當日可執行狀態，避免同步或圖表提前、重複消耗 high-water。

## 同源輸入與時間邊界

每日工作流程將首次觀察候選的原訊號日、代表 Params、成員證據及原始 execution-plan 存入 `state/trading/signal_lineages.json`。production Params 更新後，延續訊號仍由 canonical Scanner producer 使用原代表 Params 評估。暫時不在候選內不代表可以重新綁定；registry 不實作另一套 Scanner 或指標公式。

升級時可以使用有效的舊 Scanner snapshot 建立 binding，但這只代表已保存的觀察證據，不能證明更早、從未記錄的原始參數。歷史來源不足不能用 current Params 補造；binding 損壞時明確失敗，不默默重設。

Pending 與 Position 共用同一個固定的 finalized 日期及資料來源。來源身分包括資料版本、frozen binding、重播契約版本及已確認事件進度，不能只用日期判斷重用。相同輸入重跑不異動；同日資料更正或真實成交更正則重新重播。管理同步不可修改帳務或券商欄位。

補登歷史成交時，Shadow 只能使用成交前的資訊；較新的 Pending snapshot 不得洩漏到較早的建倉日。更正成交後使用 effective account events 的真實日期重建，不覆寫原掛單意圖或歷史事件鏈。

帳戶估值仍可沿用既有較新的已驗證市場邊界；這是估值契約，不是策略可以越過 execution consumer finalized 邊界的理由。

## 查詢、圖表與持久化

單股查詢使用完整建倉歷史及 frozen Params 產生共用管理投影，裁切後的圖表只負責日期／線段／標籤呈現，不能拿顯示視窗當完整策略歷史。虛擬投影不能建立真實 POSITION 或 BUY；舊版手動接管只有在明確 adoption 事件與券商建倉證據一致時才還原歷史持股。

既有 account、pending 與 OMS 仍各自持有其事實；衍生管理有受限的可寫欄位。沒有新增第二個券商帳本、事件平台或微服務。尚未存在的可選 OMS 使用 canonical empty-state schema，不使用缺欄位的臨時字典。

## 回歸與部署限制

`tools/validate/synthetic_lifecycle_ssot_cases.py` 包含管理 owner、指標／成交前來源、持久化同步、延後與部分成交、人工裁量／更正、首次訊號、同日成交順序及固定亂數重播的能力測試。它們登錄在既有 synthetic registry 與 machine-readable checklist，不另建平行永久 gate。

依賴檢查與寫入契約補足結果對照，但不代表所有未來修改都不可能出错，也不能防止刻意繞過。獨立模擬使用暫存帳戶／OMS 檔案與合成市場儲存 fixture，實際執行策略計算與持久化；不是實際市場 Parquet、GUI、券商或本機正式整合測試。`apps/run_bundle.py` 仍是使用者本機正式驗證入口。

重播契約變更後首次同步可能需要重建持股歷史，後續相同來源則依 fingerprint 重用。完整歷史查詢優先確保 frozen 指標正確，不採不安全的裁切視窗捷徑；本輪未宣稱實際大型帳戶的延遲或 UI 效能已完成量測。

## Confirmed-history chart ownership and recovery

- `collect_confirmed_position_cycles` partitions effective corrected account facts by acquisition. Each historical or current managed cycle uses its own frozen binding and the common position replay; a later acquisition cannot supply geometry to an older one.
- Inspector construction reuses the already verified full `clean_df` together with its pinned `consumer_state`, for both fresh analysis and analysis-cache hits. It does not independently change source generations. Visible/cropped candles are a rendering window, never a new management origin.
- Legacy immutable BUY snapshots preserve acquisition evidence, but matching fill date/price/quantity alone does not prove compatibility with the pre-fill management origin. A redundant source-read failure must not blanket-clear verified management geometry. Complete frozen-input replay can recover it; otherwise retained persisted evidence is marked pending synchronization, not silently asserted current. Missing binding is never replaced with today's Params.
- Confirmed inventory consumes pre-existing prefill intent. Signals dated inside an actual holding interval (including full-exit day) cannot reappear as SHADOW after exit. A genuine signal strictly after the full exit remains valid, and unrelated unfilled history remains visible. The same exclusion governs timelines, direct sidebar resolution, annotations and future preview.
- A direct backfill's later registration/information date is not a new buy signal. A prefill plan must predate its confirmed entry. Actual execution dates and reasons come from corrected account facts; a manual sell must not be inferred as an indicator sell.
- The full-exit bar retains its management geometry, while post-fill inventory is zero and the next bar does not extend the old cycle. Adjacent acquisitions have separate line segments. Research historical metrics remain independent and unchanged.


## Direct-fill acquisition provenance

`services/trading/account_trade_entry.py` resolves an unselected/typed ticker against the canonical account-aware Scanner snapshot. A matching candidate goes through the same freshness and exact-reference guard as an explicitly selected candidate and retains that candidate's frozen strategy lineage. A ticker merely present in the scanned universe, but not a candidate, is not classified as strategy. A stale/corrupt match is rejected rather than silently converted to custom. Existing pending intent must be filled through its owning pending entry, not replaced by a new acquisition origin.

The confirmation dialog and submitted command share the resolved candidate and route. A source change between preview and commit requires confirmation again. User-reduced quantity remains distinct from the strategy reference quantity. Corrected transactions retain their acquisition source. Account sell-detail source follows the acquired position; `execution_reason` separately identifies a manual sell, so a strategy-origin holding can be manually sold without becoming custom. No historical manual record is relabelled from today's candidate membership in the absence of original provenance.

The existing capability catalog now includes `validate_trading_direct_fill_source_contract_case`: actual persisted candidate/account records, source validation, direct/clicked equivalence, frozen/current Params separation, manual reductions, corrections, sell source/reason independence, stale guards and the real GUI confirmation handler. External market and runtime providers use isolated test fixtures; this is not a claim about a live broker or user account.


## Frozen geometry availability and historical order dates

`core/trading_lifecycle_plans.py` owns the temporal input contract.
`core/prefill_origin.py::materialize_prefill_origin` is the unique frame-bound
producer used by pending synchronization, confirmed acquisition and the chart.
A user order date, original signal identity, saved snapshot date and completed
market-information cutoff are different facts. Merely relabelling a later
Stop/Target/ATR as an earlier value is always illegal.

An available immutable first-signal seed remains authoritative, with unchanged
numeric results. A later saved seed alone is not proof of earlier geometry;
neither is it proof that the historical management plan cannot be computed.
When the original frozen policy and its canonical origin OHLCV are available,
the producer explicitly materializes a historical plan using that exact policy
and ONLY the completed prefix through the origin. It calls the existing signal,
ATR and entry-plan producers, never carries later Stop/Target/ATR backwards, and
never substitutes current Params. Strategy reconstruction requires the original
canonical signal/limit; manual reconstruction uses the original accepted order
limit and origin-date ATR. Later user choices belong to their decision history.

Reconstructed rows carry `origin_resolution`, the frozen policy identity and the
recorded snapshot date. They are a deterministic replay under the persisted
policy, not evidence that an unrecorded live snapshot or earlier Optimizer fit
existed. Provider representation/finalized legality remains the responsibility
of the pinned canonical market source. Missing frozen policy, origin bar or ATR
warmup cannot be filled with today's data. Corrupted identity fails explicitly;
unavailable prerequisites retain dated evidence and independent pending facts.

New historical manual orders construct their immutable management seed at the
requested completed information day, not the current registration date. This
prevents future records from creating the same missing-prefix defect. Existing
account events, order dates and frozen Params are not rewritten by replay.

A mature saved Shadow is consumed only at its own completed information bar.
Missing earlier history is reconstructed separately; forward execution first validates its management state against the canonical
prefix. Compatible checkpoints resume unchanged without consuming their bar
twice; incompatible derived checkpoints are recomputed from the same frozen
origin and labelled in the read model, never silently spliced into it.
A live checkpoint conflicting with an earlier canonical termination is rejected,
not used to resurrect an invalid lineage. Fill/cancel boundaries apply to all
states. Acquisition consumes only completed bars strictly before the actual
fill. STOP/TP completion uses the continuation barrier owner already used by
Research, including a completed half-sale or deferred first-day obligation.

Quantity, date and price decisions do not create a new management origin. The
original seed remains independently preserved; new choices use their own
`order_decision_date`. Synchronization fingerprints include the origin contract,
market identity, frozen source and accepted decision history. Identical input is
idempotent, while old cached success is not evidence for the revised contract.

The existing registered prefill-provenance capability covers original geometry,
late-seed and suffix poisoning, frozen/current parameter separation, sufficient
history, real strategy signals, manual order creation, all four Shadow lines,
confirmed acquisition inheritance, actual sell boundaries, checkpoint conflict,
shared intent projection, edit safety and non-writing repeat synchronization.
No parallel incident gate or new Research metric is introduced.


## Confirmed pending intent and pre-fill history

A management availability boundary is not an order-existence boundary.
`services/trading/pending_entry_state.py::project_trading_pending_intent_entries`
reads the validated pending journal once per consumer operation.
`core/trading_lifecycle_plans.py::build_pending_order_intent` dates its accepted
limit, quantity and reservation. Synchronization, fill and chart consume the same
projection; the renderer has no second journal parser. Its first historical
registration is an explicit user order fact; subsequent changed choices are
available only from their own decision information date. A daily synchronization
snapshot with unchanged accepted values is not a new decision. It does not
replace an earlier accepted choice or move the original order annotation.
The sidebar and limit line use the same as-of accepted choice, including when
management is already SIGNAL/SHADOW. Management-plan reference values remain
separate metadata; accepting an order does not overwrite its Shadow state.

`core/trade_lifecycle.py::overlay_confirmed_order_intent` is the sole owner of the
join between those facts and the existing management timeline. A confirmed order
without lawful management evidence is `PENDING`, not `SHADOW`, `POSITION` or
no-trade. The order day has no active transaction lines. From the next completed
bar its proven buy limit can be drawn, while Stop/Target/Shadow entry remain
absent until their lawful source is available. Missing frozen Params cannot
create geometry from current Params, but cannot erase independently confirmed
order facts either. A verifiable earlier seed retains the complete canonical
Shadow history and its unchanged numeric results. A recoverable late seed goes
through the shared frozen-policy materializer first; it must not be treated as
permanently unavailable merely because the saved snapshot is newer.

A confirmed pending interval takes precedence over unrelated hypothetical
Research signals; only corrected account BUY events create POSITION. Historical
filled orders retain their pre-buy pending interval. Fill/cancel boundaries
close intent even when a management seed is dated later; a later seed cannot
revive the consumed signal after a real exit. A closed broker audit with a voided
fill is not an outstanding order. Genuine strategy termination remains attached
to its actual termination date and remains a synchronization error even if the
unexecuted order still exists. Keeping an order visible never declares its
strategy valid, reduces confirmed inventory or releases reserved cash.

The inspector supplies the validated journal and pinned frame to the common
projection. Its sidebar, transaction lines and future preview consume the same
rows; the renderer cannot infer another timeline. The existing registered
prefill-provenance capability covers both axes, persisted legacy fill/close,
read-only replay, real early geometry, edit history and competing hypothetical
signals. An earlier assertion that expected no historical row before geometry
availability has been corrected: only unproven management must be absent, not
the real order. No parallel incident gate or new Research metric is introduced.


## Replay cost and read-model persistence

Historical origin recovery adds a canonical signal/ATR computation on the
origin prefix. The subsequent complete-frame indicator calculation is cached
within the existing lifecycle operation by parameter and frame identity; no
new persistent market or policy truth is created. Pending synchronization still
short-circuits on its complete fingerprint. Large-account latency has not been
benchmarked; correctness takes precedence over reusing an unproven late seed.
Accepted-decision history is a compact projection, never persisted recursively
inside new pending snapshots. Historical journal events remain unchanged.


## Accepted reservation versus management synchronization

`accept_entry_quantity_decision` remains the only acceptance owner for NEW or
explicitly AMENDED quantities. Once accepted, the order limit, quantity, cash
reservation, saved approval ceiling and user override are decision facts, not
outputs of daily management replay. A changed reconstructed/trailing stop or a
changed current sizing capital must not retroactively rerun that acceptance.

`validate_accepted_entry_reservation` validates an existing accepted plan against
its saved ceiling (when present), canonical exact buy ledger and currently free
cash after all other active reservations. It does not compute another strategy
quantity. Legacy accepted snapshots without an optional ceiling are not assigned
a newly invented historical ceiling. Inconsistent quantity/limit/reservation
copies are rejected, not normalized silently.

The Pending sync adapter may update only derived management geometry, its
checkpoint and evaluation identity. It never changes accepted economics or
approval metadata. Resource integrity/coverage is checked BEFORE a management
cache hit: a same-day cash correction cannot hide behind a previously LATEST
status. Cash-only changes do not force market replay when management identity
is unchanged. Genuine STOP/TP termination, missing/corrupt lineage or uncovered
reservations still fail without cancelling, shrinking or releasing the order.

These boundaries are exercised in the existing persistent-lifecycle capability:
accepted-versus-new sizing divergence, exact reservation, sibling locks, cache
hits after cash correction, explicit amendments and preserved Shadow history.
No extra permanent gate or separate Research sizing policy is introduced.


## Display precision is not lifecycle input

The Inspector carries its already validated full market frame as an ephemeral
`prefill_market_frame` reference in its read model. Pending/Shadow replay consumes
this frame, not float32 display-candle arrays. Generated rows are mapped to the
visible chart by date, so display precision and cropping cannot change a stop,
target or management origin. POSITION already consumes the same pinned frame.
The reference is not persisted or loaded as an independent market-data source;
legacy standalone callers without it retain their explicit chart-frame fallback.
Future-dated input is rejected even when the account has no confirmed position.


## Confirmed acquisition upgrade and origin compatibility

`core/trading_position_projection.py::build_confirmed_position_origin` owns the
confirmed acquisition handoff for live synchronization and historical inspection.
For a pending/strategy lifecycle, it resolves the same frozen pre-fill entry plan
as a new confirmation, using completed bars strictly before the actual fill.
An existing `entry_position_state` or immutable BUY snapshot cannot bypass that
resolution merely because its broker date, price and quantity still match.
A valid original dated seed and compatible management state retain their values;
a late/invalid derived snapshot is not an independent strategy authority.

`services/trading/pending_entry_links.py::resolve_position_pending_entry` is the
shared evidence join for both consumers. It delegates to the established
original-BUY/correction matcher, including legacy manual-order ambiguity. An
unrelated same-ticker order or a later acquisition is not a permissible source.
Manual direct acquisitions without an order or Scanner signal keep their existing
acquisition contract; no fictitious pre-fill interval is created.

The replay contract includes acquisition-origin version, pre-fill-origin version,
full frozen binding, accepted original order evidence, market context and actual
confirmations. Previously synchronized legacy states must therefore be reevaluated
once even on the same finalized date. Identical subsequent input is idempotent.
Synchronization changes only allowed management fields and records
`acquisition_replay` hashes in the management state and its appended audit event.
Original entry seeds/states, frozen Params, past journal records, broker facts,
accepted reservations and cash are not rewritten. Recorded acquisition economics
and original risk evidence are not retroactively replaced by a management-cache
repair. Actual fill corrections continue to use their effective broker inputs.

The existing pre-fill-provenance capability now covers an already-persisted late
seed, a valid timely seed, a new acquisition, historical/closed cycles, partial
manual sells, read-only inspection, management-only synchronization, checkpoint
compatibility, cache invalidation and repeatability. Positive post-fill parity is
not a substitute for checking both sides of the actual acquisition boundary.
