# GPT 交付前檢查表

用途：assistant 每輪交付前操作檢查表；不作本地端 formal test 主表、不記 `B` / `T` / `G` / `E` 狀態，也不取代 `PROJECT_SETTINGS.md` 與 `TEST_SUITE_CHECKLIST.md`；本地端 formal test、`apps/test_suite.py`、synthetic registry 與 bundle 檢查均不驗證本檔內容。

文件分工：`PROJECT_SETTINGS.md` 管原則與邊界；`TEST_SUITE_CHECKLIST.md` 管本地端 formal test suite 收斂與維護；`GPT_DELIVERY_CHECKLIST.md` 只管 GPT 交付前操作檢查。

原則：本檔只保留高層、可泛化、可操作的交付 guard；不得長期累積只對單次事故、單一句子、單一 wording 或局部實作特例有效的補丁型條款。

## A. 開始前

1. 確認本輪最新基準，並回報當前工作基準、ZIP 檔名、SHA256、全新解壓目錄與已讀文件。
2. 先檢查是否存在尚未列入 `doc/TEST_SUITE_CHECKLIST.md`、但應由正式入口涵蓋的 formal coverage gap。
3. 若確認沒有新增 formal coverage gap，交付時明確回報正式入口已涵蓋目前需求。

## B. 正式鏈與文件契約

1. 若本輪修改 formal test chain，同一 target 必須整鏈核對 definition、import、registry、`doc/TEST_SUITE_CHECKLIST.md`、parser、guard、正式入口摘要、help 與對應 meta guard；任一層未同步，不得交付。
2. 若本輪修改 canonical 名稱、追蹤 ID、validator 名稱、正式入口名稱、共享 schema，或將 validator / contract 的實際被測來源改指向另一份文件或模組，必須同步檢查所有引用層並改正殘留舊來源名稱；不得只改函式本體，卻保留舊 case_id、T 項摘要、help theme 或 registry 描述。
3. 若本輪將 formal-facing 文字從舊列舉改為穩定主題、結構契約或新 canonical 邊界，必須同步更新正式入口、validator、registry 與 `doc/TEST_SUITE_CHECKLIST.md` 的正式語意；但同步範圍只限 shipped 正式介面與其摘要，不再把 validator 內部命名、sub-check 文字或排除片段長短上升為新的 formal 規則。
4. 負向檢查只針對 shipped 正式輸出、`--help`、schema、檔案樹與正式 payload 本體；不要再為 validator 函式名、sub-check 名稱、literal 寫法或相鄰檢查器的字串 hygiene 追加新的 formal 要求。
5. 若本輪修改 `doc/TEST_SUITE_CHECKLIST.md` 的主表、`T`、`G`、`E` 等機械真理區，交付前必須整表核對排序、摘要、最新狀態與 transition 連續性；不得只補單一列、單一 summary 或單一日期區塊。若新增列插入既有日期區塊，且 tracking ID 排序鍵小於當前尾列，必須抽出整個同日區塊重排後整段覆寫回原位，不得把較小 ID 直接追加在較大 ID 後。若是回補同日歷史收斂列或把較小 ID 插回較晚位置，仍須從該日期首列到末列做一次機械 monotonicity 掃描；不得只目視新插入點附近幾列後就交付。
6. 若本輪修改任何受 formal validator 驗證的文字面，交付前只需反查對應最小必要 fragment / token 與直接被測輸出；不要再對排除詞的最短片段、exemplar 完整性、sibling 去重或 validator 內部措辭建立第二層檢查規則。
6a. 若 formal contract 名義上驗 CLI / `--help` / report / export / payload 輸出，交付前必須確認 validator 直接讀取該 shipped 輸出本體或其最接近的正式產物；不得改成掃 source 內 `print(...)` 字面、局部 string literal 或其他實作內文來冒充輸出驗證。
7. 若 bundle 或再檢查出現 shared helper / utility / path helper 類 `NameError`、`AttributeError` 或匯入失敗，交付前必須回到失敗模組逐一核對實際使用的共享符號、from-import / module import 與 alias；不得只修 summary、registry 或相鄰文件而不補回缺失依賴。
8. 若本輪修改正式 summary / manifest / artifact schema，交付前必須同步更新 synthetic contract fixture、required keys / tokens 與 stale-key 排除檢查；只核對正式輸出與 contract 本體，不再為 active sub-check 命名或相鄰 validator 的文字 hygiene 追加第二層規則。
9. `doc/ARCHITECTURE.md`、`doc/CMD.md` 與 `apps/test_suite.py --help` 只保留穩定主題摘要；不得把高波動 helper / support 長清單、暫時演進敘事、完整 validator 枚舉或局部事故修補語句重新灌回這三個面，也不要把 internal helper / support exact file-tree 列舉升格為長期 formal 契約。
10. 若本輪進行文件瘦身或重組，交付前必須逐一反查對應 validator 要求的全部最小必要 fragment / entry；不得因縮句、合併或改寫，只保留部分同主題片段。
11. 涉及多份治理文件或角色邊界的敘述時，必須使用顯式檔名；不得回退成模糊代稱或未指名的 bare `checklist` 用詞。
12. 若 validator 要排除 bare term，但正式文字允許 canonical qualified 版本，交付前應先白名單 qualified canonical token，再檢查剩餘 bare term；此要求只約束正式輸出本體，不延伸到 validator 命名或補丁式文字 hygiene。
13. `PROJECT_SETTINGS.md` 與 `GPT_DELIVERY_CHECKLIST.md` 只供 GPT 讀取與執行；formal test suite、validator、synthetic registry、bundle 檢查、`doc/TEST_SUITE_CHECKLIST.md`、`doc/ARCHITECTURE.md`、`doc/CMD.md` 與 `apps/test_suite.py --help` 不得引用、映射、驗證或假設這兩份文件的存在；三者之間的關連只可由 GPT 透過實作、formal checklist 與交付輸出落實。
14. 若本輪修改 formal-facing 文件或正式入口摘要，交付前必須確認其中不再直接提及 `PROJECT_SETTINGS.md` 或 `GPT_DELIVERY_CHECKLIST.md`；formal 世界只承接自身主表、架構、操作與輸出，不顯式回指 GPT 控制面。
15. 若 formal-facing 文件或正式入口需要表達治理 / guardrail 語意，使用中性主題名稱即可；不要再為避免單一句型或單一來源名稱而層層追加新的 formal 檢查。
15a. 正式入口 `--help` 若需摘要 formal 覆蓋範圍，只保留對使用者仍有意義的穩定主題；不要把 checklist heading、唯一性檢查、治理 guard 名稱或其他內部 meta 細節升格為 help 必備 token。
15b. 若 formal-facing 文件、`doc/CMD.md`、`doc/ARCHITECTURE.md`、report 或 retention 說明提到輸出目錄，必須使用完整 canonical `outputs/<category>/` 路徑；不得把 bare 類別名、legacy 目錄尾名或省略 `outputs/` 的片段當成正式路徑。
16. 除使用者明確要求註解清理，或該註解／docstring 會被 parser、`--help`、UI、report、export、bundle 或 formal contract 直接讀取／輸出外，一般註解不納入 GPT 交付前最嚴格檢查與交付阻塞。
17. 若本輪將既有 formal contract 改列 `N/A`、compatibility stub 或其他非正式長期路徑，交付前必須同步檢查 registry completeness、defined/imported validator set、`doc/TEST_SUITE_CHECKLIST.md` parser 與 `done/unfinished` 摘要。
18. 若本輪調整 checklist 治理規則，交付前只需確認 `G` 收斂紀錄仍可供人工閱讀，且不會破壞機械排序、狀態轉移或摘要同步；不要再把僅屬收斂備註欄 hygiene 的文字細節上升為新的 formal blocker。

## C. Bundle 與收斂

1. 若使用者提供 bundle，必須逐條對照 bundle 原始失敗項完成閉環；不得以相鄰文件、註解、help 或 `doc/TEST_SUITE_CHECKLIST.md` 已同步視為修復完成。
2. 若問題屬既有 Bxx / Txx / validator contract 鏈，交付前必須建立同源 / 同鏈 / 同契約收斂清單；至少列出鏈根、掃描範圍、逐項結果與未清阻塞，並一次清空相鄰缺口。
3. 若前一輪修改在本輪仍被 bundle 或再檢查證明有錯，除修正原始失敗外，必須同步把防再犯要求上提為可泛化、可操作的交付前規則；不得再補單一案例型條款。

## D. 交付前

1. 依本輪最嚴格檢查結果，一次找出並修正所有目前可發現問題；不得將同源、同鏈或同契約的已知相鄰問題拆成多輪逐步釋出。
2. 若仍有無法同輪清除的阻塞，必須明確揭露；不得以局部修補視為完成。
3. 完成 GPT 端最嚴格檢查後，僅在確認無已知問題時交付 patch / ZIP，並列出本輪修改檔案與必要替換方式。