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
3. 若本輪將 validator / contract 的語意從 exact-contract、逐項列舉或舊規則轉為穩定主題 token、結構契約或新 canonical 邊界，必須同步更新函式名、case_id、registry、`doc/TEST_SUITE_CHECKLIST.md` 摘要與收斂描述；不得讓 active 名稱持續誤標實際語意。
4. 若本輪修改 `doc/TEST_SUITE_CHECKLIST.md` 的主表、`T`、`G`、`E` 等機械真理區，交付前必須整表核對排序、摘要、最新狀態與 transition 連續性；不得只補單一列、單一 summary 或單一日期區塊。
5. 若本輪修改任何受 formal validator 驗證的文字面，交付前必須反查對應 required fragment、required token、長度上限與排除詞；validator 若宣稱只驗穩定 token / 結構，內部也不得再依賴高波動全文、長前綴或完整文案作定位錨點。
6. 若 bundle 或再檢查出現 shared helper / utility / path helper 類 `NameError`、`AttributeError` 或匯入失敗，交付前必須回到失敗模組逐一核對實際使用的共享符號、from-import / module import 與 alias；不得只修 summary、registry 或相鄰文件而不補回缺失依賴。
7. 若本輪修改正式 summary / manifest / artifact schema，交付前必須同步更新 synthetic contract fixture、required keys / tokens 與 stale-key 排除檢查；不得只改正式輸出實作，卻讓 contract case sample payload 或 nested schema 仍停在舊欄位。
8. `doc/ARCHITECTURE.md`、`doc/CMD.md` 與 `apps/test_suite.py --help` 只保留穩定主題摘要；不得把高波動 helper 長清單、暫時演進敘事、完整 validator 枚舉或局部事故修補語句重新灌回這三個面。
9. 若本輪進行文件瘦身或重組，交付前必須逐一反查對應 validator 要求的全部最小必要 fragment / entry；不得因縮句、合併或改寫，只保留部分同主題片段。
10. 涉及多份治理文件或角色邊界的敘述時，必須使用顯式檔名；不得回退成模糊代稱或未指名的 bare `checklist` 用詞。
11. `PROJECT_SETTINGS.md` 為上位原則來源；formal test suite、validator 與 `doc/TEST_SUITE_CHECKLIST.md` 只可記錄 coverage mapping 與下游落實，不得反向驗證 `PROJECT_SETTINGS.md` 的 wording、摘要格式或固定字串。
12. 除使用者明確要求註解清理，或該註解／docstring 會被 parser、`--help`、UI、report、export、bundle 或 formal contract 直接讀取／輸出外，一般註解不納入 GPT 交付前最嚴格檢查與交付阻塞。
13. 若本輪將既有 formal contract 改列 `N/A`、compatibility stub 或其他非正式長期路徑，交付前必須同步檢查 registry completeness、defined/imported validator set、`doc/TEST_SUITE_CHECKLIST.md` parser 與 `done/unfinished` 摘要。

## C. Bundle 與收斂

1. 若使用者提供 bundle，必須逐條對照 bundle 原始失敗項完成閉環；不得以相鄰文件、註解、help 或 `doc/TEST_SUITE_CHECKLIST.md` 已同步視為修復完成。
2. 若問題屬既有 Bxx / Txx / validator contract 鏈，交付前必須建立同源 / 同鏈 / 同契約收斂清單；至少列出鏈根、掃描範圍、逐項結果與未清阻塞，並一次清空相鄰缺口。
3. 若前一輪修改在本輪仍被 bundle 或再檢查證明有錯，除修正原始失敗外，必須同步把防再犯要求上提為可泛化、可操作的交付前規則；不得再補單一案例型條款。

## D. 交付前

1. 依本輪最嚴格檢查結果，一次找出並修正所有目前可發現問題；不得將同源、同鏈或同契約的已知相鄰問題拆成多輪逐步釋出。
2. 若仍有無法同輪清除的阻塞，必須明確揭露；不得以局部修補視為完成。
3. 完成 GPT 端最嚴格檢查後，僅在確認無已知問題時交付 patch / ZIP，並列出本輪修改檔案與必要替換方式。