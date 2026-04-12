# GPT 交付前檢查表

用途：assistant 每輪交付前操作檢查表；不作本地端 formal test 主表、不記 `B` / `T` / `G` / `E` 狀態，也不取代 `PROJECT_SETTINGS.md` 與 `TEST_SUITE_CHECKLIST.md`；本地端 formal test、`apps/test_suite.py`、synthetic registry 與 bundle 檢查均不驗證本檔內容。

文件分工：`PROJECT_SETTINGS.md` 管原則與邊界；`TEST_SUITE_CHECKLIST.md` 管本地端 formal test suite 收斂與維護；`GPT_DELIVERY_CHECKLIST.md` 只管 GPT 交付前操作檢查。

## A. 開始前

1. 確認本輪最新基準，並回報當前工作基準、ZIP 檔名、SHA256、全新解壓目錄與已讀文件。
2. 先檢查是否存在尚未列入 `doc/TEST_SUITE_CHECKLIST.md`、但應由正式入口涵蓋的缺口。
3. 若確認沒有新增 formal coverage gap，交付時明確回報正式入口已涵蓋目前需求。

## B. Formal test chain 變更時

1. 對同一 target 逐項自檢 definition、import、registry、`doc/TEST_SUITE_CHECKLIST.md`、parser、guard、正式入口摘要、help 與對應 meta guard 是否已實際同步。
2. 追蹤 ID 或 canonical 主題更名時，須同步檢查所有引用層，不得只改局部別名、摘要或相鄰文件。
3. 若修改 `doc/TEST_SUITE_CHECKLIST.md` 主表、`T`、`G`、`E` 等機械排序區塊，交付前必須整段重排並對照既有排序 guard。
4. 若本輪新增 `G` 紀錄，交付前必須重新抽出整個對應日期區塊，依 formal tracking ID sort key（prefix / numeric / suffix）穩定排序後整段覆寫回原位；不得以人工目測、字典序、尾端追加或局部插入取代正式排序。
5. 若本輪動到 `G` 區，交付前還必須再對整個 `G` 表執行一次日期 / tracking ID 全表 guard 檢查；不得只檢當前日期區塊。
6. 若本輪修改 `doc/PROJECT_SETTINGS.md`、`doc/ARCHITECTURE.md`、`doc/CMD.md` 或 `apps/test_suite.py --help`，交付前必須清掉未指明檔名的清單文件用詞。
7. 若本輪修改任何受 `tools/validate/synthetic_meta_cases.py` 驗證的文字面，交付前必須反查對應 expected literal；不得只改正文後遺漏 meta contract。
8. `doc/ARCHITECTURE.md`、`doc/CMD.md` 與 `apps/test_suite.py --help` 只保留穩定主題摘要；不得把高波動 helper 長清單、暫時演進敘事或完整 validator 枚舉重新灌回這三個面。
9. 若本輪修改上述三個面，交付前必須確認逐字比對只剩 canonical 名稱、正式入口、section heading 與最小必要 fragment；不得把高波動描述做成 exact-string contract。
10. 若本輪修改 `apps/test_suite.py --help` 或對應 help meta contract，交付前必須確認 help 只保留穩定 theme token，不再逐項列舉 validator / exact-contract 名稱，且涉及 TEST_SUITE_CHECKLIST 時必須顯式指名，不得回退成 bare `checklist` 用詞。
11. 若本輪修改 `doc/TEST_SUITE_CHECKLIST.md` 主表項目、`DONE` 摘要或索引式摘要縮句，且對應 meta contract 仍要求最小必要 literal / theme token，交付前必須反查對應 validator 的 required fragment 與長度上限；不得只縮字數而漏掉穩定必需詞。
12. 若本輪修改 help theme token 類 validator，交付前不得再用整句人類說明、長前綴或完整 help 文案作定位錨點；必須改以穩定前綴、結構位置或最小必要 token 抽取目標列，避免 theme-token contract 表面瘦身、內部仍綁高波動 exact-string。
13. 若本輪進行文件瘦身，不得移除既有 formal contract 仍明確要求的最小必要 fragment；至少必須反查 `tools/validate/meta_contracts.py` 與 `tools/validate/synthetic_meta_cases.py` 中既有 single-entry 與 shipped module fragment contract 是否仍被文件保留。
14. 除使用者明確要求註解清理，或該註解／docstring 會被 parser、`--help`、UI、report、export、bundle 或 formal contract 直接讀取／輸出外，一般註解不納入 GPT 交付前最嚴格檢查與交付阻塞。
15. 若本輪將既有 `validate_*` 契約改列 `N/A` 或改成 compatibility stub，交付前必須反查 meta-registry completeness guard、defined/imported validator set、`done/unfinished` 摘要與 `doc/TEST_SUITE_CHECKLIST.md` parser。
16. 若本輪問題屬既有 Bxx / Txx / validator contract 鏈，交付前必須先建立同源 / 同鏈 / 同契約收斂清單；至少列出鏈根、掃描範圍、逐項結果與未清阻塞。
17. 對同一 validator function 或同一 impacted_modules 集合內的相鄰缺口，必須一次掃完；不得只修第一個命中項就交付。
18. 若本輪調整文件分工、角色邊界，或某文件是否納入本地 formal 驗證的規則，交付前必須同步檢查 `doc/ARCHITECTURE.md` 與 `doc/CMD.md` 是否仍沿用舊邊界。
19. 若本輪修改 `doc/ARCHITECTURE.md` 的 shipped 模組索引，交付前必須逐一對照實際檔案與文件索引一致，並確認檔名與註解分隔符未黏連成錯誤 path token。

## C. Bundle 修復時

1. 逐條對照 bundle 原始失敗項完成閉環；不得以相鄰文件、註解、help 或 `doc/TEST_SUITE_CHECKLIST.md` 已同步視為修復完成。
2. 交付前必須逐條確認 bundle 原始失敗項已消失。
3. 若前一輪修改在本輪仍被 bundle 或再檢查證明有錯，除修正原始失敗外，必須同步更新本檔，將防再犯要求上提為可泛化、可操作的交付前檢查。

## D. 交付前

1. 依本輪最嚴格檢查結果，一次找出並修正所有目前可發現問題；不得將同源、同鏈或同契約的已知相鄰問題拆成多輪逐步釋出。
2. 若仍有無法同輪清除的阻塞，必須明確揭露；不得以局部修補視為完成。
3. 完成 GPT 端最嚴格檢查後，僅在確認無已知問題時交付 patch / ZIP，並列出本輪修改檔案與必要替換方式。
