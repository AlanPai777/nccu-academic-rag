# Agentic RAG 遷移計畫：把`proto3_langgraph.py`的主邏輯換成`agentic_main.py`架構

**日期**：2026-08-31
**性質**：正式遷移計畫文件——不是取代（不是兩套系統擇一留下、刪掉另一套），是把production `rag/proto3_langgraph.py` + `rag/nodes/`的主邏輯，逐步換成這個session（Step 13/clean pipeline系列）在`rag/agentic_main.py`裡驗證過的做法。文件性質比照當初`kind-greeting-patterson.md`那份原始plan——逐步驟、每步都要有驗證方式、每步都是一個獨立git checkpoint，不是一次性大改。

**方法論依據**：`rag/eval.py`的判分邏輯純粹讀答案文字，跟走哪條架構無關，所以兩邊可以直接用同一套criteria公平比較——這是這次遷移敢用「逐步替換、每步驗證」方式進行的前提，不需要為了比較而發明新的測試框架。

---

## 0. 已定案的決策（遷移前先討論過，不在遷移過程中重新開放）

### 0.1 `self_eval`的retry次數上限——採用agentic版本，不採用production的`max 2 retries`硬上限

production的`self_eval_node`有明確`_MAX_SELF_EVAL_RETRIES = 2`；`agentic_main.py`的`self_eval_node`沒有專屬retry計數器，靠`_SELF_EVAL_MAX_TURN=20`（risk-only天花板，不是正確性判斷）＋doom-loop偵測（連續3輪相同tool_calls簽章）當唯一安全網。**決定：遷移後維持agentic版本的無明確上限設計**，這是跟這個session一路「不要用deterministic上限迴避LLM決策」的實驗性精神一致的刻意選擇，不是遺漏保護機制。

### 0.2 `Send`不能直接用於agentic_rag的即時多輪迴圈——已用程式碼證實原因

讀`rag/nodes/state.py`確認：production `AgentState`裡所有會被`Send`分支寫入的欄位（`context_pages`/`sources`/`sub_query_results`）全部標了`Annotated[..., operator.add]`，是純累加器；production的`sub_query_node`（`Send`分支）走的是**單次**router→retrieval→office_lookup→extraction，不是被多個graph node反覆遞增的迴圈狀態。

`agentic_main.py`完全不是這個形狀——`turn`/`rewritten`/`stuck_turns`是`rewrite_node`/`agent_node`等**多個獨立graph node**在同一分支內反覆讀寫的即時迴圈狀態，不是累加器。這正是`scratchpad/spike_send.py`當時實測出`InvalidUpdateError`的原因：兩個`Send`分支同時對同一個無reducer欄位（`turn`）寫不同值，直接報錯。

**結論（非「Send完全不能用」，是「Send不能直接指向agentic_rag這種即時多輪迴圈的node」）**：`scratchpad/spike_nested_invoke.py`驗證過的可行模式是`Send`分支指向的node內部自己`.invoke()`一個獨立編譯的子圖，分支內部狀態完全隔離，只有`messages`（有reducer）跨界合併回外層。`agentic_main.py`現有的`multi_sub_query_node`（v1序列`for`迴圈＋`_build_loop_graph().invoke()`）已經是這個模式的雛形。**遷移時production `decomposition.py`現有的`Send`平行用法不能直接照搬進新架構**——如果KNOWLEDGE變成即時多輪迴圈，production現在對`sub_query_node`的直接`Send`會踩到同樣的`InvalidUpdateError`。這點在Part 5 Step 6明確處理。

### 0.3 Branch策略與entry point命名——已定案（2026-08-31）

**查證修正**：原本以為`main`已經有production的`proto3_langgraph.py`，直接查`git ls-tree main`後發現是錯的——**`main`目前只有classic RAG的檔案**（`main.py`/`pipeline.py`/`retriever.py`等），完全沒有`proto3_langgraph.py`、沒有`rag/nodes/`、也沒有`agentic_main.py`。這兩套agentic系統（production的proto3＋這個session重建的`agentic_main.py`）**全部都只存在於`feature/agentic-rag-proto`這一條分支**，`main`目前的agentic RAG部分是空的。

**Branch策略（已建立）**：從`feature/agentic-rag-proto`切出`migration/proto3-to-agentic`（已執行：`git checkout -b migration/proto3-to-agentic`），遷移工作在這條分支上進行。**採單一分支＋乾淨獨立commit，不是stacked branch**——Part 5的每個Step對應至少一個commit（granularity判準是「這個commit單獨看有沒有意義、能不能單獨revert」，不強行湊成剛好10個）。理由：單人專案，10條疊起來的分支鏈維護（rebase）成本大於好處；單一分支＋commit序列一樣能達到`git bisect`定位、單獨`revert`某一步、完整可讀歷史的效果。每步沿用既有的「Git 檢查點流程」：改完印出commit指令，暫停不自行執行，等使用者確認後才進下一步。最終`migration/proto3-to-agentic`完成後merge回`feature/agentic-rag-proto`；`feature/agentic-rag-proto`何時／要不要merge進`main`，是後續另一個決定，不在這次branch規劃範圍內。

**Entry point命名（已定案）**：`proto3_langgraph.py`本身連同retire的舊node檔案，遷移完成後**整個刪除**（不是「內部邏輯換掉但外殼檔名繼續存在」）——production在遷移完畢後，proto3的部分可以刪除，留下agentic的部分。新entry point叫**`rag/agentic_rag.py`**，維持`python -m rag.agentic_rag`這種簡潔的呼叫方式，取代`python -m rag.proto3_langgraph`。CLAUDE.md「Running — Agentic RAG」章節所有指令範例（Step 10）要跟著更新成這個新指令。

---

## Part 1：Proto3（production）現況架構

### 1.1 完整Graph結構

```
                                              START
                                                │
                                                ▼
                                  ┌─────────────────────────┐
                                  │ query_decomposition_node │  純keyword層偵測（_keyword_route()
                                  └────────────┬────────────┘   逐句比對type，2種以上才算複合）
                                                │
                              ┌─────────────────┴─────────────────┐
                         有sub_queries                        無sub_queries
                              │                                    │
                              ▼                                    ▼
                    ┌──────────────────┐                  ┌───────────────┐
                    │  [Send ×N]        │                  │  router_node   │  route()：2層keyword→LLM
                    │  sub_query_node    │                  └───────┬───────┘  分類 PROCEDURE/CONTACT/
                    │ （每子句各自跑一次  │                          │           KNOWLEDGE（RESOURCE另計）
                    │  route()→對應retrieval                       │
                    │  →office_lookup_node                  ┌──────┼──────┬──────────┐
                    │  →extraction_node，                "PROCEDURE""KNOWLEDGE""RESOURCE""CONTACT"
                    │  單次、非迴圈）                          │      │      │          │
                    └─────────┬────────┘                    ▼      ▼      ▼          │
                              │                     ┌──────────────┐ ┌───────────┐ ┌───────────┐
                              ▼                     │retrieval_anchor│ │retrieval_ │ │resource_  │
                    ┌──────────────────┐            │_node（determi- │ │node（KNOW-│ │node       │
                    │    merge_node     │            │nistic grep+    │ │LEDGE，整個 │ │(grep→get_ │
                    │（flatten+dedupe   │            │get_page）      │ │ReAct迴圈包│ │page→extract│
                    │ office_context/   │            └───────┬────────┘ │在一個func │ │_form_ids→ │
                    │ extraction_       │                    │          │內，對外不 │ │get_form全  │
                    │ checklist）        │              [Send ×N]        │可見/不可中│ │部，自己寫  │
                    └─────────┬────────┘             retrieval_expand   │斷，自己寫 │ │答案）      │
                              │                       _node（跟隨anchor│ │答案）     │ └─────┬─────┘
                              │                       抓到的連結/表單   └─────┬─────┘       │
                              │                       數量動態展開）           │             │
                              │                             │                 │             │
                              │                             ▼                 │             │
                              │                    ┌──────────────────┐       │             │
                              │                    │ office_lookup_node │◀─────┴─────────────┘
                              │                    │（無條件執行：      │      （全部3條非複合路徑
                              │                    │_offices_from_context│      都會流經這裡，即使
                              │                    │(...) or _PROCEDURE_│      沒有辦公室需要查）
                              │                    │OFFICES 6個硬編碼   │
                              │                    │名稱fallback）      │
                              │                    └─────────┬─────────┘
                              │                              ▼
                              │                    ┌──────────────────┐
                              │                    │  extraction_node  │（無條件執行：regex抽取
                              │                    │                    │ person_names/forms/notes/
                              │                    └─────────┬─────────┘ stations checklist）
                              │                              │
                              └──────────────────────────────┤
                                                              ▼
                                                    ┌──────────────────┐
                                                    │  synthesis_node   │  依query_type分支：
                                                    │                    │  RESOURCE→純pass-through
                                                    └─────────┬─────────┘  KNOWLEDGE→pass-through
                                                              ▼            +parametric fallback(E7)
                                                    ┌──────────────────┐  PROCEDURE/CONTACT→真的
                                                    │  self_eval_node   │  呼叫_SYNTHESIS_PROMPT
                                                    │ （只評估PROCEDURE，│
                                                    │  KNOWLEDGE/CONTACT/│
                                                    │  RESOURCE直接跳過）│
                                                    └─────────┬─────────┘
                                                     correction_hint set？
                                                    ┌─────────┴─────────┐
                                                   有(≤2次)              無
                                                    │                    │
                                                    ▼                    ▼
                                              回synthesis_node          END
```

### 1.2 Node職責一覽

| Node/檔案 | 職責 | 關鍵特徵 |
|---|---|---|
| `query_decomposition_node`（`decomposition.py`） | 純keyword層偵測複合query（`_keyword_route()`逐句比對type） | 已是`Send`平行（Step 2.5→Send升級完成） |
| `router_node`（`routing.py`） | `route()`分類PROCEDURE/CONTACT/KNOWLEDGE（21行，最小的node） | 純function call包裝，不做任何額外邏輯 |
| `retrieval_anchor_node`＋`retrieval_expand_node`（`retrieval_procedure.py`） | PROCEDURE專屬：anchor（deterministic grep+get_page）→ expand（`Send`平行，跟隨anchor抓到的連結/表單數量動態展開） | 唯一使用`Send`的地方之一（另一個是`sub_query_node`），寫入的都是`operator.add`累加器欄位 |
| `retrieval_node`（`retrieval_knowledge.py`） | KNOWLEDGE：**整個ReAct迴圈包在一個Python function裡**，內部自己跑多輪tool call，對外graph只看到一個node執行完 | 對`.stream()`不可見每一輪、無法在多輪中途插`interrupt()` |
| `resource_node`（`retrieval_resource.py`） | RESOURCE：grep→get_page→`extract_form_ids()`（掃**全部**grep到的頁面）→`get_form()`全部抓取（fetch-all，不篩選）→**自己內建一個mini synthesis prompt**直接寫答案 | 唯一不靠共用`synthesis_node`寫答案的路徑之一（KNOWLEDGE也是自己寫） |
| `office_lookup_node`（`office_lookup.py`） | 辦公室聯絡資訊查詢，**無條件執行**於PROCEDURE/KNOWLEDGE/RESOURCE/CONTACT全部4條非複合路徑 | `_offices_from_context(...) or _PROCEDURE_OFFICES`——6個硬編碼辦公室名稱（生僑組/住宿組/出納組/圖書館/國際合作事務處/教務處）當fallback，換一個procedure不保證還work |
| `extraction_node`（`extraction.py`） | 無條件執行的pre-synthesis checklist：regex抽取`person_names`/`forms`/`notes`（`_extract_candidate_notes`，涵蓋阿拉伯數字頓號/中文數字頓號/阿拉伯數字句點3種編號慣例）/`stations`（`_STATION_MARKER_RE`） | 獨立node，不是折進resource_node內部 |
| `synthesis_node`（`synthesis.py`） | **依`query_type`分支**：RESOURCE純pass-through（不呼叫LLM）；KNOWLEDGE pass-through+`_parametric_fallback`(E7)兜底；PROCEDURE/CONTACT才真的呼叫`_SYNTHESIS_PROMPT`生成答案 | 答案撰寫責任分散在3個不同地方（KNOWLEDGE自己/RESOURCE自己/這裡），不是統一集中 |
| `self_eval_node`（`self_eval.py`） | **`if state["query_type"] != QueryType.PROCEDURE: return state`**——只評估PROCEDURE答案 | 機制是deterministic checklist（`_SELF_EVAL_CRITERIA`：sources/procedure_format關鍵字+動態person_names/stations/notes checklist比對）＋**條件式**LLM Router-as-judge（只在`is_ambiguous()`為真時才多花一次LLM呼叫），不是單一LLM判斷；retry固定回`synthesis_node`，`_MAX_SELF_EVAL_RETRIES=2` |
| `_parametric_fallback`（`agent_runtime.py`，E7） | 當KNOWLEDGE/PROCEDURE/CONTACT完全查無資料時，**改用LLM自己的訓練知識回答**（標註「非政大官方文件」），不是誠實拒答 | **CLAUDE.md已記載為已知失效模式**：「Gemma4:31b occasionally fails to call the right tools...triggering E7 parametric fallback with inaccurate answer」——這是一個要不要遷移過去都需要重新評估的機制，不是單純「production有、agentic該補上」 |

### 1.3 已知限制（production自己文件裡記載的，遷移時要意識到，不是遷移才發現）

- CONTACT路徑`sources`永遠是空的（office_lookup直接注入，不走retrieval，來源URL缺失）
- `_STRIP_RE`/`_RESOURCE_STRIP_RE`這類關鍵字剝離正規表達式是whack-a-mole模式，這個session（D12當時）已經因為同樣理由拒絕在`agentic_main.py`裡採用類似設計（改用LLM語意判斷）
- KNOWLEDGE路徑turn-1多數決取樣（8-A）只對取樣雜訊型錯誤有效，模型系統性偏誤時無效

---

## Part 2：agentic_rag（`agentic_main.py`）現況架構

完整diagram/node table已經寫在`docs/phase_g_clean_pipeline_design.md` §O（2026-08-31新增），這裡不重複全文，只列出跟Part 1對照時最關鍵的幾個特徵：

- `plan_node`只讓CONTACT/RESOURCE/複合3種情況真正分流，PROCEDURE/KNOWLEDGE合併成一條路（D2）
- KNOWLEDGE是**真正的多node graph迴圈**（`rewrite_node↔domain_router_node↔agent_node↔tools`），每一輪對`.stream()`可見，理論上可以在輪次之間插`interrupt()`
- resource/contact是**deterministic marker觸發**（`_FORM_MARKER_RE`/`_OFFICE_MARKER_RE`經`_after_tools`/`_after_resource`路由），沒訊號就不觸發，不是無條件關卡
- `_detect_offices()`：Layer 1（substring pre-check，N.7新增）＋LLM judge對全部433筆catalog語意比對（D12），聯集回傳——比production的6個硬編碼名稱泛化範圍大得多
- `resource_node`內建D15的結構化抽取（`_extract_station_roles`/`_offices_from_role_keywords`/`_extract_checklist_blocks`），不是獨立extraction_node
- **答案撰寫完全集中**在唯一的`synthesis_node`，不管走哪條路徑，`agent_node`自己的`.content`一律discard
- `self_eval_node`是**單一LLM判斷**（不是keyword checklist+條件式LLM judge的組合），輸入含原始完整query，涵蓋所有非直接分流路徑（不是PROCEDURE-only），retry分兩層（情況A回`rewrite_node`／情況B回`plan_node`重新分類）
- 沒有`_parametric_fallback`：完全依賴`_SYNTHESIS_PROMPT`規則5的prompt層級誠實指示（「已經搜尋多次仍找不到，誠實說明未查到，不要杜撰」），沒有deterministic的「查無資料→退回訓練知識」機制

---

## Part 3：完整差異對照表

| 維度 | Proto3（production） | agentic_rag（`agentic_main.py`） | 遷移方向（初步判斷，Part 5展開） |
|---|---|---|---|
| 查詢分類 | 4分類**真的分流graph** | CONTACT/RESOURCE直接分流＋複合偵測；PROCEDURE/KNOWLEDGE合併（D2） | 採agentic版：拿掉PROCEDURE/KNOWLEDGE分流 |
| PROCEDURE路徑 | 獨立anchor+expand（`Send`平行） | 併入統一rewrite/agent/tools迴圈，無獨立路徑 | 採agentic版：整個retirement |
| KNOWLEDGE路徑 | 整個ReAct迴圈包在一個function裡，對外不可見 | 真正多node graph迴圈，逐輪可見/可中斷 | 採agentic版：這是原始rebuild的核心動機之一 |
| RESOURCE路徑 | fetch-all（掃**全部**grep到的頁面找表單編號），自己內建mini synthesis prompt | fetch-all（`extract_form_ids(context_text)`偵測到即抓，`_judge_forms()`僅在無context時fallback），答案交給共用`synthesis_node` | 採agentic版的fetch-all原則（兩邊精神一致，agentic版已驗證＋N.4修正過pre-fetch judge風險）；答案撰寫改成統一走`synthesis_node`（見「答案撰寫責任」列） |
| CONTACT路徑 | 無條件流經`office_lookup_node`，`_PROCEDURE_OFFICES`6個硬編碼名稱fallback | `contact_node`，`_detect_offices()`偵測到才觸發，全catalog範圍 | 採agentic版：這是D12已經解決、production還沒解決的closed-loop問題 |
| 辦公室偵測機制 | substring/硬編碼名單 | Layer1 substring＋LLM對433筆catalog語意判斷（N.7新增Layer1，14/15樣本從失敗變成功） | 採agentic版 |
| office_lookup/extraction執行時機 | **無條件**執行於全部4條非複合路徑 | Deterministic marker觸發，沒訊號不執行 | 採agentic版：這是使用者當初「procedure塞了很多不會用到的關卡」抱怨的直接對應修法 |
| 結構化checklist抽取（stations/notes/forms） | 獨立`extraction_node`，無條件執行，regex涵蓋3種編號慣例（阿拉伯頓號/中文數字頓號/阿拉伯句點） | 折進`resource_node`內部（D15），只在resource_node實際跑到時執行 | **需要決定**：agentic版目前的`_extract_checklist_blocks`規則驗證覆蓋面比production的`_extract_candidate_notes`窄（只驗證過2份表單 vs production涵蓋3種編號慣例且用在更多procedure上）——這塊production可能比agentic更成熟，見Part 6.1 |
| 答案撰寫責任 | 分散：KNOWLEDGE自己寫、RESOURCE自己寫、PROCEDURE/CONTACT才共用`synthesis_node` | 完全集中：唯一`synthesis_node`，`agent_node`的`.content`一律discard | 採agentic版：集中化本身就是一個簡化，遷移後RESOURCE/KNOWLEDGE的答案撰寫邏輯要並入共用`synthesis_node`（KNOWLEDGE natural，因為新架構KNOWLEDGE本來就沒有自己寫答案的機制；RESOURCE需要把production`_RESOURCE_SYNTHESIS_PROMPT`的「不落地不可編造連結」精神併入共用`_SYNTHESIS_PROMPT`規則4） |
| self_eval涵蓋範圍 | **只評估PROCEDURE**，KNOWLEDGE/CONTACT/RESOURCE完全不評估 | 涵蓋所有非直接分流路徑 | 採agentic版：更完整，但需注意涵蓋範圍變大後LLM呼叫成本也變大（Part 6.3） |
| self_eval機制 | Deterministic checklist（`_SELF_EVAL_CRITERIA`+動態person_names/stations/notes）＋條件式LLM Router-as-judge | 單一LLM判斷（讀原始完整query＋答案＋分類＋resource/contact-fired旗標） | **需要合併設計**，不是單純二選一——見Part 6.2 |
| self_eval retry目標 | 固定回`synthesis_node` | 情況A回`rewrite_node`／情況B回`plan_node`（能救回錯誤分類） | 採agentic版：兩層設計能力更強，能接住`_MAX_SELF_EVAL_RETRIES`治不了的「一開始就分類錯」情況 |
| self_eval retry上限 | 明確`max 2` | 無明確上限（doom-loop+turn ceiling代替） | **§0.1已決定**：採agentic版 |
| 複合query偵測 | `_keyword_route()`逐句比對，純keyword無LLM | 同樣邏輯（`plan_node`內建），`_CLAUSE_SPLIT_RE`切句 | 兩邊邏輯幾乎相同，可直接沿用 |
| 複合query執行 | `Send`平行`sub_query_node`（單次route()→retrieval→office_lookup→extraction，非迴圈） | v1序列`for`迴圈＋`_build_loop_graph().invoke()`（nested、獨立編譯子圖，非`Send`） | **§0.2已定案**：不能直接搬`Send`模式，見Part 5 Step 6 |
| Domain routing | `route_domain()`/`is_ambiguous()`個別在各node內呼叫，各自對自己的查詢文字判斷 | `domain_router_node`獨立graph node，第1輪執行一次，設定`subdomain_hint`供全程使用 | 採agentic版：更集中，且第1輪判斷、後續複用，比每個node各自判斷更有效率 |
| Turn/迴圈上限 | 未在這輪讀code時明確找到專屬上限（`retrieval_node`內部可能有隱藏上限，需要之後查證） | 無`_MAX_TURNS`（刻意拿掉），只靠doom-loop偵測 | 沿用agentic版，但遷移時要順便查證production`retrieval_node`內部原本用什麼上限，避免遺漏一個沒被記錄的保護機制 |
| Parametric fallback（E7） | 有：`_parametric_fallback()`，查無資料時退回LLM訓練知識回答（標註非官方） | 無：純prompt層級誠實指示，沒有deterministic fallback | **需要決定，不是單純移植**：CLAUDE.md記載E7是已知失效模式（觸發時給出不準確答案），見Part 6.4 |
| Tool呼叫機制 | 各node內直接呼叫`agent_tools.py`的plain function（`grep_texts`/`get_page`等） | 5個真正LangChain `@tool`函式，經`ToolNode`分派，支援`InjectedState` | 採agentic版：`ToolNode`+`InjectedState`已驗證能自動讀取`subdomain_hint`等狀態，比手動傳參數更簡潔 |
| Tool-calling LLM client | `llm_client.py`原生`ollama`/`chat_with_tools()` | `ChatOllama.bind_tools()`（LangChain原生tool-calling） | 這個session已花一輪spike驗證`ChatOllama`+Ollama Cloud相容（前提：system prompt要求明確），採agentic版；但要注意這牽動`llm_client.py`是否要新增依賴`langchain-ollama`（已裝過，見§13紀錄） |

---

## Part 4：目標遷移後架構

### 4.1 檔案層級的去留

| Production現有檔案 | 遷移後狀態 |
|---|---|
| `rag/nodes/routing.py`（`router_node`） | **保留＋修改**：拿掉PROCEDURE/KNOWLEDGE的分流輸出，只回傳CONTACT/RESOURCE/其餘（比照`plan_node`的`_after_plan`） |
| `rag/nodes/decomposition.py` | **保留＋修改**：`query_decomposition_node`的keyword偵測邏輯幾乎不用動；`sub_query_node`／`_dispatch_sub_queries`／`merge_node`要整個換成agentic版的nested-invoke模式（§0.2），不能維持現有直接`Send`進迴圈的寫法 |
| `rag/nodes/retrieval_procedure.py` | **整個retire**——PROCEDURE作為獨立路徑消失（D2），anchor+expand機制沒有續用的地方 |
| `rag/nodes/retrieval_knowledge.py` | **整個retire**，換成agentic版的`rewrite_node`/`domain_router_node`/`agent_node`/`tools`四個獨立node |
| `rag/nodes/retrieval_resource.py` | **換成agentic版`resource_node`**（fetch-all＋D15結構化抽取），但要把production `_RESOURCE_SYNTHESIS_PROMPT`裡「不可編造下載連結」的明確警語併入共用`_SYNTHESIS_PROMPT` |
| `rag/nodes/office_lookup.py` | **換成agentic版`contact_node`**＋`_detect_offices()`（Layer1+LLM對全catalog判斷），拿掉`_PROCEDURE_OFFICES`硬編碼fallback |
| `rag/nodes/extraction.py` | **整份retire**（§6.1已定案）——`_extract_candidate_notes()`不搬（whack-a-mole編號慣例列舉，跟`_STRIP_RE`/舊版`_OFFICE_NAME_MAP`同一種已被拒絕過的模式），只保留D15機制（折進`resource_node`） |
| `rag/nodes/synthesis.py` | **換成agentic版`synthesis_node`**（統一集中，N.5流程手冊格式，N.4表單篩選規則），但要吸收production RESOURCE分支「不可編造連結」的精神（併入新規則） |
| `rag/nodes/self_eval.py` | **需要合併設計**（Part 6.2），不是單純二選一 |
| `rag/nodes/state.py` | **擴充**：新增`turn`/`rewritten`/`stuck_turns`/`subdomain_hint`等agentic用到、production目前沒有的欄位；保留`sub_queries`等複合query相關欄位（邏輯沿用） |
| `rag/agent_runtime.py` | `_offices_from_context`/`_offices_from_query`/`_PROCEDURE_OFFICES`**跟著office_lookup.py一起retire**；`_parametric_fallback`去留見Part 6.4；`_staleness_warning`（E6）**保留**，兩邊都用得到，遷移不受影響 |
| `rag/proto3_langgraph.py` | **整個刪除**（§0.3已定案），換成新entry point`rag/agentic_rag.py`（`python -m rag.agentic_rag`） |
| **新增**：`rag/agentic_rag.py` | 新entry point：`build_graph()`/`run()`/CLI，只做graph組裝，比照舊`proto3_langgraph.py`的角色 |
| **新增**：`rag/agentic/`package整套 | 取代舊`rag/nodes/`，內部依tools/logic/nodes三層組織，完整結構見Part 7 |

### 4.2 目標Graph結構（遷移完成後）

```
                                    START
                                      │
                                      ▼
                              ┌───────────────┐
                              │   plan_node    │  route()簡化為CONTACT/RESOURCE/其餘
                              └───────┬───────┘  ＋複合偵測（沿用decomposition.py的
                                      │            _keyword_route()邏輯）
                     ┌────────────────┼────────────────┬─────────────────┐
                  "compound"      "resource"        "contact"        "knowledge"
                     │                │                  │          (含原PROCEDURE)
                     ▼                ▼                  ▼                  │
         ┌───────────────────┐  ┌───────────┐   ┌───────────────┐          │
         │multi_sub_query_node│  │resource_node│  │ contact_node   │          │
         │（nested-invoke，   │  │(fetch-all+  │  │(_detect_offices│          │
         │ 非Send直接進迴圈， │  │ D15結構抽取)│  │ Layer1+LLM)    │          │
         │ 見§0.2；是否升級  │  └─────┬─────┘   └───────┬───────┘          │
         │ Send見Part5 Step6）│        │                    │                  │
         └──────────┬─────────┘        ▼                    │                  │
                     │            _after_resource            │                  │
                     │                  │                     │                  │
                     │             "contact"/"rewrite"        │                  │
                     │                  └────────┬───────────┘                  │
                     │                            ▼                              ▼
                     │                    ┌───────────────┐            ┌───────────────┐
                     │                    │ contact_node   │───────────▶│ rewrite_node   │◀─┐
                     │                    └───────────────┘             └───────┬───────┘  │
                     │                                                          ▼          │
                     │                                                ┌─────────────────┐  │
                     │                                                │domain_router_node│  │
                     │                                                └────────┬────────┘  │
                     │                                                          ▼           │
                     │                                                  ┌───────────────┐   │
                     │                                                  │  agent_node    │   │
                     │                                                  └───────┬───────┘   │
                     │                                              tools/end（doom-loop     │
                     │                                                  │     偵測，無_MAX_   │
                     │                                                  ▼     TURNS）        │
                     │                                            ┌───────────────┐│         │
                     │                                            │tools(ToolNode)│└─────────┘
                     │                                            └───────┬───────┘
                     │                                          resource/contact/rewrite
                     │                                          （回上面對應node）
                     └────────────────────────────────────────────────────┤
                                                                            ▼
                                                                  ┌─────────────────┐
                                                                  │  synthesis_node  │（統一，吸收
                                                                  └────────┬────────┘ RESOURCE防編造
                                                                            ▼          連結規則）
                                                                  ┌─────────────────┐
                                                                  │  self_eval_node  │（合併設計，
                                                                  └────────┬────────┘ Part 6.2）
                                                                            │
                                                                    "end"/"rewrite"/"plan"
                                                                            ▼
                                                                           END
```

（跟`agentic_main.py`現況架構的差異：這是**production的query_decomposition_node keyword邏輯+plan_node**要不要合併成一個node，還是保留兩個各司其職——見Part 5 Step 1，圖上暫時標成`plan_node`一個節點，實際定案後可能拆回兩個）

---

## Part 5：逐步遷移計畫

每步比照原始plan.md的「Git 檢查點流程」：改完印出commit指令，**暫停不自行執行**，等使用者確認後才進下一步。每步都要有明確的驗證方式，不能只是「看起來對」。

### Step 1：✅ 已完成（2026-09-01）——`state.py`欄位擴充 + `plan_node`／`query_decomposition_node`合併方式定案

**決定**：合併成一個node（`plan_node`），直接沿用`agentic_main.py`已經驗證過的實作（複合偵測`_CLAUSE_SPLIT_RE`+`_keyword_route()`優先，非複合才`_classify_query()`）。理由：production把這兩個職責拆成`decomposition.py`+`routing.py`兩個檔案，沒有找到任何文件記載這是刻意的設計取捨（只是歷史上分批漸進開發留下的結果），agentic版已經合併驗證過整個session，直接沿用比重新拆分更單純。實際`plan_node`程式碼在Step 2實作，這步只定案設計方向。

**實作內容**：
1. 新增`rag/agentic/`套件骨架（`agentic/__init__.py`、`agentic/nodes/__init__.py`、`agentic/tools/__init__.py`、`agentic/logic/__init__.py`）
2. 新增`rag/agentic/state.py`：以production`rag/nodes/state.py`的`AgentState`為基礎（**全部17個既有欄位保留**），新增4個agentic版多輪迴圈需要、production從未有過的欄位——`turn`/`stuck_turns`（`rewrite_node`/`agent_node`跨多個獨立graph node遞增的即時迴圈狀態，正是§0.2記載的、讓`Send`不能直接指向這個迴圈的那種欄位形狀）、`rewritten`（本輪`rewrite_node`輸出）、`subdomain_hint`（`domain_router_node`第1輪設定，全程複用；production沒有對應欄位，因為`route_domain()`/`is_ambiguous()`是每個retrieval node各自呼叫，不是共用一個狀態）

**驗證結果**：
- `python -c "from rag.agentic.state import AgentState"`——import正常，21個欄位（17舊+4新）全部確認存在
- `git status`確認`rag/nodes/`／`rag/proto3_langgraph.py`完全未被觸碰（純新增檔案，沒有修改任何production現有檔案）
- `python -m rag.proto3_langgraph "出納組電話幾號" --no-eval`——production既有pipeline完整跑過一次，答案正確（出納組分機62123），完全不受影響

**過程中的插曲**：驗證途中一度發現`python`指令完全找不到（`which python`/`type python`皆失敗），排查後確認是使用者本地venv環境意外關閉，不是這次改動造成的問題——venv重新啟用後（`/mnt/d/NCCU_DATA/NCCU-AI-SYSTEM/venv`，在repo外層一層，說明先前用`find`在repo內搜尋找不到venv目錄的原因），全部驗證正常通過。

### Step 2：`routing.py`拿掉PROCEDURE/KNOWLEDGE分流
**改動**：`_after_router`（或整併後的`_after_plan`）只回傳`contact`/`resource`/`compound`/`knowledge`（PROCEDURE不再是獨立分支）。
**驗證**：對6-7個既有regression案例（休學/復學/在職生復學/退宿規定/選課上限/出納組電話/表單下載），確認分類結果符合預期（原PROCEDURE案例現在應該落在`knowledge`分支）。

### Step 3：接入agentic版的`rewrite_node`/`domain_router_node`/`agent_node`/`tools`，取代`retrieval_node`＋`retrieval_anchor_node`/`retrieval_expand_node`
**改動**：這是最大的一步——整個KNOWLEDGE/原PROCEDURE路徑換成真正的多node迴圈。`retrieval_procedure.py`／`retrieval_knowledge.py`的既有邏輯**先不刪檔案**（保留當備份對照，等Step 3驗證通過、確認新路徑穩定後才在後續步驟移除）。
**驗證**：休學（原PROCEDURE）＋選課上限/退宿規定（原KNOWLEDGE）三題，`rag/eval.py`分數不能明顯低於production既有基準；用`.stream()`確認每一輪對外可見（這是這一步存在的核心目的，順便驗證）。

### Step 4：`resource_node`／`contact_node`換成agentic版
**改動**：`office_lookup.py`的`_PROCEDURE_OFFICES`硬編碼fallback整個拿掉，換成`_detect_offices()`；`retrieval_resource.py`換成fetch-all版本，`_RESOURCE_SYNTHESIS_PROMPT`的「防編造連結」精神併入`synthesis.py`的共用prompt（不是resource自己保留一份獨立synthesis）。
**驗證**：出納組電話幾號（1輪，Layer1修好的案例）／休學表單下載／休學表單裡7+個蓋章站點的聯絡人完整度（比對D12/N.7已經在`agentic_main.py`驗證過的結果，遷移後應該要重現同樣的完整度，不能退步）。

### Step 5：`synthesis_node`統一化 + `extraction.py`整份retire（Part 6.1已定案）
**改動**：`extraction.py`整份retire——`resource_node`只保留agentic版的`_extract_checklist_blocks`/`_extract_station_roles`/`_offices_from_role_keywords`，不搬`_extract_candidate_notes()`（§6.1理由：whack-a-mole編號慣例列舉，不符合這個專案的設計原則）。
**驗證**：休學/退宿規定兩份表單（已驗證過D15機制的案例）抽取結果不退步；額外確認一次「KNOWLEDGE路徑頁面有站點/notes但未觸發resource_node」這個已知缺口（§6.1）目前有沒有被任何既有regression案例踩到（預期沒有，用來確認這確實只是理論風險，不是被忽視的真實問題）。

### Step 6：複合query處理遷移（`Send` vs nested-invoke，取決於Part 6討論或當下技術驗證）
**改動**：`decomposition.py`的keyword偵測邏輯沿用；`sub_query_node`／`merge_node`換成`multi_sub_query_node`的nested-invoke模式。**明確待辦，不在這步解決**：要不要把nested-invoke也升級成`Send`（`Send`分支指向一個內部做nested`.invoke()`的wrapper node，這是spike_nested_invoke.py驗證過的安全模式，不是production現有的直接`Send`模式）——先用v1序列版驗證正確性，比照當初agentic_main.py自己的D14先例（先序列驗證，Send升級列為明確後續）。
**驗證**：「如何辦理休學，圖書館的電話是多少」這類複合案例，兩個子問題都要完整回答，跟production現有`Send`版本的答案品質對照，不能退步。

### Step 7：`self_eval_node`兩階段設計實作（Part 6.2/6.3已定案）
**改動**：實作Stage 1（deterministic checklist，複用production `_SELF_EVAL_CRITERIA`風格＋agentic D15結構化資料當比對依據）+ Stage 2（agentic版`_SELF_EVAL_PROMPT`單一LLM判斷，只在Stage 1通過才執行）。**明確：對所有路徑（含Plan_node直接分流成功的案例）一致執行，不做任何「路徑看起來簡單就跳過」的優化**（§6.3撤回理由：路徑順利不代表答案正確，`_detect_offices()`過去14/15失敗案例都是「一次分到位」的路徑）。
**驗證**：至少涵蓋production現有`_SELF_EVAL_CRITERIA`能抓到的案例＋agentic版兩層retry能救回的複合問題漏答案例，兩邊都要能通過；額外驗證「出納組電話幾號」這類Layer1修好後1輪完成的案例，self_eval仍正確執行兩階段（不因為是直接分流案例而被跳過）。

### Step 8：Parametric fallback（E7）——不遷移（Part 6.4已定案）
**改動**：`_parametric_fallback()`/`_PARAMETRIC_SYSTEM`不遷移，agentic版維持現況（純prompt層級誠實指示，無deterministic退回訓練知識機制）。
**驗證**：故意構造「完全查無資料」的query，確認遷移後系統誠實回應「查無資料」，不會退回LLM自己的訓練知識回答（不管有沒有標註來源）。
**明確不在這步範圍內**：§6.4記錄的「deterministic強制覆蓋、給固定誠實回應」替代機制——這是獨立的後續設計項目，這一步只確認E7本身沒有被遷移過去，不代表要在這步順便把替代機制也做掉。

### Step 9：`rag/proto3_langgraph.py`刪除，新entry point`rag/agentic_rag.py`（§0.3已定案）
**改動**：`rag/proto3_langgraph.py`整個刪除；新增`rag/agentic_rag.py`（`build_graph()`/`run()`/CLI，只做graph組裝，import自`rag.agentic.*`），取代舊entry point，CLI呼叫方式改成`python -m rag.agentic_rag`。`rag/nodes/`裡Step 1-8確認要retire的檔案（`retrieval_procedure.py`/`retrieval_knowledge.py`/`extraction.py`／舊版`office_lookup.py`/`retrieval_resource.py`）在這步一併實際刪除（前面幾步保留備份對照用，這步才是真正清除）。
**驗證**：完整跑一次CLAUDE.md記載的既有eval基準（如何辦理休學26/26），確認遷移後的分數不低於遷移前；確認`python -m rag.proto3_langgraph`已經無法執行（檔案真的刪了，不是留著沒人用）。

### Step 10：全面regression + CLAUDE.md文件更新
**改動**：CLAUDE.md的「Agentic RAG System」章節（architecture diagram、eval分數表、already-resolved issues表、所有`python -m rag.proto3_langgraph`指令範例）需要整份更新，反映遷移後的真實現況（`rag/agentic_rag.py`＋新架構）——不能讓文件繼續描述已經刪除的`proto3_langgraph.py`/`retrieval_procedure.py`/`_PROCEDURE_OFFICES`等機制。
**驗證**：CLAUDE.md原本列出的所有已知issue（KNOWLEDGE路徑不穩定/CONTACT source attribution/LaTeX符號）逐項確認遷移後是否仍然存在，更新狀態。

---

## Part 7：程式碼組織——`rag/agentic/`套件結構（tools/logic/nodes三層）

**動機**：遷移本來就要大幅重寫`rag/nodes/`的內容，這不是額外的投機重構——藉這個機會把現在`agentic_main.py`裡混在一起的「graph node轉接層」跟「純業務邏輯」拆開，理由：`logic/`層的函式（`_detect_offices()`/`_judge_forms()`這類）不依賴LangGraph state，可以直接單元測試（這個session這幾輪一直在用臨時腳本手動測`_detect_offices()`/`_judge_forms()`的可靠度，如果它們一開始就是不依賴state的純函式，這類測試會更自然、不用組假的`AgentState`字典）。

### 7.1 套件結構

```
rag/
├── agentic_rag.py          — entry point：build_graph()/run()/CLI（取代proto3_langgraph.py）
└── agentic/                 — 套件，取代舊rag/nodes/
    ├── state.py                — AgentState schema
    ├── tools/                    — @tool函式（LangChain工具定義），4個檔案，理由見7.2
    │   ├── search.py               search_texts
    │   ├── grep.py                  grep_texts_tool
    │   ├── page.py                   get_page_tool, extract_links_tool
    │   └── form.py                    get_form_tool
    ├── logic/                    — 純業務邏輯，不依賴LangGraph state，可獨立單元測試
    │   ├── office_detection.py       _detect_offices()（Layer1 substring + LLM judge，N.7）
    │   ├── form_extraction.py         _judge_forms/_list_forms_metadata/_extract_station_roles/
    │   │                                _offices_from_role_keywords/_extract_checklist_blocks（D15）
    │   ├── rewrite.py                  _rewrite_query/_judge_candidates
    │   └── self_eval_checks.py          Stage 1 deterministic checklist邏輯（Part 6.2）
    └── nodes/                    — 薄的graph node轉接層，讀state、呼叫logic/、回傳state delta
        ├── plan.py                   plan_node, _after_plan（含複合偵測，見Step 1決定是否吸收
        │                              decomposition.py的keyword邏輯）
        ├── loop.py                    rewrite_node, domain_router_node, agent_node, _after_agent
        ├── resource.py                 resource_node, _after_resource
        ├── contact.py                   contact_node
        ├── synthesis.py                  synthesis_node
        ├── self_eval.py                   self_eval_node（Stage1+2）, _after_self_eval
        └── compound.py                     multi_sub_query_node, _build_loop_graph
```

### 7.2 `tools/`拆4個檔案——已定案

拆成`search.py`/`grep.py`/`page.py`/`form.py`（不是單一`tools.py`），理由（使用者明確要求）：這5個`@tool`函式現在雖然不大，但之後還可能增加或延伸新工具，先建立好每個工具各自一個檔案的架構，比之後工具變多才回頭拆分更有秩序。`page.py`把`get_page_tool`跟`extract_links_tool`放同一個檔案（兩者都圍繞著「已知URL」這個共同主題操作），其餘各自獨立成檔。

### 7.3 這個結構跟Part 4/5既有規劃的對照

Part 4提到的「新增`rag/nodes/plan.py`」「新增`rag/tools.py`」等條目，正式路徑改為`rag/agentic/nodes/plan.py`／`rag/agentic/tools/*.py`（4檔）——Part 4表格本身不重複修訂路徑細節，以這節（Part 7.1）的套件結構為準。Part 5的每個Step在實作時，新增的檔案要放進對應的`tools/`/`logic/`/`nodes/`子目錄，不是散放在單一目錄下。

---

## Part 6：討論定案的決策點（2026-08-31，四項全部定案）

### 6.1 ✅ 已定案：`extraction.py`折進`resource_node`（agentic架構），不搬`_extract_candidate_notes()`

**架構位置**：採agentic版——抽取折進`resource_node`內部，只在有訊號（`extract_form_ids()`偵測到表單）時才執行，不做production式的「無條件對全部`context_pages`跑」。理由：跟D3/D6「非必要不用deterministic固定關卡」的精神一致，符合這個session一路的架構原則。

**內容範圍**：**不搬**production的`_extract_candidate_notes()`（3種編號慣例：阿拉伯頓號/中文數字頓號/阿拉伯句點）。理由（使用者明確指出）：這是「靠手動列舉的編號慣例去猜測文字結構」，本質上跟這個session已經拒絕過的`_STRIP_RE`/舊版`_OFFICE_NAME_MAP`同一種whack-a-mole模式——只涵蓋目前觀察到的3種格式，遇到第4種編號慣例又要手動加規則，production自己的docstring也承認「no claim this exhausts every convention」。

**最終範圍**：`resource_node`只保留`_extract_checklist_blocks`（□核取方塊多選項）+ `_extract_station_roles`/`_offices_from_role_keywords`（編號站點+角色反查catalog）。

**明確接受的已知缺口，不修**：KNOWLEDGE路徑的頁面如果本身有站點表格或條列注意事項、但沒有引用任何moltke表單編號（未觸發`resource_node`），這類內容不會被抽取——production的`extraction_node`因為對全部`context_pages`都跑，理論上不會漏。這個缺口目前只是理論風險（沒有實際觀察到案例），比照這個session一貫「沒證據不預先修」的原則，記錄但不現在處理。

### 6.2 ✅ 已定案：`self_eval_node`兩階段設計

**Stage 1（deterministic checklist，免費）**：沿用production風格的keyword/結構化檢查（來源URL、步驟格式、姓名、站點——複用agentic版D15已抽取出的結構化資料當比對依據）。
**Stage 2（LLM整體判斷，只在Stage 1通過才執行）**：agentic版`_SELF_EVAL_PROMPT`的單一LLM判斷（讀原始完整query，抓Stage 1抓不到的「格式對但離題」「複合問題漏答」）。

Stage 1失敗直接給明確、deterministic的correction hint（不需要LLM描述缺什麼，checklist本身就知道缺什麼），跳過Stage 2；Stage 1通過才進Stage 2。

### 6.3 ✅ 已定案：self_eval對全部路徑一致執行，不做「bounded案例跳過」優化

**原提案（已撤回）**：曾提議「Plan_node直接分流成功、agent_node全程沒呼叫過工具」時跳過Stage 2的LLM判斷，省成本。

**撤回理由（使用者指出，成立）**：路徑順不順利（有沒有經過`agent_node`的工具選擇）跟答案對不對，是兩件不相關的事。Plan_node一次分流成功，只代表分類判斷+deterministic抓取沒出錯，不代表`_detect_offices()`挑對了辦公室、`_judge_forms()`挑對了表單、或`synthesis_node`把資料組得完整正確。**這個session自己的N.6/N.7發現是最好的反例**：`_detect_offices()`在Layer 1修好前15次裡14次判斷失敗，每一個失敗案例當時都是「Plan_node一次分到位」的路徑——如果self_eval因為路徑看起來簡單就被跳過，這些錯誤會完全沒有第二層把關。

**結論**：self_eval（兩階段都算）對所有路徑一致執行，不因為執行路徑「看起來簡單/直接」而省略。6.2的Stage 1免費檢查仍然是唯一的省成本手段，不再疊加額外的「整段跳過」優化。

### 6.4 ✅ 已定案：不遷移Parametric fallback（E7），且不只是因為「未測試」——是這個方向本質上不太可能有用

**E7實際機制**（讀`agent_runtime.py`確認）：`_PARAMETRIC_SYSTEM`——「官方文件搜尋未找到相關資料，請根據你的訓練知識回答，並在答案最前方加上【以下來自模型訓練知識，非政大官方文件，請至官網確認】」。也就是retrieval失敗時，讓LLM自由用自己的訓練知識回答（有標註來源）。CLAUDE.md已記載這是已知失效模式（「觸發時給出不準確答案」）。

**使用者提出的更根本理由，比「未測試」更站得住腳**：NCCU的內部行政規定/表單/流程是高度機構特定、公開網路資料極少覆蓋的長尾知識——不管換多強的LLM，訓練資料裡本來就不太可能有可靠的政大特定資訊，這不是「模型不夠強」能解決的問題，是這個領域本來就不在任何通用LLM的訓練分布裡；加上目前部署的又不是最強的模型，是雙重不可靠。E7整個設計前提（retrieval失敗時退回LLM自己的知識還能給點有用資訊）在這個領域大機率不成立。

**結論**：不遷移E7。**未來方向（記錄，暫不設計/實作）**：如果要處理「retrieval真的什麼都沒查到」的情況，不應該讓LLM在「誠實拒答」跟「用自己的訓練知識回答」之間自由選——應該用一個deterministic檢查（`context_pages`/`office_context`是否真的都是空的，這是客觀可驗證的狀態，跟production自己E7的觸發條件用同一個訊號）強制覆蓋synthesis輸出，直接給固定格式的誠實回應（例如「查無相關資料，建議直接洽詢OO處室或查閱政大官網」），完全不讓LLM有機會退回自己的訓練知識——從機制上杜絕「聽起來像有根據、其實是幻覺」的可能，不是依賴prompt指示LLM乖乖誠實。**這個機制本身還沒設計，列為遷移完成後的獨立後續項目，不在這輪遷移範圍內**。

---

## 待辦：這份文件的維護方式

比照`phase_g_clean_pipeline_design.md`的模式——遷移過程中每完成一個Step，要回來更新對應章節的狀態（標記完成／記錄實際遇到的問題／訂正原本的假設），不要讓這份文件變成一次性寫完就過時的規劃稿。Part 6的每個討論點定案後，要把結論寫回Part 6對應小節（不是刪掉，保留討論過程），並視情況回頭修正Part 4/5裡依賴這些決定的地方。
