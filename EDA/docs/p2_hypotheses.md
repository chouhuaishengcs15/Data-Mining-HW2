# P2 Feature Hypotheses 與 P5 Handoff

- 撰寫：P2（Claude Code Fable）；初版 2026-11-10，R3、R5、Final 修訂 2026-11-11（依 GPT-6 審查）
- 證據：`EDA/outputs/*.csv`（全部只用 train；`python EDA/p2_eda.py` 可重現）
- 狀態：H1、H3 都是 **Hypothesis**，**Pending P5 Verification**；H2′ 是補充的負面觀察。
- 正式定義：**H1 = `ALL21` 對 `NO_VARIABILITY17`；H3 = `ALL21` 對 `DROP_WIDTH20` 對 `PRUNED18`。** `PRUNED17` 只是選做對照。


> **R5 更新（2026-11-11）**：H3 的主要特徵集合改為 **PRUNED18**（移除 Width、Mode、Mean；先移除 Width 再分群，Min 保留）。PRUNED17（另外移除 Min）保留為選做對照，不是主結果，也不得宣稱無資訊損失。主要比較為 `ALL21`、`DROP_WIDTH20`、`PRUNED18`（H3）與 `NO_VARIABILITY17`（H1）；定義見 `EDA/outputs/feature_sets.json`，尚待 P3／P5 確認。H1、H3 目前狀態皆為 **Pending P5 Verification**。下文的 Train-only 探針沒有測過 PRUNED18。

## R3 修訂摘要

| 修訂 | 原本的寫法 | 現在的寫法 |
|---|---|---|
| H3 的數學錯誤 | 「Min 可由 Width 與 Max 還原」 | Width 與 Min 同時移除後，Min **無法**還原。PRUNED17 是啟發式挑選，不是無損精簡 |
| H3 的主張 | 「不損失效能」 | 「能否維持近似表現」待驗證；加入只移除 Width 的 20 特徵對照 |
| H1 的主張 | 「變異度群是必要的」 | 「提供互補的偵測資訊」，待模型驗證 |
| 子群命名 | 平坦型／減速型、「不需要標籤的規則」 | DP-absent／DP-present pathologic segments；分組前已用 Pathologic 標籤篩選 |
| 探針的定位 | 列為支持證據 | 只供內部排序假說；它使用了訓練集的異常標籤 |

## 0. 先講結論

| ID | 假說 | 訓練集 EDA 的支持程度 | 建議 |
|---|---|---|---|
| **H1** | 變異度特徵群（ASTV、MSTV、ALTV、MLTV）提供其他特徵補不上的偵測資訊 | 分布證據支持，值得正式測試 | 主要假說 |
| **H3** | 依正常資料相關性挑選的 18 個特徵（PRUNED18），表現與 21 個相近 | 冗餘證據明確；效能是否相近未知 | 主要假說 |
| **H2′** | 減速的類型有資訊；把 DL、DS、DP 加總沒有幫助 | 原本「衍生組合更好」的 H2 不被支持 | 補充（負面結果） |

推薦正式寫 **H1 與 H3**。H2′ 在消融表中多一列即可。

### 支撐假說的共同觀察：病理片段依 DP 分成兩個描述性子群

做法：**先選出 train 的 104 筆 Pathologic，再依 `DP > 0` 切成兩組**（`EDA/outputs/pathologic_phenotypes.csv`、`fig5`）。

| 中位數 | Normal（972 筆） | DP-absent pathologic（39 筆／18 exams） | DP-present pathologic（65 筆／12 exams） |
|---|---|---|---|
| ASTV | 40 | 71 | 62 |
| MSTV | 1.2 | **0.3** | **2.7** |
| ALTV | 0 | **69** | 0 |
| MLTV | 8.1 | 3.9 | 0 |
| Width | 68 | **18** | **113** |
| Mean | 134 | 135 | **96** |
| Variance | 8 | **0** | **57** |
| ASTV ≥ 60 的比例 | 16% | 95% | 83% |

- 兩個子群在 MSTV、ALTV、Width、Variance 上往相反方向偏離正常；ASTV 在兩個子群都偏高。這解釋了 MSTV 的單向 AUC 只有 0.59，而雙向（|z|）AUC 有 0.80。
- 意涵：單一方向的重要度指標會低估雙峰特徵；one-class 模型衡量「離正常多遠」，比較接近 |z| 的觀點。
- **限制**：
  - 這是用了 Pathologic 標籤之後的描述性分組，不是臨床病理分型，也不是不看標籤的發現。
  - DP-present 只來自 12 個 exam；有 2 個 exam 同時出現在兩組。
  - 文件中若出現「曲線較平」「有延長減速」等說法，只是描述這兩組的數值樣貌。

### Train-only 探針（內部參考，不進 §4）

做法：只在 train 內，用 exam 分組的 5-fold CV（重複 5 次）；每一折只用 Normal 估計 z-score 與 Ledoit–Wolf 共變異數，對 held-out exam 算 Mahalanobis 距離。Pathologic 基率 0.097。（`EDA/outputs/group_probe.csv`）

| 特徵集合 | 特徵數 | AUROC | PR-AUC |
|---|---|---|---|
| ALL21 | 21 | 0.947 | 0.647 |
| ALL21 − 變異度群 | 17 | 0.853 | 0.565 |
| ALL21 − AC/DL/DS/DP | 17 | 0.934 | 0.480 |
| ALL21 − Width | 20 | 0.948 | 0.649 |
| PRUNED17（選做對照；探針沒有測 PRUNED18） | 17 | 0.954 | 0.665 |
| ALL21 − 全部 Histogram | 11 | 0.961 | 0.696 |
| ALL21，DL/DS/DP → DEC_SUM | 19 | 0.935 | 0.512 |
| ALL21，AC/DL/DS/DP → ACC_DEC | 18 | 0.935 | 0.500 |

這張表的使用限制：

1. **它用了訓練集的異常標籤來評分。** 依它排序假說，是「看過訓練標籤」的決定，不能描述成完全無監督的特徵選擇。若它影響了 P3 正式的特徵集合，報告必須揭露。
2. 重複間 PR-AUC 的 SD 約 0.06。PRUNED17 對 ALL21 的 +0.018 **既不能證明變好，也不能證明等效**。
3. 這個 SD 只反映 fold 切法不同，不是跨檢查／跨孕婦的泛化不確定性。PR-AUC 由各折分數合併後計算，也可能受各折分數尺度不一致影響。
4. 它是線性高斯探針，不是 P3 的 Autoencoder；數字不可當成模型成績，也不可放進 §7–§8。
5. Mahalanobis 對「加入既有特徵的線性組合」不變，所以「ALL21 + 衍生特徵」在此探針下本來就不會有差異（CSV 內有，表中省略）。

---

## H1：FHR Variability Features

1. **研究問題**：ASTV、MSTV、ALTV、MLTV 是否提供事件與直方圖特徵補不上的偵測資訊？
2. **Dataset Observation（Verified，train）**
   - ASTV：KS 0.71（exam-bootstrap 95% CI 0.56–0.83），中位數 40→65；拿掉兩個最大 Pathologic exam 後仍有 0.66。
   - MLTV：KS 0.54（0.44–0.67）；Pathologic 51% 為 0，Normal 4%。
   - MSTV：雙峰，單向 AUC 0.59、|z| AUC 0.80。ALTV：整體 KS 0.22；DP-absent 子群中位數 69。
   - 四個特徵彼此不冗餘：ASTV–MSTV ρ=−0.36、ALTV–MLTV ρ=−0.02、ASTV–ALTV ρ=0.23。
3. **External Evidence**：Huang et al. 2021 報告 MSTV、MLTV、ASTV 與病理的 Spearman 相關（0.5、0.39、0.34，待對照原文）；UH capstone 2025 的 SHAP 前十包含 ASTV、MSTV、MLTV，同時指出 MSTV 在 Normal–Pathologic 之間差異不顯著，與我們看到的非單調現象相符。
4. **Feature Design**：不新增特徵。比較 `ALL21` 與 `NO_VARIABILITY17 = ALL21 − {ASTV, MSTV, ALTV, MLTV}`。
5. **假設**
   - H0：移除變異度群後，驗證集 PR-AUC 的下降不超過 seed 間的變動。
   - H1：移除後 PR-AUC 下降，且下降主要出現在 DP-absent pathologic segments 的召回率。
6. **P5 Controlled Ablation**：A1。
7. **預期方向與可能失敗原因**：預期下降。可能失敗：直方圖的 Width／Variance 接近 0 也能反映曲線很平（DP-absent 子群 Width 中位數 18、Variance 0），Autoencoder 可能從這些特徵補上缺口。若如此，結論應寫成「變異度群的資訊可被直方圖特徵部分取代」。
8. **不主張的事**：不主張這四個特徵對任何模型都「必要」，也不主張結論可推廣到其他資料集。

## H2′：Acceleration／Deceleration Features（補充，負面結果）

1. **研究問題**：AC、DL、DS、DP 的衍生組合是否比個別欄位更能反映病理異常？
2. **衍生特徵定義**（公式本身不使用標籤；單位為次／秒）
   - `DEC_SUM = DL + DS + DP`：每秒減速總次數。
   - `DEC_SEVERE = DS + DP`：每秒重度與延長減速次數。
   - `ACC_DEC = AC − (DL + DS + DP)`：每秒加速次數減去減速次數。
   - 這些是我們自己的設計，沒有找到外部先例。
3. **Dataset Observation（Verified，train）**
   - DP 出現率 Normal 4.7%、Pathologic 62.5%（CI 31–80%）；AC 出現率 67% 對 19%；DL 在 Normal 就有 45%（KS 0.25）；DS 全 train 只有 7 筆非 0。
   - `DEC_SUM` 的 KS 0.40，低於 DP 單獨的 0.60：DL 在正常片段很常見，加總後稀釋了 DP。
   - `DEC_SEVERE` 0.64（0.36–0.80）與 DP（0.29–0.78）無法區分；`ACC_DEC` 0.53（0.46–0.65）。
   - **穩健性警告**：DP 的 KS 在拿掉兩個最大 Pathologic exam 後降到 0.41。
4. **External Evidence**：Huang et al. 2021 的關聯規則包含 DP>0.0028/s、DS>0.00094/s 對應病理。
5. **假設**
   - H0：把 DL、DS、DP 換成 `DEC_SUM` 後，PR-AUC 不低於 ALL21。
   - H1：換成加總後 PR-AUC 下降。
6. **P5 Controlled Ablation**：A3（選做 A2、A4）。
7. **可能失敗原因**：非線性模型可能從加總特徵學到不同的東西，結果可能與 EDA 的單變數判讀不同；若如此應如實報告。

## H3：Histogram Redundancy

1. **研究問題**：依正常資料的相關性挑掉重複的直方圖特徵後，Autoencoder 能否維持相近的偵測表現？
2. **Dataset Observation（Verified，train）**
   - `Width = Max − Min` 在 train 每一列都完全成立（殘差 0），21 個特徵的共變異數矩陣秩為 20。
   - Spearman：Mode–Median 0.97、Mean–Median 0.94、Mode–Mean 0.90、Width–Min −0.89、Width–Variance 0.82；跨群組 LB–Median 0.81。
   - VIF（Normal-train）：Median 34.5、Mean 18.3、Mode 14.4、LB 9.0。
3. **External Evidence**：Huang et al. 2021 列出共線群 {LB, Mode, Median, Mean} 與 {Width, Variance, Nmax, MSTV, Max, DL}。
4. **Feature Design**（`EDA/outputs/histogram_pruning.json`）

   規則（只用 Normal-train，不使用異常標籤；門檻 0.8 事先固定、沒有調過）：
   1. 先移除 Width（Width = Max − Min，確定性冗餘）。
   2. 計算其餘 9 個直方圖特徵之間的 |Spearman|。
   3. Complete linkage 分群，同群內任兩個特徵 |ρ| ≥ 0.8。
   4. 每群保留 medoid（與同群其他成員平均 |ρ| 最高者；平手取欄位順序在前者）。

   結果：
   - {Mode, Mean, Median} → 保留 Median。
   - Min、Max、Variance、Nmax、Nzeros、Tendency 各自成群，全部保留。
   - `PRUNED18 = ALL21 − {Width, Mode, Mean}`（H3 的正式集合）。

   **兩種移除的性質不同，必須分開說：**

   | 移除 | 性質 | 資訊損失 |
   |---|---|---|
   | 只移除 Width（`DROP_WIDTH20`） | 確定性冗餘：Width = Max − Min | 無。可由保留的 Max、Min 精確算回 |
   | 再移除 Mode、Mean（`PRUNED18`） | 高度相關的近似重複 | 有，但小：保留的 18 個特徵可線性解釋其變異的 93%、94%（Normal-train） |

   所以 PRUNED18 是**依冗餘程度做的啟發式特徵挑選**，不是無損精簡。它是否影響偵測表現，只能由實驗回答。

   **選做對照 `PRUNED17`**（`ALL21 − {Width, Min, Mode, Mean}`）：這是較早的規則順序（先分群、最後才移除 Width）產生的結果。{Width, Min} 一群的代表是 Width，Width 被移除後整群沒有留下成員，Min 也跟著被移除。Width 不在時 Min 無法由 Max 還原（Normal-train 中 Min–Max ρ=−0.25），保留的 17 個特徵只能線性解釋 Min 變異的 80%。它**不是主要消融，也不得宣稱無資訊損失**。

5. **假設**
   - H0：PRUNED18 的驗證集 PR-AUC 與 ALL21 的差距在 seed 變動範圍內。
   - H1：PRUNED18 明顯較差，代表被移除的 Mean／Mode 帶有模型用得到的資訊。
   - 注意：「差距在 seed 變動範圍內」只能寫成「未觀察到差異」，不能寫成「證明等效」。
6. **P5 Controlled Ablation**：`ALL21`、`DROP_WIDTH20`、`PRUNED18` 三組一起比（狀態：Pending P5 Verification）。
   - DROP_WIDTH20 對 ALL21：檢驗移除確定性冗餘是否完全無影響（理論上對線性模型無影響，對 Autoencoder 應該也很小）。
   - PRUNED18 對 DROP_WIDTH20：檢驗額外移除 Mode、Mean 的代價。
7. **可能失敗原因**：非線性模型可能從加總特徵學到不同的東西，結果可能與 EDA 的單變數判讀不同；若如此應如實報告。

## H3：Histogram Redundancy

1. **研究問題**：依正常資料的相關性挑掉重複的直方圖特徵後，Autoencoder 能否維持相近的偵測表現？
2. **Dataset Observation（Verified，train）**
   - `Width = Max − Min` 在 train 每一列都完全成立（殘差 0），21 個特徵的共變異數矩陣秩為 20。
   - Spearman：Mode–Median 0.97、Mean–Median 0.94、Mode–Mean 0.90、Width–Min −0.89、Width–Variance 0.82；跨群組 LB–Median 0.81。
   - VIF（Normal-train）：Median 34.5、Mean 18.3、Mode 14.4、LB 9.0。
3. **External Evidence**：Huang et al. 2021 列出共線群 {LB, Mode, Median, Mean} 與 {Width, Variance, Nmax, MSTV, Max, DL}。
4. **Feature Design**（`EDA/outputs/histogram_pruning.json`）

   規則（只用 Normal-train，不使用異常標籤；門檻 0.8 事先固定、沒有調過）：
   1. 計算 10 個直方圖特徵之間的 |Spearman|。
   2. Complete linkage 分群，同群內任兩個特徵 |ρ| ≥ 0.8。
   3. 每群保留 medoid（與同群其他成員平均 |ρ| 最高者；平手取欄位順序在前者）。
   4. Width 一律移除。

   結果：
   - {Mode, Mean, Median} → 保留 Median。
   - {Width, Min} → 平手規則選了 Width 當代表，第 4 步又把 Width 移除，所以**這一群沒有留下任何成員**。
   - `PRUNED17 = ALL21 − {Width, Min, Mode, Mean}`。

   **這三種移除的性質不同，必須分開說：**

   | 移除 | 性質 | 資訊損失 |
   |---|---|---|
   | 只移除 Width（`DROP_WIDTH20`） | 確定性冗餘：Width = Max − Min | 無。可由保留的 Max、Min 精確算回 |
   | 再移除 Mode、Mean | 高度相關的近似重複 | 有，但小：保留的 17 個特徵可線性解釋其變異的 93%、94% |
   | 再移除 Min | 規則步驟順序造成的結果 | **有**：Width 已不在，Min 無法由 Max 還原（Normal-train 中 Min–Max ρ=−0.25）。保留的 17 個特徵只能線性解釋 Min 變異的 80% |

   所以 PRUNED17 是**依冗餘程度做的啟發式特徵挑選**，不是無損精簡。它是否影響偵測表現，只能由實驗回答。

   已知的規則缺陷（留給審查決定，本輪不改實驗設定）：如果先移除 Width 再分群，Min 會自成一群而被保留，得到 18 個特徵（移除 Width、Mode、Mean）。這個版本每一群都保留代表，敘述上比較乾淨。是否改用，見文末討論題。

5. **假設**
   - H0：PRUNED17 的驗證集 PR-AUC 與 ALL21 的差距在 seed 變動範圍內。
   - H1：PRUNED17 明顯較差，代表被移除的 Min／Mean／Mode 帶有模型用得到的資訊。
   - 注意：「差距在 seed 變動範圍內」只能寫成「未觀察到差異」，不能寫成「證明等效」。
6. **P5 Controlled Ablation**：A0（ALL21）、A5（DROP_WIDTH20）、A6（PRUNED17）三組一起比。
   - A5 對 A0：檢驗移除確定性冗餘是否完全無影響（理論上對線性模型無影響，對 Autoencoder 應該也很小）。
   - A6 對 A5：檢驗額外移除 Min、Mode、Mean 的代價。
7. **可能失敗原因**：Mean（單變數 KS 0.58）在 DP-present 子群是明顯的訊號。若 PRUNED18 變差而 DROP_WIDTH20 不變，結論是「只有確定性冗餘可以安全移除」。

---

## P5 Handoff Table

**課程要求提醒**：P5 至少 2 個 controlled ablations，其中**至少 1 個是異常偵測專屬**（threshold／scoring／contamination 等）。下表全部是 Feature Ablation，**不能取代**異常偵測專屬的那一個。建議的最小組合：一組特徵消融（A0＋A1，或 A0＋A5＋A6，A6 為 PRUNED18）加一組 scoring-function 消融（由 P3／P5 設計）。

**共同規則**
- 資料：P1 的固定切分。模型：P3 的 Autoencoder 與超參數固定；每個設定用同一組 seeds，報告平均 ± SD。
- 前處理：z-score 只用 Normal-train fit；每個特徵集合各自重新 fit。
- 一次只改一個因素：每一列與 A0 只差特徵集合。
- 比較只看 **validation**（PR-AUC 為主、AUROC 為輔；F1 需使用同一種門檻規則）。Test 在設定凍結後才評估一次，不可用來挑特徵集合。
- 驗證集的 Pathologic 只來自 9 個 exam。seed 間的 SD 不包含這個不確定性；小於 SD 的差距請寫成「未觀察到差異」。
- 特徵消融與 threshold／scoring 消融分成兩張表。

| ID | 假說 | 特徵集合名稱 | 內容 | 預期（相對 A0） | 何種結果算不支持 |
|---|---|---|---|---|---|
| A0 | baseline | `ALL21` | `load_data.FEATURES` | — | — |
| A1 | H1 | `NO_VARIABILITY17` | 移除 ASTV、MSTV、ALTV、MLTV | PR-AUC 下降 | 下降幅度小於 seed SD |
| A5 | H3 | `DROP_WIDTH20` | 移除 Width | 無可辨別差異 | 明顯下降 |
| A6 | H3 | `PRUNED18` | 移除 Width、Mode、Mean | 待觀察；預期差距在 seed SD 內 | 下降超過 seed SD |
| A6b（選做） | H3 對照 | `PRUNED17` | 移除 Width、Min、Mode、Mean | 僅作對照，不是主結果 | — |
| A3（選做） | H2′ | `DEC_SUM19` | DL、DS、DP 換成 `DEC_SUM` | 持平或下降 | 明顯上升 |
| A2（選做） | H2′ | `NO_ACCDEC17` | 移除 AC、DL、DS、DP | PR-AUC 下降 | 下降幅度小於 seed SD |
| A4、A7（選做） | — | `ALL21 + ACC_DEC`；移除全部直方圖特徵 | 探索用 | — | — |

優先順序：A0、A1 必做；A5、A6 建議做；其餘視時間。

**建議的附加分析（不是消融）**：事先固定把驗證集的 Pathologic 依 `DP > 0` 分成 DP-present／DP-absent，分別報告召回率。若 A1 的損失集中在 DP-absent，就直接對應 H1。子群筆數很小時請一併報告筆數與 exam 數。

**欄位名稱**：全部沿用 `load_data.FEATURES`。衍生特徵的程式在 `EDA/p2_eda.py::add_derived`。

## 尚待審查的討論題

1. （已決定，R5）H3 採用「先移除 Width 再分群」的 PRUNED18；PRUNED17 降為選做對照。Feature Set 定義仍待 P3／P5 確認。
2. 驗證集只有 9 個 Pathologic exam，P5 如何呈現不確定性（例如對驗證集做 exam-level bootstrap）？
