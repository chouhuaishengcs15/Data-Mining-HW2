# P2 Feature Hypotheses 與 P5 Handoff

- 撰寫：P2（Claude Code Fable），2026-11-10
- 證據：`EDA/outputs/*.csv`（全部只用 train；`python EDA/p2_eda.py` 可重現）
- 狀態：三個都是 **Hypothesis**，尚未經 P5 的正式消融驗證。

## 0. 先講結論

| ID | 假說 | EDA 支持程度 | 建議 |
|---|---|---|---|
| **H1** | FHR 變異度特徵群（ASTV、MSTV、ALTV、MLTV）是必要的 | 支持 | **保留，主要假說** |
| **H2** | 加速／減速特徵的**衍生組合**比個別欄位更好 | **不支持**衍生組合；但這個特徵群本身很重要 | 改寫成 H2′（保留原始 AC、DP，不要加總），列為次要 |
| **H3** | 移除冗餘的 Histogram 特徵不會明顯降低效能 | 支持 | **保留，主要假說** |

我推薦報告正式寫 **H1 與 H3**：一個是「拿掉會變差」，一個是「拿掉不會變差」，方向相反，P5 的消融表能同時呈現特徵的必要性與精簡性。H2′ 只需要在同一張消融表多一列，成本很低，建議一併做，並如實記錄衍生特徵的負面結果。

### 支撐三個假說的共同觀察：Pathologic 不是單一族群

用不需要標籤的規則（`DP > 0` 與否）把 train 的 104 筆 Pathologic 分成兩群（`EDA/outputs/pathologic_phenotypes.csv`、`fig5`）：

| 中位數 | Normal（972 筆） | Pathologic「平坦型」DP=0（39 筆／18 exams） | Pathologic「減速型」DP>0（65 筆／12 exams） |
|---|---|---|---|
| ASTV | 40 | 71 | 62 |
| MSTV | 1.2 | **0.3** | **2.7** |
| ALTV | 0 | **69** | 0 |
| MLTV | 8.1 | 3.9 | 0 |
| Width | 68 | **18** | **113** |
| Mean | 134 | 135 | **96** |
| Variance | 8 | **0** | **57** |
| ASTV ≥ 60 的比例 | 16% | 95% | 83% |

- 兩群在 MSTV、ALTV、Width、Variance 上往**相反方向**偏離正常；只有 ASTV 兩群都升高。這解釋了為什麼 ASTV 是最強的單一特徵，以及為什麼 MSTV 的單向 AUC 只有 0.59，但雙向（|z|）AUC 有 0.80。
- 意涵：單一方向的特徵重要度（相關係數、單向 AUC）會低估雙峰特徵；one-class 模型衡量的是「離正常多遠」，比較貼近 |z| 的觀點。
- 限制：「平坦型／減速型」是我們依 `DP > 0` 做的描述性分群，不是臨床分型；減速型的 65 筆只來自 12 個 exam。

### Group probe（探索性，不是 P3 的模型）

做法：只在 train 內，用 exam 分組的 5-fold CV（重複 5 次）；每一折只用 Normal 估計 z-score 與 Ledoit–Wolf 共變異數，對 held-out exam 算 Mahalanobis 距離。Pathologic 基率 0.097。（`EDA/outputs/group_probe.csv`）

| 特徵集合 | 特徵數 | AUROC | PR-AUC（±重複間 SD） |
|---|---|---|---|
| ALL21 | 21 | 0.947 | 0.647 ± 0.064 |
| 只有變異度 | 4 | 0.862 | 0.372 |
| 只有 AC、DL、DS、DP | 4 | 0.850 | 0.681 |
| 只有 Histogram | 10 | 0.803 | 0.420 |
| ALL21 − 變異度 | 17 | **0.853** | 0.565 |
| ALL21 − AC/DL/DS/DP | 17 | 0.934 | **0.480** |
| ALL21 − Histogram | 11 | 0.961 | 0.696 |
| ALL21 − Width | 20 | 0.948 | 0.649 |
| ALL21，Histogram 剪枝後 | 17 | 0.954 | 0.665 |
| ALL21，DL/DS/DP → DEC_SUM | 19 | 0.935 | 0.512 |
| ALL21，AC/DL/DS/DP → ACC_DEC | 18 | 0.935 | 0.500 |

解讀時的注意事項：
- PR-AUC 在重複之間的 SD 約 0.06，小於這個幅度的差異不要解讀。
- 這是線性高斯探針，Isolation Forest／Autoencoder 的結果可能不同，所以才需要 P5 驗證。
- Mahalanobis 對「加入既有特徵的線性組合」不變，所以「ALL21 + DEC_SUM」這類列在此探針下**本來就不會有差異**（已從上表省略，CSV 內有）。

---

## H1：FHR Variability Features

1. **研究問題**：ASTV、MSTV、ALTV、MLTV 是否構成不可被其他特徵取代的辨識資訊？
2. **Dataset Observation（Verified，train）**
   - ASTV：KS 0.71（exam-bootstrap 95% CI 0.56–0.84），中位數 40→65，拿掉兩個最大 Pathologic exam 後仍有 0.66，是最穩定的單一特徵。
   - MLTV：KS 0.54（0.44–0.68）；Pathologic 51% 為 0，Normal 只有 4%。
   - MSTV：雙峰，單向 AUC 0.59、|z| AUC 0.80。ALTV：整體 KS 0.22，但平坦型中位數 69。
   - 四個特徵彼此不冗餘：ASTV–MSTV ρ=−0.36、ALTV–MLTV ρ=−0.02、ASTV–ALTV ρ=0.23。
   - Probe：拿掉這四個特徵，AUROC 0.947→0.853，是所有 drop-group 中 AUROC 掉最多的。
3. **External Evidence**：Huang et al. 2021 指出 MSTV、MLTV、ASTV 與病理相關；UH capstone 2025 的 SHAP 前十包含 ASTV、MSTV、MLTV。反例：同一份海報指出 MSTV 在 Normal–Pathologic 之間的 SHAP 差異不顯著，這與我們觀察到的「雙峰、非單調」相符。
4. **Feature Design**：不新增特徵。比較 `ALL21` 與 `ALL21 − {ASTV, MSTV, ALTV, MLTV}`。
5. **假設**
   - H0：移除變異度群後，驗證集 PR-AUC 的下降不超過 seed 間的變動。
   - H1：移除後 PR-AUC 與 AUROC 明顯下降，且下降集中在「平坦型」（DP=0）Pathologic 的召回率。
6. **P5 Controlled Ablation**：見下方 Handoff Table 的 A1。
7. **預期方向與可能失敗原因**：預期下降。可能失敗：Histogram 的 Width／Variance 接近 0 也能代表「平坦」（平坦型 Width 中位數 18、Variance 0），在非線性模型中可能補上缺口，使下降幅度比探針小。

## H2：Acceleration／Deceleration Features

1. **研究問題**：AC、DL、DS、DP 的衍生組合是否比個別欄位更能反映病理異常？
2. **衍生特徵定義**（皆不需標籤；單位為次／秒）
   - `DEC_SUM = DL + DS + DP`：每秒減速總次數，不分類型。
   - `DEC_SEVERE = DS + DP`：每秒重度與延長減速次數。
   - `ACC_DEC = AC − (DL + DS + DP)`：每秒加速次數減去減速次數；正值代表以加速為主。
   - 臨床合理性：加速通常代表反應性良好，減速代表壓力，把兩者相減是直覺的「反應性淨值」。這是我們自己的設計，沒有外部先例（見 `p2_prior_work.md`）。
3. **Dataset Observation（Verified，train）**
   - 個別欄位：DP 出現率 Normal 4.7%、Pathologic 62.5%（CI 31–80%）；AC 出現率 67% 對 19%（CI 1–38%）；DL 在 Normal 就有 45%，KS 只有 0.25；DS 全 train 只有 7 筆非 0。
   - **DEC_SUM 比 DP 差**：KS 0.40 對 0.60。原因是 DL 在正常片段很常見，加總後把 DP 的訊號稀釋了。
   - DEC_SEVERE：KS 0.64（0.36–0.80），和 DP（0.29–0.78）的區間幾乎完全重疊，沒有可辨別的增益（DS 太稀少）。
   - ACC_DEC：KS 0.53（0.46–0.65），沒有高於 DP 或 ASTV，但區間較窄，拿掉兩個最大 exam 後不變（0.54）。
   - Probe：用 DEC_SUM 或 ACC_DEC **取代**原始欄位，PR-AUC 從 0.647 降到 0.51／0.50。
   - 但整個群組很重要：拿掉 AC/DL/DS/DP，PR-AUC 0.647→0.480，是所有 drop-group 中 PR-AUC 掉最多的；只用這四個特徵 PR-AUC 就有 0.681。
   - **穩健性警告**：DP 的 KS 在拿掉兩個最大 Pathologic exam 後從 0.60 降到 0.41；減速型只來自 12 個 exam。
4. **External Evidence**：Huang et al. 2021 的病理規則包含 DP>0.0028/s、DS>0.00094/s，AC 與正常關聯最強。沒有找到任何使用減速加總或加減速對比的先例。
5. **假設（改寫為 H2′）**
   - H0：把 DL、DS、DP 換成 `DEC_SUM`（或把四個換成 `ACC_DEC`）後，PR-AUC 不低於 ALL21。
   - H1：換成加總後 PR-AUC 下降，因為減速類型（特別是 DP）本身帶有資訊。
6. **P5 Controlled Ablation**：A2（drop-group）、A3（取代）、A4（附加，僅對 Isolation Forest 這類非線性不變的模型有意義）。
7. **預期方向與可能失敗原因**：預期「取代」會變差、「附加」沒有差異。可能失敗：Isolation Forest 是軸對齊切割，`ACC_DEC` 這種斜向組合理論上可能幫上忙，結果可能與線性探針不同；若如此應如實報告。

## H3：Histogram Redundancy

1. **研究問題**：移除高度冗餘的 Histogram 特徵，能否減少特徵數而不明顯損失偵測效能？
2. **Dataset Observation（Verified，train）**
   - `Width = Max − Min` 在 train 每一列都**完全成立**（殘差 0），21 個特徵的共變異數矩陣秩為 20。
   - Spearman：Mode–Median 0.97、Mean–Median 0.94、Mode–Mean 0.90、Width–Min −0.89、Width–Variance 0.82；跨群組 LB–Median 0.81。
   - VIF（Normal-train）：Median 34.5、Mean 18.3、Mode 14.4、LB 9.0；Width／Min／Max 因完全共線而發散。
   - Probe：拿掉 Width 沒有變化（0.948／0.649）；剪枝到 17 個特徵為 0.954／0.665；拿掉全部 10 個 Histogram 特徵為 0.961／0.696。三者都沒有比 ALL21 差。
3. **External Evidence**：Huang et al. 2021 列出共線群 {LB, Mode, Median, Mean} 與 {Width, Variance, Nmax, MSTV, Max, DL}，與我們的群集一致。
4. **Feature Design：不看標籤的剪枝規則**（`EDA/outputs/histogram_pruning.json`）
   - 只用 **Normal-train** 計算 Histogram 特徵間的 |Spearman|；complete linkage 分群，門檻 |ρ| ≥ 0.8；每群留 medoid（與同群其他成員平均 |ρ| 最高者）；Width 因完全共線一律移除。
   - 結果：群集 {Mode, Mean, Median}→留 **Median**；{Width, Min}→都移除（Width 是確定性組合，Min 可由 Width 與 Max 還原）。
   - `PRUNED17` ＝ 11 個非 Histogram 特徵 ＋ {Max, Nmax, Nzeros, Median, Variance, Tendency}。
   - 因為規則不使用異常標籤，採用 PRUNED17 不會破壞「模型流程不使用異常標籤」的說法。門檻 0.8 是事先固定的慣例值，沒有調過。
5. **假設**
   - H0：PRUNED17 的驗證集 PR-AUC 與 ALL21 的差距在 seed 變動範圍內（或更高）。
   - H1：PRUNED17 明顯較差，代表被移除的 Mean／Min／Mode 帶有模型用得到的資訊。
6. **P5 Controlled Ablation**：A5、A6、A7。
7. **預期方向與可能失敗原因**：預期無損失。可能失敗：被移除的 Mean（單變數 KS 0.58）與 Min（0.40）是減速型的強訊號，對於無法從 Median、DP 組合出同樣資訊的模型（例如樹深有限的 Isolation Forest），可能會有損失。若 PRUNED17 變差但 ALL21−Width 不變，結論應改為「只有確定性冗餘可以安全移除」。

---

## P5 Handoff Table

**共同規則（所有列都適用）**
- 資料：P1 的固定切分，不重切。模型：P3 的主要模型與超參數**固定不變**；每個設定跑相同的一組 seeds（建議 ≥ 5），報告平均 ± SD。
- 前處理：z-score 只用 Normal-train fit；每個特徵集合各自重新 fit。
- 一次只改一個因素：下表每一列與 baseline（ALL21）只差「特徵集合」。
- 選擇與比較只看 **validation**（PR-AUC 為主、AUROC 為輔；F1 需固定門檻規則，例如固定用 ALL21 在 val 上選的同一種規則）。**Test 只在最後對事先指定的設定各評估一次，不可用 test 回頭挑特徵集合。**
- 這些是 **Feature Ablation**。門檻 τ 的選法、異常分數的定義、contamination 等屬於 **Threshold／Scoring Ablation**，請另列一張表，不要和特徵消融混在同一個比較裡。

| ID | 對應假說 | 特徵集合 | 與 baseline 的唯一差異 | 預期（相對 ALL21） | 何種結果算「不支持」 |
|---|---|---|---|---|---|
| A0 | baseline | `ALL21` | — | — | — |
| A1 | H1 | `ALL21 − {ASTV, MSTV, ALTV, MLTV}`（17） | 移除變異度群 | PR-AUC、AUROC 下降 | 下降幅度小於 seed SD |
| A2 | H2′ | `ALL21 − {AC, DL, DS, DP}`（17） | 移除加減速群 | PR-AUC 下降 | 同上 |
| A3 | H2′ | `ALL21`，以 `DEC_SUM` 取代 DL、DS、DP（19） | 減速類型合併 | PR-AUC 下降或持平 | PR-AUC 明顯上升（代表加總有益，推翻我們的 EDA 判讀） |
| A4（選做） | H2′ | `ALL21 + ACC_DEC`（22） | 附加一個衍生特徵 | 無差異 | 明顯上升（代表斜向組合對該模型有益） |
| A5 | H3 | `ALL21 − {Width}`（20） | 移除確定性冗餘 | 無差異 | 明顯下降 |
| A6 | H3 | `PRUNED17`：`ALL21 − {Width, Min, Mode, Mean}` | 不看標籤的冗餘剪枝 | 無差異或略升 | 下降超過 seed SD |
| A7（選做） | H3 | `ALL21 − 全部 10 個 Histogram`（11） | 移除整個 Histogram 群 | 無差異或略升 | 明顯下降 |

**建議的附加分析（不是消融）**：在 P5 的錯誤分析中，把 Pathologic 依 `DP > 0` 分成兩群，分別報告召回率。若 A1 的損失集中在 DP=0、A2 的損失集中在 DP>0，就直接驗證了「兩種表現型由兩個特徵群分別偵測」的說法。

**欄位名稱**：全部沿用 `load_data.FEATURES`。衍生特徵的程式在 `EDA/p2_eda.py::add_derived`。
