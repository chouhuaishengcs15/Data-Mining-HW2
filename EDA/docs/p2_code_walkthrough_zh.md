# `EDA/p2_eda.py` 程式導讀

- 先讀 `p2_eli5_learning_path_zh.md` 的第 1、2 章再讀這份，會容易很多。
- 這份文件照**程式實際執行的順序**走，不是照檔案由上到下的順序。
- 不需要一次讀完。建議分三次：第一部分（讀資料、統計、圖）→ 第二部分（Bootstrap）→ 第三部分（冗餘、剪枝、探針）。
- 執行方式（在 repo 根目錄）：

```bash
pip install -r EDA/requirements.txt
python EDA/p2_eda.py        # 約 1 分鐘；表格寫到 EDA/outputs，圖寫到 EDA/figures
```

## 全貌：`run()` 做了哪些事

程式的入口是檔案最下面的 `run()`。它依序呼叫下面這些函式：

| 順序 | 函式 | 一句話 | 輸出檔 | 對應 |
|---|---|---|---|---|
| 1 | `load_train` | 讀訓練集 | — | 全部 |
| 2 | `add_derived` | 加三個衍生欄位 | — | H2′ |
| 3 | `audit` | 檢查資料品質 | `audit_train.json` | §4 開頭、Dataset Brief |
| 4 | `separability` | 每個特徵算 KS、AUC、MI 等 | `separability.csv` | Table 1、Figure 3 |
| 5 | `exam_bootstrap` | 以檢查為單位算信賴區間 | 併入 `separability.csv` | Table 1 的 [95% CI] |
| 6 | `row_bootstrap_ks`、`leave_top_exams_out` | 兩個穩健性對照 | 併入 `separability.csv` | §4 Feature separability |
| 7 | `presence_table` | 事件的出現率 | `event_presence.csv` | Figure 2 |
| 8 | `phenotype_table` | 病理依 DP 分兩個子群的中位數 | `pathologic_phenotypes.csv` | Figure 5 |
| 9 | `redundancy` | 相關係數與 VIF | `spearman_*.csv`、`redundancy_pairs.csv`、`vif_normal_train.csv` | Figure 4、§4 Redundancy |
| 10 | `prune_histogram` | 挑掉重複的直方圖特徵 | `histogram_pruning.json` | H3 |
| 11 | `group_probe` | 訓練集內的探索性探針 | `group_probe.csv` | 只供內部挑選假說，不進 §4 |
| 12 | `fig_*` | 畫六張圖 | `EDA/figures/` | Figure 1–5、S1 |

---

# 第一部分：讀資料、統計、圖

## 1. `load_train()`

| | |
|---|---|
| 輸入 | 沒有參數。內部呼叫 P1 的 `load_split("train", return_exam=True)` |
| 計算 | 把 21 個特徵、`label`、`exam` 合成一張表 |
| 輸出 | 1076 列 × 23 欄的 DataFrame（972 筆 label 0、104 筆 label 1） |
| 為什麼需要 | 這是整支程式**唯一**讀資料的地方。只寫了 `"train"`，所以驗證集和測試集不可能被讀到 |
| 對應 | §4 第一句 “All analyses in this section use only the training portion” |

重點：這裡讀的是**完整的 train（含病理）**，因為 EDA 需要比較兩群。P3 訓練 one-class 模型時用的是 `use_normal_only=True`，只有 972 筆正常。兩種用法不同。

## 2. `add_derived(df)`

| | |
|---|---|
| 輸入 | 上一步的表 |
| 計算 | 加三欄：`DEC_SUM = DL + DS + DP`、`DEC_SEVERE = DS + DP`、`ACC_DEC = AC − (DL + DS + DP)` |
| 輸出 | 多三欄的表 |
| 為什麼需要 | 檢驗「把減速加總會不會比個別欄位更好」。結果是不會（`DEC_SUM` 的 KS 0.40，低於 DP 單獨的 0.60） |
| 對應 | H2′；Figure 2 (c) 的 X 軸 |

小例子：某片段 AC = 0.003、DL = 0.002、DS = 0、DP = 0.001，則 `DEC_SUM` = 0.003，`ACC_DEC` = 0.003 − 0.003 = 0。

## 3. `audit(df)`

| | |
|---|---|
| 輸入 | 訓練集的表 |
| 計算 | 數筆數、缺值、重複列、每個特徵有幾種值、0 佔多少、`Width` 是否等於 `Max − Min`、病理片段集中在哪幾次檢查 |
| 輸出 | 一個字典，存成 `audit_train.json` |
| 為什麼需要 | 先確認資料沒有問題，並找出會影響後面統計的事實 |
| 對應 | §4 第一段的 “28 examinations, two contribute 36”；§4 Redundancy 的 “Width = Max − Min holds exactly” |

兩個影響後面所有分析的發現：
- 104 筆病理只來自 28 次檢查，其中兩次佔 36 筆 → 第二部分要用「以檢查為單位」的 Bootstrap。
- `Width == Max − Min` 每一列都成立 → 第三部分的冗餘。

## 4. `separability(df, cols)`：這支程式的核心

| | |
|---|---|
| 輸入 | 訓練集的表；要分析的欄位（21 個特徵＋3 個衍生欄位） |
| 計算 | 對每個欄位，把 Normal 的值和 Pathologic 的值分開，各算一組統計量 |
| 輸出 | 一張表，一列一個特徵 |
| 為什麼需要 | 評分要求「量化特徵可分性」 |
| 對應 | Table 1、Figure 3、`separability.csv` |

函式裡的主要迴圈是 `for j, c in enumerate(cols)`。每一圈處理一個特徵：`a` 是 972 個正常值，`b` 是 104 個病理值。然後算出下面這些欄位。

### 4.1 描述統計：`normal_median`、`path_median`、`*_IQR`、`*_zero_frac`

- median（中位數）：排隊後正中間的值。
- IQR：第 75 百分位減第 25 百分位，也就是「中間一半資料」的寬度。
- zero_frac：剛好等於 0 的比例。

用中位數和 IQR 而不用平均和標準差，是因為很多特徵嚴重偏態或大量為 0，平均會被少數極端值拉走（學習路徑 2.5 節的 ALTV 例子）。

### 4.2 `KS`

程式：`ks_2samp(a, b).statistic`。意思與手算方式見學習路徑 2.4 節：兩條 ECDF 的最大垂直距離。

程式只取 `.statistic`，**刻意不用 p-value**。p-value 假設資料連續且彼此獨立，這兩個假設在這裡都不成立（很多特徵有大量相同的值；同一次檢查的片段彼此相似）。不確定性改由第二部分的 Bootstrap 處理。

### 4.3 `AUC_path_gt_normal`（由 `auc_path_gt_normal()` 計算）

問題：隨機抽一筆病理、一筆正常，病理的值比較大的機率是多少？相等算半次。

小例子：

```
Normal：1, 2, 3        Pathologic：2, 5
```

六種配對：(2,1) 贏、(2,2) 平、(2,3) 輸、(5,1) 贏、(5,2) 贏、(5,3) 贏。
AUC =（4 + 0.5）÷ 6 = 0.75。

怎麼讀：
- 0.5：亂猜。
- 接近 1：病理的值通常比較大（ASTV 是 0.86）。
- 接近 0：病理的值通常比較**小**（MLTV 是 0.17，同樣是分得開，只是方向相反）。

所以表裡另有一欄 `AUC_sep = max(AUC, 1 − AUC)`，不管方向只看分得多開。

### 4.4 `AUC_abs_dev`

先把每個值換成「離正常平均多少個標準差」（z-score），取絕對值，再算 AUC。它問的是：**離正常越遠的，是不是越可能是病理？**

這比較接近 one-class 模型的想法（模型只知道正常長什麼樣）。MSTV 的 `AUC_path_gt_normal` 是 0.59，`AUC_abs_dev` 是 0.80，差別就在病理往兩頭偏。

注意：這裡的平均和標準差只用 Normal 的 972 筆計算（程式裡的 `mu0`、`sd0`）。

### 4.5 `wasserstein_raw` 與 `wasserstein_normalSD`

Wasserstein distance 可以想成：把一堆沙（一個分布）搬成另一堆沙的形狀，平均每粒沙要搬多遠。

問題是它**帶單位**。ASTV 的原始值是 21.3（百分點），DP 是 0.0015（次／秒），兩個數字不能比。所以程式先除以正常資料的標準差（`wasserstein_normalSD`），單位變成「幾個正常標準差」，才可以跨特徵比較。

### 4.6 `MI_knn_nats` 與 `MI_presence_nats`

Mutual Information：知道這個特徵的值之後，對「是不是病理」的不確定性減少了多少。0 代表完全沒幫助。上限是標籤本身的不確定性（這裡是 0.32 nats，因為病理只佔 9.7%）。

- `MI_knn_nats`：用 scikit-learn 的 `mutual_info_classif`。這個估計方法有隨機成分，所以程式用 20 個不同的 seed 各算一次取平均。`Nmax`、`Nzeros`、`Tendency` 被標成離散變數。
- `MI_presence_nats`：只對事件率特徵計算，把特徵簡化成「有／沒有」（`x > 0`）再算，可以精確計算。

MI 的好處是不假設方向，所以 MSTV 的 MI（0.13）和 ASTV（0.13）差不多，不像 KS 差那麼多。限制是它只表示「有關聯」，不代表因果。

## 5. 畫圖的函式

共用的小工具：
- `_ecdf(ax, x, color, ls)`：把一組數字排序後畫成階梯狀的累積曲線。
- `_save(fig, name)`：同一張圖存成 PNG 和 PDF。

| 函式 | 圖 | 讀圖重點 |
|---|---|---|
| `fig_variability` | Figure 1 | 上排直方圖用 `weights = 1/該類別筆數`，所以 Y 軸是比例；下排 ECDF；黑線是程式找到的最大差距位置 |
| `fig_events` | Figure 2 | (a) 的長條高度與誤差線來自 `presence_table`；(b)(c) 是 ECDF，X 軸乘以 1000 變成「每 1000 秒幾次」方便閱讀 |
| `fig_ranking` | Figure 3 | 每個特徵一個點（KS）加一條橫線（95% 區間），叉號是拿掉兩次最大檢查後的 KS |
| `fig_heatmap` | Figure 4 | 21 × 21 的 Spearman 相關；顏色越深相關越強；特徵順序經過分群排列，相似的排在一起 |
| `fig_all_ecdf` | Figure S1 | 21 個特徵各一張小 ECDF |
| `fig_phenotypes` | Figure 5 | 散佈圖；點加了很小的隨機位移避免重疊，只影響畫面 |

Figure 1 的三個實作細節：
1. `bins = np.linspace(最小值, 最大值, 31)`：格子涵蓋完整範圍，沒有截掉任何資料。
2. `weights=np.full(len(n), 1 / len(n))`：每筆資料的權重是「1 ÷ 該類別筆數」，所以長條加總為 1。這就是處理類別不平衡的方法。
3. 黑線的位置：對每個出現過的值算兩條 ECDF 的差，取絕對值最大的那一個（`np.argmax(np.abs(Fn - Fp))`），和學習路徑 2.4 節手算的做法相同。

## 6. `presence_table(df)` 與 `phenotype_table(df)`

**`presence_table`**

| | |
|---|---|
| 輸入 | 訓練集的表 |
| 計算 | 對 AC、DL、DS、DP、FM、UC，分別算兩個類別中「值 > 0」的比例；再用以檢查為單位的 Bootstrap 算區間 |
| 輸出 | `event_presence.csv` |
| 為什麼需要 | 事件率特徵大量為 0，比「有沒有出現」比比平均值合適 |
| 對應 | Figure 2 (a)；§4 的 “62.5% vs 4.7%” |

**`phenotype_table`**

| | |
|---|---|
| 輸入 | 訓練集的表 |
| 計算 | 把資料分三組：Normal、Pathologic 且 DP > 0、Pathologic 且 DP = 0；每組算各特徵的中位數、筆數、檢查數 |
| 輸出 | `pathologic_phenotypes.csv` |
| 為什麼需要 | 解釋 MSTV 為什麼是雙峰 |
| 對應 | Figure 5；§4 Distribution contrast 的後半段 |

提醒：分組用到了 `label`（先挑出 Pathologic），所以這是描述訓練資料的子群，不是不看標籤的發現。

---

# 第二部分：Bootstrap

## 7. 為什麼需要信賴區間

KS = 0.71 是從「這 1076 筆」算出來的。如果當初收集到的是另一批資料，數字會不同。信賴區間回答：**換一批資料，這個數字大概會落在哪個範圍？**

Bootstrap 的做法是：從手上的資料裡「抽了放回」，重複抽出很多份和原來一樣大的假資料，每一份都算一次 KS，看這些 KS 散得多開。

## 8. `exam_bootstrap(df, cols)`

| | |
|---|---|
| 輸入 | 訓練集的表；欄位清單 |
| 計算 | 重複 2000 次：隨機抽 175 次**檢查**（抽了放回），把抽到的檢查的所有片段合起來，重新算每個特徵的 KS 和 AUC |
| 輸出 | 每個特徵的 `KS_ci_lo`、`KS_ci_hi`（2000 個 KS 的第 2.5 與第 97.5 百分位），AUC 同理 |
| 為什麼需要 | 病理片段集中在少數檢查，以片段為單位會低估不確定性 |
| 對應 | Table 1 的 [95% CI]；Figure 3 的橫線 |

**為什麼抽的單位是「檢查」而不是「片段」**

玩具例子：有 3 次檢查，

```
檢查 A：5 個片段，全部是病理，DP 都 > 0
檢查 B：1 個片段，病理，DP = 0
檢查 C：1 個片段，病理，DP = 0
```

「病理片段中有 DP 的比例」= 5 ÷ 7 = 71%。但這 71% 幾乎全靠檢查 A。

- 以**片段**為單位重抽：每次大概都會抽到好幾個 A 的片段，比例總在 71% 附近，看起來很穩定。
- 以**檢查**為單位重抽：抽到 (A, A, B) 是 10/11 = 91%；抽到 (B, C, C) 是 0%。結果大幅跳動。

第二種才反映事實：我們其實只有 3 次獨立的觀察，不是 7 次。同一次檢查的片段來自同一位孕婦的同一段時間，彼此很像，不能當成獨立的樣本。

**真實資料的對照**

| 特徵 | KS | 以片段重抽的區間 | 以檢查重抽的區間 |
|---|---|---|---|
| ASTV | 0.71 | 0.64–0.77 | 0.56–0.84 |
| DP | 0.60 | 0.51–0.68 | 0.29–0.78 |

以片段重抽的區間窄很多，那是假的自信。DP 的差別特別大，因為有 DP 的病理片段只來自 12 次檢查。

程式細節：`groups = [g for _, g in df.groupby("exam")]` 先把資料依檢查切成 175 塊；每一輪 `rng.integers(0, len(groups), len(groups))` 抽 175 個編號（可重複），再把對應的塊接起來。

## 9. 兩個對照函式

**`row_bootstrap_ks`**：故意用「以片段為單位」的錯誤做法算一次，只為了在報告裡展示它會把區間算得太窄（上表中間那一欄）。

**`leave_top_exams_out(df, cols, k=2)`**：把病理片段最多的兩次檢查整個拿掉，重算 KS。
- ASTV：0.71 → 0.66，幾乎不變，表示這個差異不是靠那兩次檢查撐起來的。
- DP：0.60 → 0.41；Mean：0.58 → 0.37，掉很多，表示這兩個特徵的強度有一部分來自那兩次檢查。

這就是 §4 寫 “ASTV is the most separable and the most stable feature” 以及對 DP、Mean 持保留態度的依據。

---

# 第三部分：冗餘、剪枝、探針

## 10. `redundancy(df)`

| | |
|---|---|
| 輸入 | 訓練集的表 |
| 計算 | (1) 21 個特徵兩兩的 Spearman 與 Pearson 相關；(2) 只用 Normal 再算一次 Spearman；(3) 每個特徵的 VIF；(4) 共變異數矩陣的秩 |
| 輸出 | 相關矩陣、依相關強度排序的配對表、VIF、秩 |
| 為什麼需要 | 評分要求「檢查候選特徵是否重複」 |
| 對應 | Figure 4；§4 Redundancy |

**Spearman 相關是什麼**
先把數值換成名次，再算名次之間的相關。它問的是：一個變大時，另一個是不是也跟著變大（不要求是直線關係）。範圍 −1 到 1。

小例子：

```
X：1, 2, 3, 4        名次 1, 2, 3, 4
Y：1, 4, 9, 100      名次 1, 2, 3, 4
```

名次完全一致，Spearman = 1。Y 不是 X 的直線函數，所以 Pearson 只有約 0.82。用 Spearman 是因為很多特徵偏態、有極端值。

**VIF 是什麼**
用其他 20 個特徵去預測這一個特徵，預測得越準，VIF 越大。VIF = 1 表示完全預測不了（沒有重複）；Median 的 VIF 是 34.5，表示它幾乎可以由其他特徵算出來。Width、Min、Max 的 VIF 趨近無限大，因為 `Width = Max − Min` 是恆等式。

**為什麼要另外只用 Normal 算一次**
one-class 模型只看得到正常資料。要拿來決定「哪些特徵重複」的依據，應該只用模型看得到的資料，才不會間接用到病理標籤。下一步的剪枝就是用這一份。

**主要結果**
- Mode–Median 0.97、Mean–Median 0.94、Mode–Mean 0.90：三個幾乎一樣。
- Width–Min −0.89、Width–Variance 0.82。
- ASTV–MSTV −0.36、ALTV–MLTV −0.02：變異度的四個特徵**不**重複。

## 11. `prune_histogram(sp_norm, thr=0.8)`

| | |
|---|---|
| 輸入 | 只用 Normal 算出的 Spearman 矩陣；門檻 0.8 |
| 計算 | 把 10 個直方圖特徵分群，每群留一個代表，再把 Width 移除 |
| 輸出 | 分群結果、保留與移除的清單（`histogram_pruning.json`） |
| 為什麼需要 | 產生 H3 要測試的 17 個特徵，而且過程不使用病理標籤 |
| 對應 | H3；§4 Redundancy 最後一句 |

**步驟**

1. 把相關換成距離：`距離 = 1 − |相關|`。相關 0.97 → 距離 0.03（很近）；相關 0.2 → 距離 0.8（很遠）。
2. 分群（complete linkage，切在距離 0.2）：同一群裡**任兩個**特徵的 |相關| 都至少 0.8。
3. 每群留「和同群其他成員平均最像」的那一個。
4. Width 一律移除（因為它等於 Max − Min）。

**小例子**：三個特徵的 |相關|

```
        Mode   Mean   Median
Mode     —     0.89    0.96
Mean    0.89    —      0.93
Median  0.96   0.93     —
```

三對都 ≥ 0.8，所以是同一群。各自和另外兩個的平均：Mode 0.925、Mean 0.91、Median 0.945。Median 最高，留 Median。

**實際結果**

| 群 | 留下 | 移除 |
|---|---|---|
| {Mode, Mean, Median} | Median | Mode、Mean |
| {Width, Min} | （沒有） | Width、Min |
| Max、Variance、Nmax、Nzeros、Tendency（各自一群） | 全部 | — |

共移除 Width、Min、Mode、Mean，21 → 17。

**重要：這不是「沒有資訊損失」的精簡**

- 只移除 Width：完全沒有損失，因為 Width 可以由 Max 和 Min 精確算回來。
- 同時移除 Width 和 Min：**Min 回不來了**。只知道 Max 沒辦法算出 Min（在正常資料中 Min 與 Max 的相關只有 −0.25）。
- 為什麼會兩個都被移除：{Width, Min} 這一群只有兩個成員，兩者的「平均相似度」相同，程式的平手規則選了排在前面的 Width 當代表；接著第 4 步又把 Width 移除，於是這一群一個都沒留下。這是規則步驟順序造成的結果，不是因為 Min 真的多餘。
- 程式另外算了一個數字幫助判斷損失多少（`linear_r2_of_dropped_from_kept`）：用留下的 17 個特徵做線性預測，可以解釋 Min 變異的 80%、Mode 的 93%、Mean 的 94%。也就是 Min 大約有 20% 的變化是留下的特徵線性組合不出來的。

所以 PRUNED17 應該稱為「依正常資料的相關性所做的啟發式挑選」。真正只移除確定性冗餘的是「21 個減掉 Width」的 20 個特徵版本。H3 因此比較三組：21、20、17。

## 12. `group_probe(df, sets)`：只需要知道它是什麼

| | |
|---|---|
| 輸入 | 訓練集的表；一組「名稱 → 特徵清單」 |
| 計算 | 把訓練集依檢查切成 5 份，輪流留一份當考卷；用其餘 4 份中的**正常片段**估計「正常的中心與形狀」，對考卷上的片段算距離（Mahalanobis）當作異常分數；算 AUROC 與 PR-AUC；重複 5 次 |
| 輸出 | `group_probe.csv` |
| 為什麼需要 | 在把假說交給 P5 之前，先粗略看哪些方向值得測 |
| 對應 | **不進 §4**。只出現在 `p2_hypotheses.md` 當作內部參考 |

讀這個函式時要記住四件事：
1. 它完全在訓練集內進行，沒有碰驗證集或測試集。
2. 它用了訓練集的病理標籤來打分數，所以用它來挑假說，就是「看過訓練標籤」的決定，不能說成完全無監督的特徵選擇。
3. 它是一個簡單的線性探針，不是 P3 的 Autoencoder，數字不能當成模型成績。
4. 重複 5 次之間 PR-AUC 的標準差約 0.06。比這個小的差距（例如 17 個特徵的 0.665 對 21 個的 0.647）既不能說變好，也不能說一樣好。而且這個標準差只反映切法不同，不代表換一批孕婦會差多少。

---

## 數字對照表：報告裡的數字從哪裡來

| §4 裡的敘述 | 檔案 | 欄位 |
|---|---|---|
| ASTV median 40 vs 65 | `separability.csv` | `normal_median`、`path_median` |
| KS 0.71 [0.56, 0.84] | `separability.csv` | `KS`、`KS_ci_lo`、`KS_ci_hi` |
| DP 62.5% vs 4.7% | `event_presence.csv` | `share_with_event` |
| 拿掉兩次檢查後 0.41、0.37 | `separability.csv` | `KS_wo_top2` |
| 以片段重抽 0.51–0.68 | `separability.csv` | `KS_rowboot_lo`、`KS_rowboot_hi` |
| MSTV AUC 0.59／0.80 | `separability.csv` | `AUC_path_gt_normal`、`AUC_abs_dev` |
| 子群中位數（MSTV 0.3／2.7 等） | `pathologic_phenotypes.csv` | 各特徵欄 |
| Spearman 0.90–0.97、−0.89、0.82 | `redundancy_pairs.csv` | `spearman_train_all` |
| 21 → 17 的清單 | `histogram_pruning.json` | `kept`、`dropped` |
| 28 examinations、36 segments | `audit_train.json` | `n_exams_with_pathologic`、`pathologic_per_exam_top5` |
