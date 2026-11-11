# P2 Dataset Brief — UCI Cardiotocography（CTG）

- 撰寫：P2（Claude Code Fable 協作），2026-11-10
- 依據：P1 commit `c2b70a4`（`data/README.md`、`data/load_data.py`、`data/make_split.py`、`data/raw/CTG.xls`、`data/splits/*.csv`）
- 數字來源：`EDA/outputs/audit_train.json`（由 `python EDA/p2_eda.py` 產生）以及本文 §3 的稽核指令

## 先讀這一段（給第一次讀的人）

1. 一筆資料是一段 CTG 監測片段，有 21 個由胎心率與宮縮訊號算出來的數字。
2. 任務是二元的：Normal = 0、Pathologic = 1。Suspect 不在主要任務裡。
3. 訓練集有 972 筆 Normal、104 筆 Pathologic，來自 175 次檢查。病理約佔 9.7%。
4. 同一次檢查會切成好幾段，彼此很像，所以 P1 以「檢查」為單位切分資料。
5. P2 做 EDA 時看完整的訓練集（含病理標籤）；P3 訓練 one-class 模型時只用 972 筆 Normal。
6. P2 完全不看驗證集與測試集的特徵。
7. 想先理解 ASTV、DP、MSTV 為什麼值得看，請先讀 `p2_eli5_learning_path_zh.md`，再回來讀本文。
8. 本文 §2 的 21 個特徵字典是查表用的，不需要一次讀完。
9. 本文 §3 是資料稽核結果，重點是「P1 的切分沒有洩漏」與「病理集中在少數檢查」。
10. 所有臨床直覺都只是幫助理解的背景，不是本資料的標註規則，也不是我們的結論。

本文把三種資訊分開標示：
**【官方】** UCI 頁面或 `CTG.xls` 的 `Description` 工作表；**【P1】** P1 自訂的任務設定；**【動機】** 本組提出的研究動機（不是官方說法）。

---

## 1. Task Definition

### 1.1 CTG 是什麼？

Cardiotocography（胎心宮縮監測）是在孕婦腹部綁兩個感測器，同時連續記錄兩條曲線：

| 訊號 | 白話 | 臨床上看什麼 |
|---|---|---|
| Fetal Heart Rate（FHR，胎心率） | 胎兒每分鐘心跳幾下（bpm） | 基準心率、心跳的「起伏程度」（variability）、短暫加快（acceleration）、短暫變慢（deceleration） |
| Uterine Contraction（UC，子宮收縮） | 子宮何時收縮 | 收縮頻率，以及胎心率在收縮前後的反應 |

直覺上，健康胎兒的心跳會不停小幅波動，偶爾加快；如果心跳變得很平、沒有加速，或出現又深又久的減速，可能代表胎兒缺氧，需要醫護人員進一步評估。

### 1.2 為什麼要偵測 Pathologic 片段？【動機】

CTG 判讀靠人眼，判讀者之間不一致是已知問題（這也是 SisPorto 這類自動分析系統出現的原因）。病理片段很少（本資料約 9.6%），適合定義成異常偵測：模型只學「正常長什麼樣子」，把偏離正常的片段排到前面，讓醫護人員優先複查。

**邊界：** 模型輸出是「這段訊號的特徵組合和正常片段差多遠」的分數，只能當作排序／提醒，不是診斷，也不能據此決定處置。標籤本身是 2000 年前後三位專家的共識判讀，不是新生兒結局（例如臍帶血 pH）。

### 1.3 一筆資料代表什麼？

一筆 ＝ **一段 CTG 監測片段（segment）**，不是一個人，也不是一次完整檢查。

- 【官方】2126 筆 FHR 訊號的量測與分類；特徵由 SisPorto 自動計算；三位產科專家給共識標籤。
- 【P1，已核對】`Raw Data` 工作表的 `FileName` 是檢查（exam）代號，`b`、`e` 是片段起訖。2126 個片段來自 352 次檢查；片段長度 117–3599 秒，中位數 799 秒。
- 資料沒有孕婦 ID，所以**無法確認不同 exam 是否屬於同一位孕婦**。Exam 分組能擋掉同一次檢查的洩漏，擋不掉同一人多次檢查的相依（寫進限制）。

### 1.4 Normal／Suspect／Pathologic

【官方】`NSP`：Normal=1、Suspect=2、Pathologic=3，是專家對「胎兒狀態」的共識分類。官方沒有寫出判讀規則。
一般臨床指引（例如 FIGO 2015，僅供理解，**不是本資料的標註規則**，且該指引針對產程中，本資料據 Huang et al. 2021 為產前紀錄）大致是：基準率、變異度、減速都正常為 Normal；有明確病理特徵（變異度持續過低、反覆晚期或延長減速等）為 Pathological；介於兩者之間為 Suspicious。

### 1.5 P1 為什麼排除 Suspect？【P1】

P1 的設定：Normal→0、Pathologic→1、Suspect→-1 另存 `suspect.csv`，不訓練也不計入主要指標。

| 好處 | 限制 |
|---|---|
| 異常定義乾淨，Ground Truth 爭議較小 | 真實部署一定會遇到 Suspect，主指標沒有衡量這部分 |
| 與 ODDS／ADBench 的 `cardio` 基準（1831 筆、176 個離群）設定相同，有外部對照 | 去掉最難的邊界樣本，指標會比實際使用情境樂觀 |
| 異常率 9.61%，符合「罕見事件」 | 290 多筆資料沒有用到；外部研究多為三分類，數字不能直接比 |

P1 已建議 P5 用 Suspect 做錯誤分析（看 Suspect 的分數落在哪裡），這是合理的補救。

### 1.6 資料使用方式（重要）

| 用途 | 資料 | 可否看異常標籤 |
|---|---|---|
| P2 描述性 EDA | 完整 `train.csv`（972 Normal＋104 Pathologic） | 可以，用來描述分布差異 |
| One-class 模型 fitting（P3） | `load_split("train", use_normal_only=True)` 的 972 筆 Normal | 不可以 |
| 類似模型的前處理（標準化、共變異數、冗餘剪枝） | 只用 train 的 Normal | 不可以 |

**誠實聲明：** 若最後依 P2 的分布比較結果（看過訓練異常標籤）挑選特徵，整條流程就不是「完全無監督」，應寫成「模型參數只用正常資料訓練；特徵集合的選擇參考了訓練集的異常標籤」。H3 的 histogram 挑選規則只用正常資料，不受此限；H1 則是看過訓練標籤後提出的。

---

## 2. Feature Dictionary（21 個特徵）

欄位說明：
- **Unit**：只寫官方有明講的單位；官方沒寫的標「待核對」。
- **Potential Leakage**：是否可能偷看到標籤。21 個特徵都是 SisPorto 從訊號自動算的，沒有直接洩漏；真正會洩漏的欄位（`A`–`E`、`AD`、`DE`、`LD`、`FS`、`SUSP`、`CLASS`、`NSP`）P1 已排除。
- **Source**：U＝UCI 變數表；X＝`CTG.xls` Description 工作表；S＝SisPorto 文獻定義（Costa et al. 2021，見 `p2_prior_work.md`）。
- 「大／小代表什麼」的臨床直覺是一般知識，僅供理解；「本組 train」欄才是實測結果（Normal 中位數 → Pathologic 中位數）。

| Feature | Meaning | Unit | Data Type（train 實測） | Clinical Intuition（大／小） | 能否單獨反映異常／需搭配 | 本組 train：N → P 中位數 | Potential Leakage | Source |
|---|---|---|---|---|---|---|---|---|
| LB | FHR 基準值（SisPorto 計算） | bpm（U） | 整數，46 個相異值 | 過高（心搏過速）或過低（心搏過緩）都可能異常，是雙向的 | 單獨幾乎無用（KS 0.22，\|z\| AUC 0.37）；需搭配變異度與減速 | 132 → 133 | 無 | U, X |
| AC | 加速次數／秒 | 次／秒（U） | 事件率，37.5% 為 0 | 有加速通常是好現象；沒有加速要搭配變異度判斷 | 中等（KS 0.50）；「沒有加速」在 Normal 也常見（33%） | 0.0025 → 0 | 無 | U, X |
| FM | 胎動次數／秒 | 次／秒（U） | 事件率，64% 為 0，極右偏（最大 0.48） | 胎動多寡受睡眠週期影響 | 弱（KS 0.32） | 0 → 0.001 | 無 | U, X |
| UC | 宮縮次數／秒 | 次／秒（U） | 事件率，10% 為 0 | 本身不代表胎兒狀態，是解讀減速的背景 | 弱（KS 0.25） | 0.005 → 0.004 | 無 | U, X |
| DL | 輕度減速次數／秒 | 次／秒（U） | 事件率，53% 為 0 | 輕度減速在正常片段也常見 | 弱（KS 0.25）；Normal 有 45% 出現過 | 0 → 0.002 | 無 | U, X |
| DS | 重度減速次數／秒 | 次／秒（U） | **近乎常數**：99.3% 為 0（train 只有 7 筆非 0） | 出現即值得注意 | 幾乎沒有涵蓋率（Pathologic 只有 5.8% 非 0） | 0 → 0 | 無；但標準化後會變成極端值（見 §3.4） | U, X |
| DP | 延長減速次數／秒 | 次／秒（U） | 事件率，89.7% 為 0 | 延長減速是指引中的病理特徵之一 | **強但只涵蓋一部分**：Pathologic 62.5% 有、Normal 4.7% 有 | 0 → 0.0017 | 無 | U, X |
| ASTV | 短期變異度異常的時間百分比 | %（U） | 整數 12–84 | 大＝大部分時間相鄰心跳幾乎不變（SisPorto：相鄰 FHR 差 < 1 bpm）＝變異度差 | **單一特徵中最強**（KS 0.71）；Normal 仍有 16% ≥ 60 | 40 → 65 | 無 | U, X, S |
| MSTV | 短期變異度平均值 | **待核對**（U 未寫；P1 README 寫 bpm，SisPorto 文獻以 bpm 計，但官方資料頁沒有明講） | 一位小數 0.2–7.0 | 小＝心跳很平；大＝起伏大（減速時也會變大） | **雙峰**：Pathologic 有一群極低（0.3）、一群偏高（2.7）；單向指標會低估它（AUC 0.59，\|z\| AUC 0.80） | 1.2 → 2.1 | 無 | U, X |
| ALTV | 長期變異度異常的時間百分比 | %（U） | 整數 0–91，64.5% 為 0 | 大＝長時間心率曲線很平（SisPorto：60 秒窗內最大減最小 < 5 bpm） | 只在 DP-absent 的 Pathologic 子群很高（中位數 69）；整體 KS 只有 0.22 | 0 → 0 | 無 | U, X, S |
| MLTV | 長期變異度平均值 | **待核對**（同 MSTV） | 一位小數 0–50.7，8.6% 為 0 | 小＝長期起伏小 | 中強（KS 0.54）；Pathologic 51% 為 0，Normal 只有 4% | 8.1 → 0 | 無 | U, X |
| Width | FHR 直方圖寬度 | 待核對（U 未寫；數值與 bpm 一致） | 整數 | 大＝心率範圍寬 | **與 Max−Min 完全相等**（train 殘差 0），零新資訊 | 68 → 101.5 | 無 | U, X |
| Min | 直方圖最小值 | 待核對（同上） | 整數 | 小＝曾出現很低的心率（深減速） | 中（KS 0.40） | 94 → 63 | 無 | U, X |
| Max | 直方圖最大值 | 待核對 | 整數 | 大＝曾出現很高的心率 | 弱（KS 0.14） | 162 → 162 | 無 | U, X |
| Nmax | 直方圖峰值數 | 個（U 寫 # of peaks） | 整數 0–16 | 多峰＝心率分布分散 | 弱（KS 0.16） | 3 → 5 | 無 | U, X |
| Nzeros | 直方圖零值數 | 個 | 整數 0–10，74% 為 0 | 直方圖中間有空 bin | 無（KS 0.05） | 0 → 0 | 無 | U, X |
| Mode | 直方圖眾數 | 待核對 | 整數 | 與基準率接近 | 與 Mean／Median 高度重複（ρ ≥ 0.90） | 137.5 → 117 | 無 | U, X |
| Mean | 直方圖平均值 | 待核對 | 整數 | 低＝整段心率被減速拉低 | 中強（KS 0.58），但主要來自 DP-present 的 Pathologic 子群 | 134 → 103.5 | 無 | U, X |
| Median | 直方圖中位數 | 待核對 | 整數 | 同 Mean | 同上（KS 0.54），與 Mode ρ=0.97 | 138 → 112 | 無 | U, X |
| Variance | 直方圖變異數 | **待核對**（官方未寫；不可假設為 bpm²） | 整數 0–195，右偏 | 大＝心率分散（減速多）；0＝幾乎一條線 | **雙向**：DP-absent 子群為 0、DP-present 子群為 57；\|z\| AUC 0.79 | 8 → 41.5 | 無 | U, X |
| Tendency | 直方圖偏態：−1 左偏、0 對稱、1 右偏（X） | 無單位（類別） | 三值有序 | 左偏＝低心率尾巴長 | 弱（KS 0.26） | 0 → 0 | 無 | X（U 頁未列編碼） |

備註：
- `Min`／`Max` 在 Description 工作表寫的是 “low freq. / high freq. of the histogram”，P1 與 UCI 變數表解讀為直方圖的最小／最大值。
- `LBE`（專家判讀的基準值）與 `LB` 在 2126 筆完全相同，P1 已移除；它是人工判讀欄位，移除是對的。

---

## 3. Dataset Audit（實測）

### 3.1 樣本數與標籤（與 P1 README 一致）

| Split | Normal | Pathologic | 異常率 | 含 N/P 片段的 exam 數 | 有 Pathologic 的 exam 數 |
|---|---|---|---|---|---|
| train | 972 | 104 | 9.67% | 175 | 28 |
| val | 342 | 35 | 9.28% | 57 | 9 |
| test | 341 | 37 | 9.79% | 75 | 13 |

> val／test 只讀了 `exam`、`label` 與重複列數來做切分完整性檢查，沒有看任何特徵分布，也沒有用於任何特徵決策。

### 3.2 檢查結果

| 項目 | 結果 |
|---|---|
| 與原始檔一致性 | train 的 21 個特徵、label、exam 逐列與 `CTG.xls` 的 `Raw Data` 重算結果相同 |
| Missing value | 0 |
| Constant feature | 無；**DS 近乎常數**（99.3% 為 0） |
| 重複列（train，21 特徵完全相同） | 1 組，同一 exam、同標籤 |
| 跨 split 重複列 | 0；沒有標籤衝突的重複列 |
| Exam 重疊 | train∩val＝0、train∩test＝0、val∩test＝0 |
| Target leakage 欄位 | `A`–`E`、`AD`、`DE`、`LD`、`FS`、`SUSP`、`CLASS`、`NSP` 都不在 CSV；`LBE`、`DR`（全 0）、`b`、`e` 也已移除 |
| 特徵轉換 | 只有一項：AC、FM、UC、DL、DS、DP ＝ 原始次數 ÷（`e − b` 秒） |
| 正規化 | CSV 是原始尺度；`fit_preprocess` 是 z-score，由呼叫者傳入的資料 fit（P3 傳 Normal-only train 即可） |
| Width 與 Max−Min | **完全相等**（train 1076 筆與原始 2126 筆皆是，最大殘差 0） |

### 3.3 與 P1 README 的差異（需回報，不影響切分）

1. **Exam 數的口徑**：README 寫 train 有 204 次檢查，這是包含 Suspect 的數字；`train.csv`（只含 Normal／Pathologic）實際是 175 次。報告若寫「train 有 204 exams」會和主任務資料對不上，建議 P1／I1 註明口徑。
2. **MSTV／MLTV 單位**：README 寫 bpm，UCI 變數表沒有給單位。報告引用前請核對原始文獻，或寫「unit not stated」。
3. **`xlrd` 相依**：`make_split.py` 需要 `xlrd`，repo 沒有 requirements 檔（README 有口頭提到）。
4. 工作目錄顯示 15 個檔案被修改，實際只是 Windows 換行（CRLF）差異，內容相同；P2 沒有提交這些檔案。

### 3.4 對下游有影響的觀察

- **Pathologic 高度集中在少數 exam**：104 筆來自 28 個 exam，其中兩個 exam（`S8001034.dsp` 20 筆、`S8001038.dsp` 16 筆）就佔 34.6%。所有統計推論都要以 exam 為單位（本組用 exam-level bootstrap）。
- **DS 標準化風險**：Normal-train 的 DS 只有 1 筆非 0，標準差極小；P1 的 z-score 會把任何一次重度減速放大成數十個標準差的極端值。這對距離型／重建型模型可能是好事也可能造成單一特徵主導，請 P3 留意。
- **共變異數矩陣不滿秩**：因為 Width＝Max−Min，21 個特徵的秩只有 20；Mahalanobis、PCA whitening、高斯模型需要先拿掉 Width 或用 shrinkage。
