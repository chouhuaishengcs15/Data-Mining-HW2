# P2 Prior Work — 同一資料集（UCI CTG）的公開研究與實作

- 整理：P2（Claude Code Fable），檢索日期 2026-11-10
- 用途：提供 §2 Related Work 與 §4 假說的外部證據。**外部結論一律標為 External Evidence，不當成本組結果。**

## 0. 閱讀狀態（誠實揭露）

檢索工具回傳的是頁面摘要而非逐字原文，下表「已讀」代表讀過內文摘要；**引用到報告前，請對照原文再確認一次逐字引述與數字。**

| 來源 | 狀態 |
|---|---|
| UCI 資料頁 | 已讀 |
| Huang et al. 2021（PMC8686372） | PMC 頁面被 reCAPTCHA 擋下；改由 Europe PMC 全文 XML 讀取（約 96%） |
| Kaya et al. 2025（ASEJ 103698） | **未讀全文**（ScienceDirect 403）；只有 Crossref 書目與 OpenAlex 摘要 |
| evercloud GitHub repo | 已讀 README 與主程式；重要度圖檔未讀 |
| Kaggle：karnikakapoor、akshat0007 | **無法開啟**（只回傳 JS 外殼），找不到鏡像；本文不對這兩份 Notebook 做任何陳述 |
| Akmal et al. 2023（PMC10252854） | 已讀（前半） |
| Mandala 2023（arXiv 2310.00505） | 已讀 PDF |
| Dhaliwal et al. 2025（UH capstone poster） | 已讀文字，圖無法讀 |
| Costa et al. 2021（Frontiers, SisPorto 4.1） | 已讀（STV／LTV 定義） |
| ODDS Cardio 頁面 | **未開啟**（SSL 失敗）；資訊來自搜尋摘要＋ADBench README（已讀） |
| FIGO 2015 指引、SisPorto 2.0 原始論文 | **未開啟**；只有搜尋摘要 |

## 1. 比較表

| Source | Dataset Version | Task / Label | Split | Feature Selection | Feature Engineering | Evidence（原文所述） | URL |
|---|---|---|---|---|---|---|---|
| **A. UCI 官方頁**（Campos & Bernardes, 2000；DOI 10.24432/C51S4N；CC BY 4.0） | 2126 筆、21 特徵、無缺值 | CLASS（型態 1–10）與 NSP（三類）；三位產科專家共識標註 | 無 | 無 | SisPorto 自動計算 | 變數定義；單位欄空白，單位只出現在描述文字（bpm、per second、percentage） | https://archive.ics.uci.edu/dataset/193/cardiotocography |
| **B. Huang et al. 2021**, BMC Med Inform Decis Mak 21:355 | UCI，2126 筆，21 特徵 | 三分類 NSP（1655／295／176） | **無切分**：全資料的統計分析 | 視覺化、Spearman、關聯規則（ARA）、結構方程模型（SEM） | 離散化區間；五個潛在類別 ACat（AC、Nmax、FM）、BCat（LB、Mean、Median、Mode）、VCat（Variance、MSTV、ALTV、ASTV、Width）、DCat（DP、DL、Nzeros）、UCat（UC） | 與病理相關：MSTV、MLTV、ASTV；與正常相關：AC、ALTV、UC。共線群：{LB, Mode, Median, Mean}、{Width, Variance, Nmax, MSTV, Max, DL}。病理規則：Mode∈[60,99)、Variance>165、DS>0.00094/s、DP>0.0028/s。SEM：ACat −0.665 | https://pmc.ncbi.nlm.nih.gov/articles/PMC8686372/ |
| **C. evercloud/machine-and-deep-learning-project** | UCI ID 193，21 特徵（已排除專家型態欄） | 三分類 NSP | 70/20/10 隨機分層（`train_test_split`, seed 42），**未依 exam 分組** | PCA（95% 變異，train fit）；RF importance；SHAP TreeExplainer | 無手工衍生特徵 | LR／SVM／RF／DNN 準確率 0.87–0.92。重要度排名只在圖檔，未能讀取 | https://github.com/evercloud/machine-and-deep-learning-project |
| **D. Akmal et al. 2023**, Diagnostics 13(11):1931 | UCI，2126 筆 | 三分類 NSP 與十分類 CLASS | 75:25 ＋ 5-fold 分層 CV | Autoencoder 產生 NEF1–14，再用 RFE（RF）留 13 個 | Autoencoder 潛在特徵（無法對回原始特徵） | 準確率 96.62%（13 特徵）對 93.61%（21 特徵 RF） | https://pmc.ncbi.nlm.nih.gov/articles/PMC10252854/ |
| **E. Dhaliwal et al. 2025**（Univ. of Houston capstone poster, HEART-fC） | UCI CTG | 三分類 NSP | 海報文字未說明 | SHAP、LIME | 無 | 前十名提到 AC、ASTV、MSTV、MLTV、DP、Width、Max、Nmax、Nzeros；**AC、MSTV、Nzeros 在 Normal–Pathologic 之間的 SHAP 差異不顯著** | https://www.bioe.uh.edu/sites/bme/files/files/capstone2025_group08.pdf |
| **F. ODDS／ADBench `cardio`** | 由 UCI CTG 衍生：1831 筆、21 特徵、176 個離群（9.61%） | Normal＝inlier、Pathologic＝outlier、Suspect 丟棄 | 基準自訂（非 exam 分組） | 無 | 無 | 與本組任務定義相同，是唯一設定吻合的外部基準 | https://github.com/Minqi824/ADBench ；https://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/ |

### 參考價值較低、列出以免誤用

| Source | 問題 | URL |
|---|---|---|
| Kaya, Kaya & Atar 2025, Ain Shams Eng J 16(11):103698 | 摘要顯示是把一維 CTG 訊號轉成 2D 影像、用 AlexNet 取深度特徵再做最佳化選特徵；**選的不是這 21 個 SisPorto 特徵**，無法提供我們的特徵排名。全文未讀，資料集細節未確認 | https://doi.org/10.1016/j.asej.2025.103698 |
| Mandala 2023, arXiv:2310.00505 | Kaggle 版資料、LightGBM＋SMOTE（順序未說明）、準確率 98.31%；有「Feature Selection」小節但沒有方法與排名，且宣稱有「maternal blood pressure」特徵（資料集沒有）。可信度低 | https://arxiv.org/abs/2310.00505 |

## 2. 重點問題的回答

**其他研究認為哪些特徵重要？**
可讀到內文的來源中，被點名的有 ASTV、MSTV、MLTV、ALTV、AC、DP（B、E），以及 Mean／Mode／Median／Variance／DS 的極端區間（B 的病理規則）。**沒有任何一個我們實際讀到的來源提供 21 個原始特徵的數值化 SHAP／tree importance 排名。**

**用什麼技術選特徵？** Spearman／關聯規則／SEM（B）、PCA＋RF importance＋SHAP（C）、Autoencoder＋RFE（D）、SHAP＋LIME（E）。沒有看到 Mutual Information 或 KS。

**有無新建衍生特徵？** 只有潛在變數（B）、PCA 成分（C）、Autoencoder 特徵（D）、CNN 深度特徵。**沒有找到手工的比例、加總或交互作用特徵**（例如 Deceleration Sum）。所以本組 H2 的衍生特徵沒有外部先例可引用，只能靠自己的資料驗證。

**類別不平衡怎麼處理？** D 用 SMOTE：樣本數變成 4965 ＝ 3×1655，推論是在切分前對全資料過採樣（論文未明講順序，這是我們的推論）。C 沒有處理。

**哪些做法適合本組的 Normal vs. Pathologic 設定？**
- 適合：F 的任務定義（直接對應）；C 的「scaler／PCA 只在 train fit」；B 的共線群（提供冗餘假說的外部佐證）。
- 不適合：SMOTE（one-class 模型不看異常，也不該合成異常）；Autoencoder／PCA 潛在特徵用於 §4（失去可解釋性）；任何三分類準確率。

**Data Leakage／無法公平比較**
1. **全部來源都沒有依 exam（FileName）分組。** 一次檢查平均有 6 個片段，隨機切分會讓同一次檢查的片段同時出現在訓練與測試。
2. D（可能還有 Mandala）在切分前做 SMOTE。
3. B 是全資料的 in-sample 統計，沒有 held-out 評估，且包含 Suspect。
4. 全部是三分類、指標多為 accuracy；與本組的二元、exam 分組、PR-AUC 無法直接比較。**外部的 9x% 準確率不能拿來當本組的對照數字。**

## 3. STV／LTV 的定義（External Evidence）

- Costa et al. 2021（Front Pediatr 9:662101，SisPorto 4.1，已讀）：abnormal STV 為「相鄰 FHR 訊號差距小於 1 bpm 的百分比」；abnormal LTV 為「1 分鐘視窗內最大與最小值差距低於 5 bpm 的百分比」。https://www.frontiersin.org/articles/10.3389/fped.2021.662101/full
- SisPorto 2.0 多中心驗證論文（僅搜尋摘要）有相同定義。https://core.ac.uk/download/143398503.pdf
- 原始論文 Ayres-de-Campos et al. 2000（J Matern Fetal Med 9(5):311–318）未能開啟。
- 注意：上述定義來自較新版本的 SisPorto 文獻；UCI 資料頁沒有寫門檻。報告中建議寫「as defined in the SisPorto literature」，不要寫成 UCI 的定義。

## 4. 臨床指引背景（僅搜尋摘要，未讀全文）

FIGO 2015（Ayres-de-Campos, Spong & Chandraharan, Int J Gynecol Obstet 131:13–24, DOI 10.1016/j.ijgo.2015.06.020）：Normal 為基準率 110–160 bpm、變異度 5–25 bpm、無反覆減速；Pathological 包含基準率 < 100 bpm、變異度降低、反覆晚期或延長減速。Olofsson et al. 2018（已讀，https://pmc.ncbi.nlm.nih.gov/articles/PMC5887886/）指出 FIGO 2015 判定正常並不需要加速。
限制：UCI 標籤（2000 年）早於此指引，且指引針對產程中。只能當背景，不能當作本資料的標註規則。

## 5. Feature Candidate Table（外部證據 × 本組 train 實測）

本組數字來自 `EDA/outputs/separability.csv`（只用 train）。KS 後方括號為 exam-level bootstrap 95% 區間。

| Feature | 外部來源怎麼說 | 本組 train 實測 | 是否一致 | 結論 |
|---|---|---|---|---|
| ASTV | B：與病理正相關；E：SHAP 前十 | KS 0.71 [0.56, 0.84]，中位數 40→65 | 一致 | 首要候選（H1） |
| MSTV | B：與病理相關（0.5）；E：N–P 的 SHAP 差異不顯著 | 單向 AUC 0.59，但 \|z\| AUC 0.80；Pathologic 呈雙峰（0.3 與 2.7） | **表面矛盾、實際可解釋**：B 與 E 的落差正好符合「非單調」 | 保留；提醒單向指標會誤判 |
| ALTV | B：與正常負相關 | KS 0.22 [0.14, 0.50]；只有平坦型很高（中位數 69） | 部分一致（外部含 Suspect，可能使 ALTV 顯得更重要） | 保留在 H1 群組內，不單獨主張 |
| MLTV | B、E：重要 | KS 0.54 [0.44, 0.68]；Pathologic 51% 為 0 | 一致 | H1 |
| AC | B：與正常關聯最強；E：N–P 不顯著 | KS 0.50 [0.34, 0.67]；有加速的比例 67% vs 19% | 大致一致 | H2 群組 |
| DP | B：DP>0.0028/s 為病理規則；E：前十 | KS 0.60 [0.29, 0.78]；出現率 4.7% vs 62.5%；但拿掉兩個最大 exam 後 KS 降到 0.41 | 一致，但**高度依賴少數 exam** | H2 群組；寫入限制 |
| DS | B：DS>0.00094/s 為病理規則 | train 只有 7 筆非 0（Pathologic 6、Normal 1） | 方向一致、涵蓋率極低 | 不單獨主張 |
| DL | 無來源點名 | KS 0.25；拿掉兩個最大 exam 後 0.08 | — | 弱 |
| Mean／Median／Mode | B：Mode∈[60,99) 為病理規則；與 LB 共線 | KS 0.58／0.54／0.48；彼此 ρ 0.90–0.97 | 一致 | 冗餘群（H3），留一個即可 |
| Variance | B：Variance>165 為病理規則 | 雙向（平坦型 0、減速型 57），\|z\| AUC 0.79 | 一致（B 只描述了高的一端） | 保留 |
| Width／Min／Max／Nmax | B：共線群；E：前十 | Width＝Max−Min 完全相等；Width–Min ρ=−0.89 | 一致 | H3：移除 Width |
| LB | B：BCat | KS 0.22，\|z\| AUC 0.37（比隨機還差） | 不支持單獨使用 | 低優先 |
| FM、UC、Nzeros、Tendency | 少或無 | KS 0.05–0.32 | — | 低優先 |
