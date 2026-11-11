# EDA — P2 資料探索與特徵設計

負責角色：P2（報告第 4 節 Exploratory Feature Analysis）

本資料夾只用訓練集（972 筆 Normal＋104 筆 Pathologic，來自 175 次檢查）描述正常與病理 CTG 片段的分布差異，找出有辨識潛力的變異度特徵，並用相關性分析找出重複資訊，形成後續 Autoencoder 特徵消融的假說。這裡沒有訓練任何偵測模型，也沒有讀取驗證集或測試集。

> **目前狀態**：H1、H3 都還是假說，**Pending P5 Verification**。消融結果出來之前，請不要在報告或簡報中把它們寫成結論。

## 1. P2 任務對照

| 任務要求 | 狀態 | 在哪裡 |
|---|---|---|
| 至少一張正常與異常分布對照圖 | 完成 | `figures/fig1_variability_contrast`（ASTV、MSTV、MLTV）、`figures/fig2_events_contrast`（AC、DL、DP 等事件率） |
| 量化特徵可分性 | 完成 | `outputs/separability.csv`：KS、kNN Mutual Information、Wasserstein、AUC，附 exam-level bootstrap 95% CI；排名圖 `figures/fig3_separability_ranking` |
| 檢查候選特徵是否重複 | 完成 | `outputs/spearman_train_normal.csv`、`redundancy_pairs.csv`、`vif_normal_train.csv`、`histogram_pruning.json`；熱圖 `figures/fig4_spearman_heatmap` |
| 提出特徵假說供 §7–§8 驗證 | 假說已提出，**驗證待 P5** | `docs/p2_hypotheses.md`、`outputs/feature_sets.json` |
| 撰寫第 4 節內文 | 完成（英文） | `../report/p2_section4.tex`、`../report/p2_refs.bib` |
| 簡報用圖與說明 | 完成（兩頁建議） | `../slides/p2_slides_notes.md` |

主要交付是 `P2_Section4_Deliverable.ipynb`：一份 Notebook 依序涵蓋以上各項，圖表都已內嵌，第 7.1 節會自動核對 LaTeX 內文的數字是否等於重新計算的結果。

## 2. 主要發現（全部是訓練集的觀察）

| 觀察 | 數字 | 對後續設計的意義 |
|---|---|---|
| ASTV 是最能區分、也最穩定的單一特徵 | 中位數 40 對 65；KS 0.71（95% CI 0.56–0.83） | 變異度特徵群是 H1 的對象 |
| DP（延長減速）在病理片段常見 | 出現率 4.7% 對 62.5%；KS 0.60（CI 0.29–0.78），拿掉兩個最大的病理檢查後降到 0.41 | 訊號強但集中在少數檢查，不確定性大 |
| MSTV 往兩個方向偏離正常 | 單方向 AUC 0.59，\|z\| 的 AUC 0.80 | 異常分數必須衡量「偏離程度」，不能只看單一方向 |
| `Width = Max − Min` 完全成立 | 21 個特徵的共變異數秩為 20 | 移除 Width 沒有資訊損失 |
| Mode、Mean、Median 高度相關 | Spearman ρ 0.90–0.97 | H3 的精簡對象；移除 Mode、Mean 不是無損 |
| 104 筆病理片段只來自 28 次檢查 | 其中兩次檢查佔 36 筆 | 所有信賴區間都以整次檢查為單位重抽 |

## 3. 假說與特徵集合

| 假說 | 比較 | 預期 |
|---|---|---|
| **H1**　變異度特徵提供其他特徵補不上的偵測資訊 | `ALL21` 對 `NO_VARIABILITY17` | 移除後 PR-AUC 下降 |
| **H3**　依冗餘精簡後表現與完整集合相近 | `ALL21` 對 `DROP_WIDTH20` 對 `PRUNED18` | 差距在 seed 變動範圍內 |

| 名稱 | 相對於 `ALL21` 移除的欄位 | 備註 |
|---|---|---|
| `ALL21` | — | 即 `load_data.FEATURES` |
| `NO_VARIABILITY17` | ASTV、MSTV、ALTV、MLTV | H1 |
| `DROP_WIDTH20` | Width | 確定性冗餘，無資訊損失 |
| `PRUNED18` | Width、Mode、Mean | H3 主要集合；規則只用正常訓練資料，是啟發式挑選 |
| `PRUNED17`（選做） | Width、Min、Mode、Mean | 僅作對照；不可宣稱無資訊損失 |

機器可讀版本在 `outputs/feature_sets.json`（鍵名 `PRUNED17_optional`）。**這些定義是 P2 的建議，尚未與 P3、P5 確認**；有不同意見請直接提出。

## 4. 給 P3：如何沿用

P2 沒有改動 `data/` 或 `model/` 的任何檔案。`model/tier2_autoencoder.py` 目前固定使用 21 個欄位，以下是加入特徵集合選項的一種做法，是否採用由 P3 決定。

```python
import json
from load_data import FEATURES, fit_preprocess, load_split

SETS = json.load(open(ROOT / "EDA" / "outputs" / "feature_sets.json"))
cols = [FEATURES.index(f) for f in SETS["PRUNED18"]]   # 由 --feature-set 參數決定

prep = fit_preprocess(X_tr_raw)      # P1 的 z-score，逐欄計算
X_tr = prep(X_tr_raw)[:, cols]       # 標準化之後再選欄位
```

- `fit_preprocess` 內部固定取 `X[FEATURES]`，所以要在轉換**之後**用欄位位置選取。z-score 是逐欄計算，這樣做和各集合各自重新 fit 的結果相同。
- 逐特徵誤差檔的欄名（目前是 `columns=FEATURES`）要改成所選集合的欄位。
- 輸出資料夾名稱請帶上特徵集合名稱，避免覆蓋 `ALL21` 的既有結果。

建模時值得留意的觀察：

1. Width 造成共變異數矩陣不滿秩。Mahalanobis 分數請使用 shrinkage 或先移除 Width。
2. DS 在正常訓練資料只有 1 筆非 0，標準差極小；驗證集只要出現一次重度減速，標準化後就是數十個標準差，可能讓單一特徵主導分數。可以考慮對 z-score 截斷。
3. 事件率特徵大量為 0（DP 90%、FM 64%、DL 53%、AC 38%），標準化後仍然高度偏態。
4. 用逐特徵重建誤差解釋示警時，請稱為 reconstruction-error diagnostic；它不是 Mahalanobis 分數的精確歸因。

## 5. 給 P5：如何沿用

完整規格在 `docs/p2_hypotheses.md` 的「P5 Handoff Table」。

| ID | 特徵集合 | 對應假說 | 優先順序 | 何種結果算不支持假說 |
|---|---|---|---|---|
| A0 | `ALL21` | baseline | 必做 | — |
| A1 | `NO_VARIABILITY17` | H1 | 必做 | 下降幅度小於 seed SD |
| A5 | `DROP_WIDTH20` | H3 | 建議 | 明顯下降 |
| A6 | `PRUNED18` | H3 | 建議 | 下降超過 seed SD |
| A6b | `PRUNED17` | H3 對照 | 選做 | — |

實驗控制：

- 每一列與 A0 只差特徵集合。P1 的切分、P3 的模型與超參數、seeds（P3 預設 0–4）都固定。
- 只用**驗證集**比較：PR-AUC 為主、AUROC 為輔；F1 要用同一種門檻規則。
- 測試集只對事先凍結的設定評估一次，不可用來挑特徵集合。
- 驗證集的病理片段只來自 9 次檢查。小於 seed SD 的差距請寫成「未觀察到差異」，不要寫成「證明等效」。
- 課程要求至少 2 個 controlled ablations，其中至少 1 個是異常偵測專屬。以上全是特徵消融，**不能取代**那一個；可搭配 P3 既有的 `--score mse|cosine|mahalanobis` 比較，並分成兩張表。
- 建議的附加分析：把驗證集的病理片段依 `DP > 0` 分兩群，分別報告召回率與筆數。若 A1 的損失集中在 DP = 0 的那一群，就直接對應 H1。
- `outputs/group_probe.csv` 是 P2 只用訓練集做的內部探針，使用了訓練集的異常標籤，**請不要把它的數字放進 §7–§8**。

**結果出來後請回報 P2**：各集合的驗證集 PR-AUC、AUROC（平均 ± SD）與 seeds。P2 會據此更新 Notebook 第 6 節與 `report/p2_section4.tex` 最後一段，把 Pending 改成實際結果。

## 6. 資料夾內容

| 路徑 | 內容 |
|---|---|
| `P2_Section4_Deliverable.ipynb` | 主要交付 Notebook（中文說明、英文 §4、一致性檢查） |
| `P2_Feature_Analysis.ipynb` | 較早的完整探索 Notebook，保留備查 |
| `p2_eda.py` | 產生所有統計表與圖的程式 |
| `tools/build_deliverable_notebook.py` | 由 `p2_eda.py` 與 LaTeX 內文組出交付 Notebook |
| `figures/` | 6 張圖，各有 PNG 與 PDF；`fig1` 用於 §4，`fig5`、`figS1` 為補充 |
| `outputs/` | 統計表與特徵集合定義 |
| `docs/p2_handoff.md` | 給 P3、P5、I1 的完整交接說明 |
| `docs/p2_hypotheses.md` | 假說的證據、限制與 P5 消融規格 |
| `docs/p2_dataset_brief.md` | 資料集與 P1 切分的稽核摘要 |
| `docs/p2_prior_work.md` | 同一資料集的公開研究與閱讀狀態 |
| `docs/p2_section4_explanation.md` | §4 的中文逐段解釋與 AI 使用聲明 |
| `docs/p2_eli5_learning_path_zh.md`、`docs/p2_code_walkthrough_zh.md` | 入門說明與程式導讀 |

## 7. 重現

在 Repository 根目錄執行：

```bash
pip install -r EDA/requirements.txt
python EDA/p2_eda.py                                   # 約 1 分鐘，seed 42
python EDA/tools/build_deliverable_notebook.py "$PWD"
jupyter nbconvert --to notebook --execute --inplace EDA/P2_Section4_Deliverable.ipynb
```

`p2_eda.py` 透過 P1 的 `load_split("train")` 讀資料，只會讀到 `data/splits/train.csv`。

## 8. 限制

- 病理片段集中在少數檢查，區間估計很寬；資料沒有孕婦 ID，無法排除同一人多次檢查。
- 標籤是專家共識，不是新生兒結局；排除 Suspect 類別會讓指標偏樂觀。
- 依 DP 分出的兩個病理子群是用了標籤之後的描述，不是臨床分型。
- 偵測結果只能輔助臨床人員判讀，不能當成診斷或治療依據。
