# P2 Handoff — 給 P3、P5、I1（以及 P1 的回報事項）

- 日期：2026-11-10；R3、R5 修訂 2026-11-11。主要交付 Notebook：`EDA/P2_Section4_Deliverable.ipynb`。分支 `p2-feature-eda`（尚未 push，待 P2 本人審查）
- 重現：`pip install -r EDA/requirements.txt && python EDA/p2_eda.py`（約 1 分鐘，seed 42）
- 所有結論只用 train；細節見 `p2_hypotheses.md`

## 給 P3（Primary Model）

**欄位介面**：沿用 `load_data.FEATURES`（21 個）。P2 沒有改任何欄位名稱或資料檔。

| 特徵集合 | 內容 | 用途 |
|---|---|---|
| `ALL21` | `FEATURES` | baseline，請先以此完成主要模型 |
| `DROP_WIDTH20` | `ALL21` 移除 `Width` | H3 的對照：只移除確定性冗餘（Width = Max − Min），沒有資訊損失 |
| `PRUNED18` | `ALL21` 移除 `Width, Mode, Mean` | **H3 主要集合**（R5 起）。規則只用 Normal-train：先移除 Width，其餘 Histogram 特徵依 \|ρ\| ≥ 0.8 分群、每群留一個。啟發式挑選，不是無損精簡（保留特徵線性解釋 Mode、Mean 變異的 93%、94%） |
| `NO_VARIABILITY17` | `ALL21` 移除 `ASTV, MSTV, ALTV, MLTV` | H1 |
| `PRUNED17`（選做） | `ALL21` 移除 `Width, Min, Mode, Mean` | 較早的規則，僅作選做對照；Width 與 Min 同時移除後 Min 無法還原，**不得宣稱無資訊損失** |

以上定義是 GPT-6 R4／R5 審查後的建議，**尚未與 P3／P5 確認**。機器可讀版本：`EDA/outputs/feature_sets.json`（由 `p2_eda.py` 產生）。

```python
DROP_WIDTH20     = [f for f in FEATURES if f != "Width"]
PRUNED18         = [f for f in FEATURES if f not in ("Width", "Mode", "Mean")]
NO_VARIABILITY17 = [f for f in FEATURES if f not in ("ASTV", "MSTV", "ALTV", "MLTV")]
```

**建模時請留意（都是 train 的實測觀察）**
1. **Width = Max − Min 完全相等**，21 個特徵的共變異數秩為 20。Mahalanobis、PCA whitening、高斯密度模型請先移除 Width 或使用 shrinkage；Isolation Forest、Autoencoder 不受影響，只是多一個重複輸入。
2. **DS 在 Normal-train 只有 1 筆非 0**，z-score 的標準差極小；val／test 只要出現一次重度減速，標準化後就是數十個標準差的極端值，可能讓重建誤差或距離被單一特徵主導。可考慮對 z-score 做截斷（clip），但請當成獨立的設計決定並記錄，不要和特徵消融混在一起。
3. **事件率特徵大量為 0**（DP 90%、FM 64%、DL 53%、AC 38%），ALTV 也有 65% 為 0；z-score 後仍然高度偏態。
4. **病理片段依 DP 分成的兩個描述性子群**（DP-absent／DP-present，見 `fig5`）在 MSTV、Width、Variance 上往相反方向偏離正常。只衡量單一方向的分數會漏掉其中一種；請確認異常分數是「偏離程度」。
5. LB 單獨沒有辨識力（|z| AUC 0.37），不建議對它做額外的特徵工程。
6. 若最終採用任何「看過訓練異常標籤」才決定的特徵集合，報告需註明；`PRUNED18` 的規則只用 Normal-train，沒有這個問題。

## 給 P5（Ablation）

主要比較四組：`ALL21`、`NO_VARIABILITY17`（H1）、`DROP_WIDTH20`、`PRUNED18`（H3）；`PRUNED17` 選做。背景與假設見 `p2_hypotheses.md`。

- **驗證狀態：H1、H3 皆為 Pending P5 Verification。** P5 完成後請回報結果，P2 會更新 `P2_Section4_Deliverable.ipynb` 第 6 節與 `report/p2_section4.tex` 最後一段。

- **課程要求**：至少 2 個 controlled ablations，其中至少 1 個是異常偵測專屬（threshold／scoring／contamination）。P2 提供的全是特徵消融，不能取代那一個；建議搭配 P3 既有的 scoring-function 比較。

- 固定：P1 切分、P3 模型與超參數、同一組 seeds、z-score 只用 Normal-train fit。
- 比較指標：validation 的 PR-AUC（主）、AUROC（輔）；F1 需使用同一種門檻規則。
- Test 只對事先指定的設定評估一次，不可用來挑特徵集合。
- 特徵消融與門檻／分數消融請分成兩張表。
- 錯誤分析建議：依 `DP > 0` 把 Pathologic 分兩群報告召回率；另外可以看 `suspect.csv` 的分數分布（P1 已建議）。
- 我們的 train-only 探針（`EDA/outputs/group_probe.csv`）只是用來決定哪些假說值得測，**請不要把它的數字放進 §7–§8**。

## 給 I1（Report Integration）

| 檔案 | 用途 |
|---|---|
| `report/p2_section4.tex` | §4 英文內文（1 張 `figure*`、1 張表）；需要 `natbib` 的 `\citep`／`\citet` |
| `report/p2_refs.bib` | §4 的兩筆 BibTeX（UCI Dataset、Huang et al. 2021） |
| `EDA/P2_Section4_Deliverable.ipynb` | P2 主要交付 Notebook；第 7 節是與 tex 相同的英文內文，7.1 節自動核對數字 |
| `EDA/figures/fig1_variability_contrast.pdf` | §4 的 Figure |
| `EDA/figures/fig5_pathologic_phenotypes.pdf`、`fig4_spearman_heatmap.pdf`、`fig3_separability_ranking.pdf`、`figS1_all_features_ecdf.pdf` | 備用／附錄／簡報 |
| `EDA/outputs/separability.csv` | 21 個特徵完整統計（附錄表） |
| `EDA/docs/p2_section4_explanation.md` | 中文逐段解釋 |
| `slides/p2_slides_notes.md` | 兩頁投影片建議 |
| `EDA/docs/p2_prior_work.md` | Related Work 可用的來源與閱讀狀態 |

- 可寫進 §9 Limitations：Pathologic 集中在少數檢查（2 次檢查佔 train 的 35%）；資料沒有孕婦 ID，無法排除同一人多次檢查；標籤是專家共識而非新生兒結局；排除 Suspect 使指標偏樂觀。
- AI Usage Statement 可用的描述：P2 使用 Claude Code（Claude Fable 5.2）讀取 P1 的程式與資料、撰寫並執行 `EDA/p2_eda.py`、檢索並摘要外部文獻、起草 §4 與相關文件；統計數字皆由程式產生並可重現；外部文獻的閱讀狀態逐項標註，未讀到全文者未引用其內容；假說的取捨由 P2 審查後決定。

## 與 P3 程式的介面（P2 只提需求，不改 P3 的檔案）

以下已對照遠端 `origin/main`（commit `1bf66d5`）的 `model/tier2_autoencoder.py` 與 `model/README.md`（只讀取，未合併、未執行）：P3 目前固定使用 `load_data.FEATURES` 的 21 個欄位，沒有特徵集合參數；已有 `--score mse|cosine|mahalanobis` 與 `--threshold`、`--final` 等參數；公布的結果是 `ALL21` 的 Validation 數字。

- 需求：P3 的訓練腳本能接受特徵集合名稱（例如 `--feature-set ALL21|DROP_WIDTH20|PRUNED18|NO_VARIABILITY17`），並讓輸出的逐特徵誤差檔跟著所選欄位。是否採用、如何實作由 P3 決定。
- 提醒：若最終異常分數是重建誤差的 Mahalanobis 距離，逐特徵的平方重建誤差排名不等於該分數的精確歸因；用它解釋示警時請稱為 reconstruction-error diagnostic。
- 不要為了 P2 的消融執行 P3 的最終測試流程。
- 未另寫 `p2_p3_integration.md`：需求已集中在本節與 `feature_sets.json`。本機分支仍未與 `main` 合併。

## 回報 P1（不需要改切分）

1. README 的「train 204 次檢查」包含 Suspect；`train.csv` 是 175 次。建議註明口徑。
2. README 把 MSTV／MLTV 的單位寫成 bpm；UCI 變數表沒有給單位，建議核對或改寫為「未註明」。
3. `make_split.py` 需要 `xlrd`，repo 沒有 requirements 檔。
4. 核對結果：train 的特徵、標籤、exam 與 `CTG.xls` 重算結果逐列一致；三個 split 之間 exam 無重疊、無跨 split 重複列。
