# 模型與實驗

本資料夾存放模型、訓練、預測與消融實驗程式。

## 分工

- P3：第二層級的主要方法
- P4：第一層級的簡單基準方法、第三層級的強力參考方法
- P5：消融實驗，與 P3 協調需要修改的模型設定

三種方法使用全組共用的資料切分（`data/splits/`）與評估指標。模型調整及門檻選擇只用訓練與驗證資料，test 只在最後評估一次。

---

# Tier 2 主要方法：Autoencoder（P3）

第一版先用全部 21 個特徵做出來。P2 的特徵分析完成後，我依照 P2 的建議**拿掉 Width**（`Width = Max − Min`，是多餘的特徵），現在的主要設定是 20 個特徵（P2 定義的 `DROP_WIDTH20`）。舊的 21 個特徵結果保留在 `*_ALL21` 資料夾，給 P5 當消融的比較基準。

## 一、目前進度

| 項目 | 狀態 |
|---|---|
| 模型程式（訓練、分數、門檻、評估） | 完成 |
| 依 P2 的建議調整特徵 | 完成，改用 `DROP_WIDTH20` |
| 分數函數比較（MSE／cosine／Mahalanobis） | 完成，val 結果在下面 |
| 超參數搜尋 | 完成（20 個特徵重跑過），只用 val |
| 特徵集合、z-score 截斷、共變異數 shrinkage 的選項 | 完成，給 P5 消融用 |
| test 評估 | 完成（2026-11-12），結果在第四節 |
| 報告 §5 方法、簡報第 6 頁 | 完成 |

## 二、方法

```
20 個特徵 → z-score（只用 train 的 Normal fit）→ Autoencoder → 分數 s(x) → s(x) ≥ τ 就發警報
```

- **特徵**：P1 的 21 個特徵拿掉 Width，也就是 P2 的 `DROP_WIDTH20`（定義在 `EDA/outputs/feature_sets.json`）。
- **訓練資料**：只用 train 的 972 筆 Normal，也就是 `load_split("train", use_normal_only=True)`。
- **模型**：20 → 16 → 8 → 16 → 20，ReLU，瓶頸層和輸出層是線性的。
- **訓練**：Adam，lr 1e-3，weight decay 1e-5，batch 64，200 輪，loss 是 MSE。每個 seed 在 CPU 上約 2 秒。
- **分數 s(x)**：先算重建誤差 e = x − x̂（20 維），再用 **Mahalanobis 距離**：

  s(x) = sqrt( (e − μ)ᵀ Σ⁻¹ (e − μ) )

  μ、Σ 是 train 正常資料重建誤差的平均和共變異數。意思是「這筆資料的重建誤差，跟正常資料的重建誤差比起來有多不尋常」。
- **門檻 τ**：取 val 上 **F2** 最高的分數（理由見下面「門檻 τ 的選法」）。val 的異常標籤只用在這裡和選超參數，不拿來訓練。

## 三、目前結果（val，5 個 seed，mean ± std）

異常比例約 9.3%，所以隨便猜的 PR-AUC 大約是 0.09。

| 分數函數 s(x) | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| MSE（重建誤差平方的平均） | 0.543 ± 0.057 | 0.896 ± 0.031 | 0.604 ± 0.054 |
| **Mahalanobis（主要設定）** | **0.734 ± 0.074** | **0.952 ± 0.033** | **0.698 ± 0.061** |
| Cosine | 0.089 ± 0.016 | 0.459 ± 0.074 | 0.191 ± 0.016 |

- Mahalanobis 比 MSE 高約 0.19，遠大於 seed 之間的變動。
- Cosine 和隨便猜差不多。它只看 x 和 x̂ 的方向、不看大小，但 Pathologic 的特徵是數值大幅偏離正常範圍，方向不一定變。這是可以寫進報告的負面結果。
- 主要設定下 precision 0.60、recall 0.85、誤報率 5.8%；每個 seed 在 val 平均發出 50 個警報（val 有 35 筆異常）。

### 為什麼拿掉 Width

| 設定 | PR-AUC | ROC-AUC | F1 | 重建誤差共變異數的條件數 |
|---|---|---|---|---|
| 21 個特徵（`ALL21`） | 0.675 ± 0.052 | 0.944 ± 0.020 | 0.667 ± 0.033 | 約 21,000 |
| **拿掉 Width（`DROP_WIDTH20`）** | **0.734 ± 0.074** | **0.952 ± 0.033** | **0.698 ± 0.061** | **約 540** |

- `Width = Max − Min` 完全成立，Width 沒有提供新資訊，卻讓 Mahalanobis 用到的共變異數矩陣接近退化（條件數約 21,000）。21 個特徵時，Min 一個特徵就佔了 val 分數的 28%；拿掉 Width 後，每個特徵的佔比都在 10% 以下。
- PR-AUC 平均高 0.06，但 5 個 seed 裡有 1 個反而下降 0.115，所以**不能說拿掉 Width 顯著比較好**。選它的理由是數值比較穩定、沒有資訊損失。

### 試過但沒有採用的做法

依照 P2 的提醒，我也在 val 上試了兩種做法，都沒有幫助：

| 做法（都用 `DROP_WIDTH20`） | PR-AUC |
|---|---|
| 主要設定 | 0.734 ± 0.074 |
| z-score 截斷在 ±10 | 0.682 ± 0.065 |
| z-score 截斷在 ±5 | 0.596 ± 0.109 |
| 共變異數用 Ledoit-Wolf shrinkage | 0.722 ± 0.069 |

- **截斷**：每種特徵集合都變差。極端值（例如 DP 達 12 個標準差）本身就是異常訊號，截斷會把它削掉。
- **Shrinkage**：拿掉 Width 後共變異數已經穩定了，所以沒有額外幫助。
- P2 提到的 DS 問題（train 正常資料只有 1 筆 DS 不是 0，標準差極小）在 val 上沒有出現，因為 val 沒有 DS 不是 0 的資料，但 test 有可能會出現。這可以寫進限制。

### 門檻 τ 的選法

Tier 2 的主規則是 **val 上 F2 最高的值**，F1 規則當對照（第四節兩個都有報）。F1 把漏報和誤報看得一樣重；F2 比較重視 recall，比較符合我們的情境：漏掉 Pathologic（胎兒窘迫）的代價，比多一次誤報、請醫師多看一次大得多。

模型不變，只換選 τ 的規則，在 val 上的差異（5 個 seed 平均，val 有 35 筆異常、342 筆正常）：

| τ 的選法 | 漏報 | 誤報 | Recall | Precision | F1 | F2 |
|---|---|---|---|---|---|---|
| F1 最高（對照） | 5.4 | 20.0 | 0.85 | 0.60 | 0.70 | 0.78 |
| **F2 最高（主規則）** | **3.8** | **26.8** | **0.89** | **0.55** | **0.67** | **0.79** |
| Recall 至少 90% | 3.0 | 52.4（seed 間 ±48） | 0.91 | 0.46 | 0.59 | 0.74 |
| Recall 至少 95% | 1.0 | 64.0（seed 間 ±57） | 0.97 | 0.42 | 0.57 | 0.74 |

- 改用 F2：平均少漏 1.6 筆異常，多 7 個誤報。
- 固定 recall 的規則漏報最少，但 val 的異常只有 35 筆，誤報數在 seed 之間差到 ±50，太不穩定，所以不建議。
- 這些數字是在 val 上選 τ、又在 val 上算的，會偏樂觀，實際表現要看 test。
- 三種方法要用同一個規則選 τ，F1、F2、recall 這些需要門檻的指標才能公平比較，請 Tier 1、Tier 3 也用 F2 當主規則。
- 程式的 `--threshold` 預設仍是 `best_f1`（第三節的 val 表格是用它跑的），跑主要設定時請加 `--threshold best_f2`。

### 超參數搜尋

搜尋範圍：模型大小 8-4、16-4、16-8、32-8、32-16、32-16-8 × 訓練輪數 50、100、200、400，共 24 組，每組 5 個 seed，分數用 Mahalanobis，特徵用 `DROP_WIDTH20`，只看 val。

| hidden | 訓練輪數 | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|---|
| 16-8 | 400 | 0.736 ± 0.057 | 0.957 | 0.701 |
| **16-8（選用）** | **200** | **0.734 ± 0.074** | **0.952** | **0.698** |
| 16-8 | 100 | 0.668 ± 0.028 | 0.934 | 0.660 |
| 32-16 | 200 | 0.665 ± 0.032 | 0.930 | 0.669 |
| 8-4 | 200 | 0.646 ± 0.104 | 0.911 | 0.685 |

選 200 輪而不是 400 輪，是因為兩者只差 0.002，遠小於 seed 之間的變動，而 200 輪的時間只要一半。用 21 個特徵搜尋時，結論也是 16-8、200 輪。完整結果在 `results/tier2/hp_search_mahalanobis_DROP_WIDTH20.csv`（21 個特徵的版本是 `hp_search_mahalanobis_ALL21.csv`）。

## 四、test 結果（只評估一次）

設定全部只用 train、val 決定後，用以下指令評估 test 一次（2026-11-12）：

```bash
python model/tier2_autoencoder.py --threshold best_f2 --final --name final_mahalanobis_best_f2_DROP_WIDTH20
```

結果在 `results/tier2/final_mahalanobis_best_f2_DROP_WIDTH20/`。test 有 37 筆 Pathologic、341 筆 Normal。

| | val | test |
|---|---|---|
| PR-AUC | 0.734 ± 0.074 | 0.565 ± 0.186 |
| ROC-AUC | 0.952 ± 0.033 | 0.878 ± 0.085 |
| F2 規則（主）：漏報／誤報 | 3.8 / 26.8 | 15.2 / 28.2 |
| F1 規則（對照）：漏報／誤報 | 5.4 / 20.0 | 18.4 / 15.2 |

各 seed 的 test 結果（F2 規則）：

| seed | PR-AUC | ROC-AUC | F2 | Precision | Recall | 警報數 |
|---|---|---|---|---|---|---|
| 0 | 0.643 | 0.928 | 0.665 | 0.625 | 0.676 | 40 |
| 1 | 0.381 | 0.758 | 0.271 | 0.500 | 0.243 | 18 |
| 2 | 0.363 | 0.818 | 0.356 | 0.208 | 0.432 | 77 |
| 3 | 0.792 | 0.951 | 0.740 | 0.604 | 0.784 | 48 |
| 4 | 0.647 | 0.934 | 0.698 | 0.448 | 0.811 | 67 |

- test 比 val 低很多（PR-AUC 0.73 → 0.57），而且 seed 之間差很大。val 的異常只來自 9 次檢查，τ 和超參數又是在 val 上選的，所以 val 的數字偏樂觀，這和報告 §5.2 的說法一致。
- F2 規則在 test 上仍然比 F1 規則少漏報（15.2 對 18.4），代價是多約 13 個誤報，取捨的方向和 val 一樣；但 seed 2 用 F2 規則時誤報達 61 筆，τ 在 seed 之間不穩定。
- 37 筆 test 異常裡，有 3 筆在 5 個 seed 都被漏掉，14 筆在至少 3 個 seed 被漏掉，可以從這些開始做錯誤分析。
- test 也沒有 DS 不是 0 的資料，所以第三節提到的 DS 問題在 test 上沒有發生。
- `threshold_comparison.csv` 列出每個 seed 在兩種規則下的 τ、漏報與誤報；兩種規則用的是同一份分數，只差 τ。
- 這是唯一一次 test 評估。用一模一樣的設定重跑會得到相同結果；但不能看了 test 再改設定重跑。

## 五、怎麼執行

需要 `pandas`、`numpy`、`scikit-learn`、`torch`。在 repo 根目錄執行：

```bash
python model/tier2_autoencoder.py --threshold best_f2            # 主要設定，只看 val
python model/tier2_autoencoder.py --score mse                    # 換分數函數
python model/tier2_autoencoder.py --feature-set NO_VARIABILITY17 # 換特徵集合
python model/tier2_autoencoder.py --contamination 0.05           # 訓練資料混入 5% 異常
python model/tier2_autoencoder.py --threshold percentile         # 換門檻策略
python model/hp_search.py                                        # 超參數搜尋
# test 已經評估過了，指令見第四節
```

| 參數 | 意義 | 預設 |
|---|---|---|
| `--feature-set` | P2 的特徵集合：`DROP_WIDTH20`、`ALL21`、`PRUNED18`、`NO_VARIABILITY17`、`PRUNED17` | `DROP_WIDTH20` |
| `--score` | 分數函數：`mse`、`cosine`、`mahalanobis` | `mahalanobis` |
| `--threshold` | 門檻策略：`best_f1`（val 上 F1 最高）、`best_f2`（val 上 F2 最高，比較重視 recall）、`percentile`（train 正常分數的第 q 百分位，不用標籤）、`budget`（val 分數最高的 k% 發警報） | `best_f1` |
| `--percentile` / `--budget` | 上面兩種策略的 q、k | 95 / 10 |
| `--contamination` | 訓練資料裡異常的比例，從 train 的 104 筆 Pathologic 抽 | 0 |
| `--clip` | z-score 截斷在 ±clip，0 表示不截斷 | 0 |
| `--cov` | Mahalanobis 的共變異數估計：`empirical`、`ledoit_wolf` | `empirical` |
| `--hidden`、`--epochs`、`--lr`、`--weight-decay`、`--batch-size` | 模型和訓練設定 | 16 8、200、1e-3、1e-5、64 |
| `--seeds` | 要跑哪些 seed | 0 1 2 3 4 |
| `--final` | 也評估 test。Tier 2 已用過一次（第四節）；**不要改了設定再加** | 不加 |
| `--name` | 輸出資料夾名稱 | 依設定自動命名 |

## 六、輸出檔（`results/tier2/<設定名稱>/`）

| 檔案 | 內容 |
|---|---|
| `config.json` | 這次執行的所有設定 |
| `metrics.csv` | 每個 seed、每個切分一列：PR-AUC、ROC-AUC、F1、F2、precision、recall、誤報率、警報數、τ、訓練秒數、每筆預測毫秒數 |
| `summary.csv` | 上面各指標在 seed 之間的平均和標準差 |
| `scores.csv` | 每筆資料一列：`row_id`、`seed`、`split`、`label`、`score`、`tau`、`pred` |
| `feature_errors.csv` | 每筆資料每個特徵的重建誤差平方：`row_id`、`seed`、`split` ＋ 所用的特徵 |
| `threshold_comparison.csv` | 只有 final 資料夾有：每個 seed、每個切分在 F2／F1 兩種門檻規則下的 τ、F1、F2、precision、recall、漏報數、誤報數 |

`row_id` 和 `data/splits/` 裡的一樣，可以直接對回原始特徵。資料夾名稱的最後一段是特徵集合，目前放上來的有：

| 資料夾 | 內容 |
|---|---|
| `final_mahalanobis_best_f2_DROP_WIDTH20` | **最終結果**：主要設定（τ 用 F2），含 val 與 test |
| `mahalanobis_best_f1_c0_h16-8_DROP_WIDTH20` | 主要設定，τ 用 F1，只有 val（第三節的表格） |
| `mahalanobis_best_f2_c0_h16-8_DROP_WIDTH20` | 主要設定，τ 用 F2，只有 val |
| `mse_best_f1_c0_h16-8_DROP_WIDTH20`、`cosine_best_f1_c0_h16-8_DROP_WIDTH20` | 分數函數比較 |
| `*_ALL21` | 第一版（21 個特徵）的三種分數函數，給 P5 當特徵消融的基準 |

## 七、給其他人的說明

- **給 P2**：已經依你的建議拿掉 Width，數字在第三節。`--feature-set` 會直接讀你的 `feature_sets.json`。截斷和 shrinkage 在 val 上都沒有幫助，所以沒有採用。
- **給 P4**：
  - 主要表格請用 `final_mahalanobis_best_f2_DROP_WIDTH20`（val 和 test 都有），數字整理在第四節。
  - Tier 2 的 τ 主規則是 val 上 F2 最高，F1 當對照；Tier 1、Tier 3 請用同樣的主規則。
  - Tier 1、Tier 3 如果也輸出成 `scores.csv` 的格式，做比較表、顯著性檢定（例如對各 seed 做配對 t 檢定，或對 test 做 bootstrap）會比較方便。
  - 訓練時間和每筆預測時間在 `metrics.csv` 的 `train_s`、`ms_per_sample`。
- **給 P5**：
  - 消融建議：
    - (1) 分數函數 MSE vs. cosine vs. Mahalanobis，已經有結果，屬於異常偵測專屬的消融。
    - (2) 訓練資料混入 0／1／5／10% 的異常，用 `--contamination` 跑。
    - P2 的特徵消融用 `--feature-set` 跑，其他設定都用預設值，就只差特徵集合。**特徵消融的基準照 P2 的設計用 `ALL21`**：H1 是 `ALL21` vs `NO_VARIABILITY17`，H3 是 `ALL21` vs `DROP_WIDTH20` vs `PRUNED18`。`NO_VARIABILITY17` 裡還有 Width，如果拿 `DROP_WIDTH20` 當基準，就會同時差兩個地方。其他消融（分數函數、訓練資料混入異常）則用主要設定 `DROP_WIDTH20` 當基準。
  - `--contamination 0.10` 需要 108 筆異常，但 train 只有 104 筆，程式會用 104 筆並印出提醒。
  - 錯誤案例：在 `final_mahalanobis_best_f2_DROP_WIDTH20/scores.csv` 找 `split=test` 中 `label=1, pred=0`（漏報）和 `label=0, pred=1`（誤報）。`pred` 是 F2 規則；要用 F1 規則，從 `threshold_comparison.csv` 取 τ 重算。
  - 模型解釋：`feature_errors.csv` 可以看出一筆資料是哪幾個特徵重建得特別差。依 P2 的提醒，報告裡請稱為 reconstruction-error diagnostic，它不是 Mahalanobis 分數的精確拆解。
- **給 W1**：val 的 35 筆異常只來自 9 次檢查，所以超參數之間、特徵集合之間的小差距不太可信，可以寫進限制。
