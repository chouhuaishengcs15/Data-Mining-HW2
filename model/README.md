# 模型與實驗

本資料夾存放模型、訓練、預測與消融實驗程式。

## 分工

- P3：第二層級的主要方法
- P4：第一層級的簡單基準方法、第三層級的強力參考方法
- P5：消融實驗，與 P3 協調需要修改的模型設定

三種方法使用全組共用的資料切分（`data/splits/`）與評估指標。模型調整及門檻選擇只用訓練與驗證資料，test 只在最後評估一次。

---

# Tier 2 主要方法：Autoencoder（P3）

P2 的特徵分析還沒完成，但時間有點趕，所以我先用 P1 整理好的全部 21 個特徵做出第一版，讓 P4、P5 有程式和結果可以先參考、先寫。**目前的數字都是暫定的**，等 P2 完成後，如果有建議的特徵轉換或刪減，我會重跑並更新這份說明和 `results/tier2/`。

## 一、目前進度

| 項目 | 狀態 |
|---|---|
| 模型程式（訓練、分數、門檻、評估） | 完成第一版 |
| 分數函數比較（MSE／cosine／Mahalanobis） | 完成，val 結果在下面 |
| 超參數搜尋 | 完成，只用 val |
| 依 P2 的建議調整特徵 | 等 P2 |
| test 評估 | 還沒跑，等設定全部定案後只跑一次 |
| 報告 §5 方法、簡報 | 進行中 |

## 二、方法

```
21 個特徵 → z-score（只用 train 的 Normal fit）→ Autoencoder → 分數 s(x) → s(x) ≥ τ 就發警報
```

- **訓練資料**：只用 train 的 972 筆 Normal，也就是 `load_split("train", use_normal_only=True)`。
- **模型**：21 → 16 → 8 → 16 → 21，ReLU，瓶頸層和輸出層是線性的。
- **訓練**：Adam，lr 1e-3，weight decay 1e-5，batch 64，200 輪，loss 是 MSE。每個 seed 在 CPU 上約 2 秒。
- **分數 s(x)**：先算重建誤差 e = x − x̂（21 維），再用 **Mahalanobis 距離**：

  s(x) = sqrt( (e − μ)ᵀ Σ⁻¹ (e − μ) )

  μ、Σ 是 train 正常資料重建誤差的平均和共變異數。意思是「這筆資料的重建誤差，跟正常資料的重建誤差比起來有多不尋常」。
- **門檻 τ**：取 val 上 F1 最高的分數。val 的異常標籤只用在這裡和選超參數，不拿來訓練。

## 三、目前結果（val，5 個 seed，mean ± std）

異常比例約 9.3%，所以隨便猜的 PR-AUC 大約是 0.09。

| 分數函數 s(x) | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| MSE（重建誤差平方的平均） | 0.503 ± 0.060 | 0.865 ± 0.028 | 0.565 ± 0.054 |
| **Mahalanobis（主要設定）** | **0.675 ± 0.052** | **0.944 ± 0.020** | **0.667 ± 0.033** |
| Cosine | 0.081 ± 0.015 | 0.414 ± 0.063 | 0.187 ± 0.025 |

- Mahalanobis 比 MSE 高約 0.17，遠大於 seed 之間的變動。
- Cosine 比隨便猜還差。它只看 x 和 x̂ 的方向、不看大小，但 Pathologic 的特徵是數值大幅偏離正常範圍，方向不一定變。這是可以寫進報告的負面結果。
- 主要設定下的 precision 0.62、recall 0.74，每個 seed 在 val 平均發出 43 個警報（val 有 35 筆異常）。

### 超參數搜尋

搜尋範圍：模型大小 8-4、16-4、16-8、32-8、32-16、32-16-8 × 訓練輪數 50、100、200、400，共 24 組，每組 5 個 seed，分數固定用 Mahalanobis，只看 val。

| hidden | 訓練輪數 | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|---|
| 16-8 | 400 | 0.684 ± 0.060 | 0.941 | 0.678 |
| **16-8（選用）** | **200** | **0.675 ± 0.052** | **0.944** | **0.667** |
| 16-8 | 100 | 0.654 ± 0.098 | 0.937 | 0.663 |
| 32-16 | 100 | 0.639 ± 0.050 | 0.942 | 0.651 |
| 8-4 | 50 | 0.628 ± 0.051 | 0.948 | 0.670 |

選 200 輪而不是 400 輪，是因為兩者只差 0.009，比 seed 之間的變動（約 0.05）小很多，而 200 輪比較穩定、時間也只要一半。完整結果在 `results/tier2/hp_search_mahalanobis.csv`。

## 四、怎麼執行

需要 `pandas`、`numpy`、`scikit-learn`、`torch`。在 repo 根目錄執行：

```bash
python model/tier2_autoencoder.py                                # 主要設定，只看 val
python model/tier2_autoencoder.py --score mse                    # 換分數函數
python model/tier2_autoencoder.py --contamination 0.05           # 訓練資料混入 5% 異常
python model/tier2_autoencoder.py --threshold percentile         # 換門檻策略
python model/hp_search.py                                        # 超參數搜尋
python model/tier2_autoencoder.py --final                        # 最後才跑：評估 test
```

| 參數 | 意義 | 預設 |
|---|---|---|
| `--score` | 分數函數：`mse`、`cosine`、`mahalanobis` | `mahalanobis` |
| `--threshold` | 門檻策略：`best_f1`（val 上 F1 最高）、`percentile`（train 正常分數的第 q 百分位，不用標籤）、`budget`（val 分數最高的 k% 發警報） | `best_f1` |
| `--percentile` / `--budget` | 上面兩種策略的 q、k | 95 / 10 |
| `--contamination` | 訓練資料裡異常的比例，從 train 的 104 筆 Pathologic 抽 | 0 |
| `--hidden`、`--epochs`、`--lr`、`--weight-decay`、`--batch-size` | 模型和訓練設定 | 16 8、200、1e-3、1e-5、64 |
| `--seeds` | 要跑哪些 seed | 0 1 2 3 4 |
| `--final` | 也評估 test。**全組設定定案前請不要加** | 不加 |
| `--name` | 輸出資料夾名稱 | 依設定自動命名 |

## 五、輸出檔（`results/tier2/<設定名稱>/`）

| 檔案 | 內容 |
|---|---|
| `config.json` | 這次執行的所有設定 |
| `metrics.csv` | 每個 seed、每個切分一列：PR-AUC、ROC-AUC、F1、precision、recall、誤報率、警報數、τ、訓練秒數、每筆預測毫秒數 |
| `summary.csv` | 上面各指標在 seed 之間的平均和標準差 |
| `scores.csv` | 每筆資料一列：`row_id`、`seed`、`split`、`label`、`score`、`tau`、`pred` |
| `feature_errors.csv` | 每筆資料每個特徵的重建誤差平方：`row_id`、`seed`、`split` ＋ 21 個特徵 |

`row_id` 和 `data/splits/` 裡的一樣，可以直接對回原始特徵。目前放上來的有三個資料夾，分別是三種分數函數：`mahalanobis_best_f1_c0_h16-8`（主要設定）、`mse_best_f1_c0_h16-8`、`cosine_best_f1_c0_h16-8`。

## 六、給其他人的說明

- **給 P2**：你的特徵分析完成後，如果建議取 log、刪掉高度相關的特徵（例如 Mode、Mean、Median）或新增特徵，跟我說一聲，我改完會重跑。「全部特徵 vs. 你建議的特徵」也可以當成一個消融。
- **給 P4**：
  - 主要表格請用 `mahalanobis_best_f1_c0_h16-8`。test 的數字我最後會用 `--final` 補上。
  - Tier 1、Tier 3 如果也輸出成 `scores.csv` 的格式，做比較表、顯著性檢定（例如對各 seed 做配對 t 檢定，或對 test 做 bootstrap）會比較方便。
  - 訓練時間和每筆預測時間在 `metrics.csv` 的 `train_s`、`ms_per_sample`。
- **給 P5**：
  - 消融建議：(1) 分數函數 MSE vs. cosine vs. Mahalanobis，已經有結果；(2) 訓練資料混入 0／1／5／10% 的異常，用 `--contamination` 跑。門檻策略也可以，用 `--threshold`。
  - `--contamination 0.10` 需要 108 筆異常，但 train 只有 104 筆，程式會用 104 筆並印出提醒。
  - 錯誤案例：在 `scores.csv` 找 `label=1, pred=0`（漏報）和 `label=0, pred=1`（誤報）。
  - 模型解釋：`feature_errors.csv` 可以看出一筆資料是哪幾個特徵重建得特別差。
- **給 W1**：val 的 35 筆異常只來自 9 次檢查，所以超參數之間的小差距不太可信，可以寫進限制。
