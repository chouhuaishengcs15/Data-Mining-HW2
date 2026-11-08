# 資料與前處理（P1）

大家好，這裡是 P1 整理好的資料和全組共用的切分。後面所有方法（Tier 1／2／3）、EDA、消融和錯誤分析都請用這裡的檔案，這樣大家的數字才能放在同一張表比較。有任何問題或想改切分方式，先在群組跟我說，不要各自重切。

## 一、我們用的資料

- **資料集**：Cardiotocography（CTG，胎心宮縮監測），UCI Machine Learning Repository  
  https://archive.ics.uci.edu/dataset/193/cardiotocography
- **原始出處**：SisPorto 系統（Ayres-de-Campos et al., 2000），W1 寫相關研究時可以引用
- **資料類別**：表格資料（Tabular）
- **偵測單位**：一段 CTG 監測片段（segment），每段有 21 個由胎心率和宮縮訊號算出來的特徵
- **規模**：2126 個片段，來自 352 次檢查（exam，也就是原始檔的 `FileName`），一次檢查平均 6 個片段（1–34 個）
- **標籤**：三位產科專家判讀後的共識分類 NSP
  - Normal（NSP = 1）：1655 筆
  - Suspect（NSP = 2）：295 筆
  - Pathologic（NSP = 3）：176 筆
- **原始檔**：`raw/CTG.xls`（UCI 原檔，沒有改動，我們用裡面的 `Raw Data` 工作表）

## 二、什麼算異常

| label | 意義 | 筆數 |
|---|---|---|
| 0 | Normal，正常 | 1655 |
| 1 | **Pathologic，異常** | 176 |
| -1 | Suspect，臨床灰色地帶，不放進主要任務 | 295 |

- 我們的主要任務是 **Normal vs. Pathologic**，共 1831 筆，**異常比例 9.61%（約 1 : 9.4）**。
- Suspect 我另外存成 `splits/suspect.csv`，不拿來訓練，也不算進主要指標。P5 可以拿來做錯誤分析或消融，例如「把 Suspect 當正常／當異常，結果會怎麼變」。

## 三、特徵說明（21 個，都沒有缺值）

| 特徵 | 意義 | 單位 |
|---|---|---|
| LB | 胎心基準率（baseline FHR） | bpm |
| AC | 加速次數 | 次／秒 |
| FM | 胎動次數 | 次／秒 |
| UC | 宮縮次數 | 次／秒 |
| DL / DS / DP | 輕度 / 重度 / 延長減速次數 | 次／秒 |
| ASTV | 短期變異度異常的時間比例 | % |
| MSTV | 短期變異度平均值 | bpm |
| ALTV | 長期變異度異常的時間比例 | % |
| MLTV | 長期變異度平均值 | bpm |
| Width / Min / Max | 胎心率直方圖的寬度、最小值、最大值 | bpm |
| Nmax / Nzeros | 直方圖峰值數、零值數 | 個 |
| Mode / Mean / Median | 直方圖眾數、平均值、中位數 | bpm |
| Variance | 直方圖變異數 | |
| Tendency | 直方圖偏態（-1 左偏、0 對稱、1 右偏） | |

- AC、FM、UC、DL、DS、DP 原本是片段內的次數，我已經除以片段長度（`e - b` 秒）換成每秒次數，和 UCI `Data` 工作表的數值一模一樣。
- **以下欄位我刻意拿掉了，請不要加回去：**
  - `A`–`E`、`AD`、`DE`、`LD`、`FS`、`SUSP`、`CLASS`：這些是專家標的胎心型態類別，跟 NSP 標籤是同一個來源，放進模型等於偷看答案（標籤洩漏）。
  - `LBE`：和 `LB` 完全相同。
  - `DR`：全部都是 0。
  - `b`、`e`：只是片段的起訖時間。

## 四、正常和異常的例子（取自 train）

| | row_id | LB | AC | UC | DL | DP | ASTV | MSTV | ALTV | Variance |
|---|---|---|---|---|---|---|---|---|---|---|
| 正常 | 1 | 132 | 0.0064 | 0.0064 | 0.0032 | 0 | 17 | 2.1 | 0 | 12 |
| 異常 | 54 | 148 | 0 | 0 | 0 | 0 | 70 | 0.3 | 69 | 0 |

- **正常例**：胎心基準率 132 bpm，有加速，短期變異度異常的時間只佔 17%。
- **異常例**：基準率偏高（148 bpm），完全沒有加速，70% 的時間短期變異度異常、69% 的時間長期變異度異常，直方圖變異數是 0，胎心幾乎沒有起伏。

## 五、資料切分

我是**以「檢查」（exam）為單位**切的：同一次檢查的所有片段只會在同一段。原因是同一位孕婦同一次檢查的片段很像，如果隨機切，訓練和測試會出現幾乎一樣的資料，分數會虛高（資料洩漏）。

切法：依「這次檢查有沒有 Pathologic 片段」分層，固定 seed = 42，按片段數切成約 60 / 20 / 20。

| 切分 | 檢查數 | Normal | Pathologic | 異常比例 | Suspect（另存） |
|---|---|---|---|---|---|
| train | 204 | 972 | 104 | 9.67% | 190 |
| val | 64 | 342 | 35 | 9.28% | 48 |
| test | 84 | 341 | 37 | 9.79% | 57 |
| 全部 | 352 | 1655 | 176 | 9.61% | 295 |


### 檔案在哪裡（`splits/`）

| 檔案 | 內容 |
|---|---|
| `train.csv` / `val.csv` / `test.csv` | `row_id`、`exam`、`label`（0 / 1）＋ 21 個特徵，只有 Normal 和 Pathologic |
| `suspect.csv` | 所有 Suspect 片段：`row_id`、`exam`、`split`（屬於哪一段）＋ 21 個特徵 |
| `split.csv` | 全部 2126 個片段的索引：`row_id`、`exam`、`segment`、`date`、`nsp`、`label`、`split` |
| `stats.csv` | 上面那張切分統計表 |

`row_id` 是 `CTG.xls` 裡 `Raw Data` 工作表的資料列順序（從 0 開始）。如果需要重新產生（要先裝 `pandas`、`xlrd`）：

```bash
python data/make_split.py
```

## 六、使用規則

1. **直接用 `splits/` 的檔案或 `load_data.py`。**
2. **P3（主要模型）**：Isolation Forest、Autoencoder 這類一類／重建型模型，只能用 train 的 972 筆 Normal 訓練，也就是 `load_split("train", use_normal_only=True)`。
3. **P4（Tier 3）**：有監督的方法（例如 XGBoost）可以用 train 裡的 104 筆 Pathologic。
4. **val** 裡的 35 筆 Pathologic 只能用來選門檻 τ 和超參數，不能拿來訓練模型權重。
5. **test 只在最後評估一次。** P2 做 EDA 也只能看 train，不能看 test。
6. 標準化之類的前處理只能用 train 來 fit，再套到 val、test。
7. 如果要在 train 裡面再做交叉驗證，請用 `return_exam=True` 拿到 exam，依 exam 分組切，不然一樣會洩漏。

用 Python 讀資料的範例：

```python
import sys; sys.path.append("data")
from load_data import load_split, fit_preprocess

X_tr, y_tr = load_split("train", use_normal_only=True)
X_va, y_va = load_split("val")
X_te, y_te = load_split("test")
prep = fit_preprocess(X_tr)
X_tr, X_va, X_te = prep(X_tr), prep(X_va), prep(X_te)
```

## 七、大家寫報告時可以注意的地方

- **給 P5、W1**：Pathologic 片段集中在 50 次檢查（train 28、val 9、test 13），其中一次檢查就有 20 個 Pathologic 片段，所以結果容易受少數幾次檢查影響，可以寫進 §9 的限制。
- **給 P3、P4**：不同 seed 只改變模型本身的隨機性，資料切分是固定的。
- **給 P4（§6）**：有 17 次檢查的日期記成 1980 年，看起來是預設值，所以我們沒有依時間切分，而是依檢查分組切分。
