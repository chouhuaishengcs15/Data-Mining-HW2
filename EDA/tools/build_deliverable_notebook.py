"""Builds EDA/P2_Section4_Deliverable.ipynb. Section 7's English text is converted
from report/p2_section4.tex so the two cannot drift apart."""
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1])
tex = (ROOT / "report" / "p2_section4.tex").read_text(encoding="utf-8")


def tex2md(t: str) -> str:
    t = re.sub(r"(?m)^%.*\n", "", t)
    rep = [
        (r"\citep{campos2000ctg}", "(Campos and Bernardes, 2000)"),
        (r"\citet{huang2021interpretability}", "Huang et al. (2021)"),
        (r"Figure~\ref{fig:variability}", "Figure 1"), (r"Table~\ref{tab:sep}", "Table 1"),
        (r"2{,}000", "2,000"), (r"\%", "%"), (r"vs.\ ", "vs. "), (r"\,", " "),
        (r"\S7--\S8", "§7–§8"), (r"\S8", "§8"), ("~", " "),
    ]
    for a, b in rep:
        t = t.replace(a, b)
    t = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", t)
    t = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", t)
    t = re.sub(r"(?<=\d)--(?=\d)", "–", t)
    t = t.replace("Kolmogorov--Smirnov", "Kolmogorov–Smirnov").replace("ASTV--MSTV", "ASTV–MSTV").replace("ALTV--MLTV", "ALTV–MLTV")
    t = re.sub(r"\$\|z\|\$", "|z|", t).replace("$--$", "$–$")
    return " ".join(t.split())


body = re.sub(r"\\begin\{figure\*\}.*?\\end\{figure\*\}", "", tex, flags=re.S)
body = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", body, flags=re.S)
paras = dict(re.findall(r"\\paragraph\{([^}]*)\}\n(.*?)(?=\\paragraph|\Z)", body, flags=re.S))
assert list(paras) == ["Analysis protocol.", "Distribution contrast and separability.",
                       "Feature redundancy.", "Exploratory hypotheses."], list(paras)
P = {k: tex2md(v) for k, v in paras.items()}
fig_cap = tex2md(re.search(r"\\caption\{(.*?)\}\s*\\label\{fig:variability\}", tex, flags=re.S).group(1))
tab_cap = tex2md(re.search(r"\\caption\{(.*?)\}\s*\\label\{tab:sep\}", tex[tex.index(r"\begin{table}"):], flags=re.S).group(1))
rows = re.findall(r"(?m)^(\w+)(?: \([^)]*\)|\$\^\{\\dagger\}\$)?\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*(\.\d+ \[[^\]]+\])\s*&\s*(\.\d+)\s*&\s*(\.\d+)\s*&\s*(\.\d+)", tex)
assert len(rows) == 8, rows
unit = {"ASTV": "ASTV (%)", "DP": "DP†", "Mean": "Mean (bpm)", "AC": "AC†", "ALTV": "ALTV (%)", "LB": "LB (bpm)"}
tab_md = "| Feature | Median N | Median P | KS [95% CI] | AUC | \\|z\\| | MI |\n|---|---|---|---|---|---|---|\n" + \
    "\n".join("| " + " | ".join([unit.get(r[0], r[0]), *r[1:]]) + " |" for r in rows)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# P2｜§4 Exploratory Feature Analysis — 正式交付 Notebook

**課程**：Data Mining HW02 Anomaly Detection　**角色**：P2（資料探索與特徵設計，書面報告 §4）

這份 Notebook 回答 §4 的核心問題：**在建立任何模型之前，Normal 與 Pathologic 的特徵分布暗示了什麼？**
每一節的順序固定為「目的 → 程式碼 → 圖表／結果 → 數據解讀 → 研究意義或限制」。

| 節 | 內容 | 對應評分項目 |
|---|---|---|
| 1 | Task & Dataset | 資料邊界、只用 Training Portion |
| 2 | Distribution Contrast | 至少一張 Normal／Pathologic 分布對照圖 |
| 3 | Feature Separability | 量化可分性（KS、MI、Wasserstein） |
| 4 | Feature Redundancy | 檢查候選特徵是否重複 |
| 5–6 | Exploratory Hypotheses 與驗證狀態 | 提出 H1、H3，交 P5 於 §7–§8 驗證 |
| 7 | Paper Report §4（英文） | §4 內文 |
| 8 | References、重現方式、Checklist | — |

**重現方式**（在 Repository Root 執行，約 1 分鐘）：

```bash
pip install -r EDA/requirements.txt
jupyter nbconvert --to notebook --execute --inplace EDA/P2_Section4_Deliverable.ipynb
```

所有統計由 `EDA/p2_eda.py` 的函式重新計算，不是貼上的數字。全程只讀 `data/splits/train.csv`。
""")

md(r"""
## 1. Task & Dataset

**目的**：說明偵測情境、一筆資料代表什麼，並確認本 Notebook 只使用 P1 固定的 Training Portion。

Cardiotocography（CTG）在產檢時同時記錄胎兒心率（FHR）與子宮收縮。資料來自 UCI Cardiotocography：每一筆是一個 **CTG Segment**，由 SisPorto 軟體算出 21 個數值特徵，並由產科專家標註 NSP。
本組的二元任務是 **Normal（NSP=1）= 0、Pathologic（NSP=3）= 1**；Suspect（NSP=2）不納入主要任務。偵測結果只用來提醒醫護人員優先複查，不是診斷。
""")

code(r"""
import hashlib, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

# 找到 Repository Root（不論從 root 或 EDA/ 啟動都可以）
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "load_data.py").exists())
sys.path.insert(0, str(ROOT / "EDA"))

# 記錄這份 Notebook 讀過的每一個 CSV，最後用來證明沒有讀 Validation / Test
READ_LOG = []
_read_csv = pd.read_csv
def _logged_read_csv(path, *args, **kwargs):
    READ_LOG.append(Path(str(path)).resolve().relative_to(ROOT).as_posix())
    return _read_csv(path, *args, **kwargs)
pd.read_csv = _logged_read_csv

import p2_eda as p2                      # 既有的分析函式；內部呼叫 P1 的 load_split("train")
from p2_eda import FEATURES, FAMILY, FIG, OUT

pd.set_option("display.width", 200, "display.max_columns", 30)
df = p2.add_derived(p2.load_train())     # 21 個特徵 + label + exam；另加 3 個只供比較的加總欄位
print("讀取的檔案：", READ_LOG)
""")

code(r"""
path = df[df.label == 1]
per_exam = path.exam.value_counts()
summary = pd.DataFrame({"數值": {
    "Segments（Train）": len(df),
    "Normal (label 0)": int((df.label == 0).sum()),
    "Pathologic (label 1)": int((df.label == 1).sum()),
    "Pathologic 比例": f"{df.label.mean():.1%}",
    "Examinations": df.exam.nunique(),
    "含 Pathologic 的 Examinations": path.exam.nunique(),
    "最大兩個 Exam 的 Pathologic Segments": f"{int(per_exam.iloc[:2].sum())} / {len(path)}",
    "候選特徵數": len(FEATURES),
    "缺值": int(df[FEATURES].isna().sum().sum()),
}})
display(summary)
display(pd.DataFrame({"features": {k: ", ".join(v) for k, v in FAMILY.items()}}))
assert (len(df), int((df.label == 0).sum()), int((df.label == 1).sum()), len(FEATURES)) == (1076, 972, 104, 21)
""")

md(r"""
**解讀**

- Training Portion 有 972 筆 Normal 與 104 筆 Pathologic（9.7%），共 21 個候選特徵，分成 baseline、events、variability、histogram 四群，沒有缺值。
- 同一次檢查（Exam）會切出多個 Segment，彼此很像。P1 以 Exam 為單位切分 Train／Val／Test，避免同一次檢查的片段同時出現在訓練與測試，造成 Data Leakage。
- 104 筆 Pathologic 只來自 28 次檢查，其中兩次就佔 36 筆。所以後面的信賴區間以「整次檢查」為單位重抽（Examination-level Bootstrap），不把 104 筆當成獨立樣本。

**限制**：標籤在這裡只用來描述分布。One-class 模型（P3 的 Autoencoder）仍只用 Normal 訓練。

本節用到的三個特徵：**ASTV** 是短期變異度異常的時間百分比；**DP** 是延長減速事件的發生率（次／秒）；**MSTV** 是短期變異度的平均值（UCI 未註明單位）。
""")

md(r"""
## 2. Distribution Contrast

**目的**：用圖直接比較 Normal 與 Pathologic 在四個 FHR Variability 特徵（ASTV、ALTV、MSTV、MLTV）上的分布。

先計算 Section 3 會用到的統計（KS 等），因為圖的標題會顯示 KS。下面的程式重新產生 Figure 1 並直接顯示。
""")

code(r"""
cols = FEATURES + p2.DERIVED
sep = p2.separability(df, cols)                    # KS、MI、Wasserstein、Median 等
sep = sep.join(p2.exam_bootstrap(df, cols))        # Examination-level Bootstrap（2000 次，seed 42）
sep["KS_wo_top2"] = p2.leave_top_exams_out(df, cols, 2)

def regenerate(name, draw):
    # 重新畫圖，並回報 PNG 是否與 Repository 中既有的檔案逐位元組相同
    f = FIG / f"{name}.png"
    before = hashlib.md5(f.read_bytes()).hexdigest() if f.exists() else None
    draw()
    return before == hashlib.md5(f.read_bytes()).hexdigest()

FIG_SAME = {"fig1_variability_contrast": regenerate("fig1_variability_contrast", lambda: p2.fig_variability(df, sep))}
display(Image(filename=str(FIG / "fig1_variability_contrast.png"), width=980))
""")

md(r"""
**Figure 1**　上排：Histogram，各類別自己加總為 1（避免 972 對 104 的不平衡讓 Pathologic 看不見）。下排：ECDF（小於等於某個值的累積比例）；黑色直線是兩條 ECDF 的最大垂直差距，也就是 KS Statistic。藍色為 Normal，橘色虛線為 Pathologic。

**1. 分布有什麼不同？**
ASTV 的 Pathologic 整體偏右（數值較高）。MLTV 的 Pathologic 大量集中在 0。MSTV 的 Pathologic 在低值與高值兩側各有一處集中。ALTV 兩群大多是 0，只有部分 Pathologic 很高。

**2. 哪些差異有統計證據？**
ASTV 中位數 40 對 65，KS = 0.71（95% CI 0.56–0.83）。MLTV = 0 的比例是 Normal 4% 對 Pathologic 51%，KS = 0.54。MSTV 的 KS 只有 0.36，ALTV 只有 0.22（數值見 Section 3 的表）。

**3. 單靠這張圖無法確認什麼？**
這是單一特徵的分布比較，不代表特徵放進 Autoencoder 後偵測效果會變好。兩群有重疊，任何單一門檻都會誤判。這裡也只看 Training Portion，尚未在 Validation 上驗證。
""")

md(r"""
### 2.1 事件型特徵：DP 出現率

**目的**：加速（AC）與減速（DL、DS、DP）是「事件發生率」，大部分 Segment 是 0，Histogram 不好讀。改比較「至少出現一次」的比例。
""")

code(r"""
pres = p2.presence_table(df)                       # 出現率與 Examination-level Bootstrap CI
FIG_SAME["fig2_events_contrast"] = regenerate("fig2_events_contrast", lambda: p2.fig_events(df, pres, sep))
display(Image(filename=str(FIG / "fig2_events_contrast.png"), width=980))
t = pres[pres.feature.isin(["AC", "DL", "DS", "DP"])].copy()
t["share_with_event"] = (100 * t.share_with_event).round(1)
t["95% CI"] = [f"[{100*a:.1f}, {100*b:.1f}]" for a, b in zip(t.ci_lo, t.ci_hi)]
display(t[["feature", "class", "n", "n_with_event", "share_with_event", "95% CI"]].rename(
    columns={"share_with_event": "出現率 (%)"}).set_index(["feature", "class"]))
""")

md(r"""
**Figure 2**　(a) 各類事件「至少出現一次」的 Segment 比例，誤差線為 Examination-level Bootstrap 95% CI。(b)(c) 加速率與減速率加總的 ECDF。

**看到了什麼**：延長減速（DP）在 Pathologic 很常見，在 Normal 很少見。加速（AC）則相反，Normal 較常出現。

**數值證據**：DP > 0 的比例是 Pathologic 62.5%（65/104）對 Normal 4.7%（46/972）。AC > 0 是 19.2% 對 67.2%。輕度減速（DL）在 Normal 也有 44.8%，所以「有減速」本身不等於異常。

**限制**：DP 出現率的區間很寬（Pathologic 31%–80%），因為這些片段集中在少數檢查。另外有 37.5% 的 Pathologic 沒有 DP，46 筆 Normal 有 DP，DP 不能單獨當作判斷規則。
""")

md(r"""
## 3. Feature Separability

**目的**：把「兩群分得多開」量化。主要比較 ASTV、DP、MLTV、MSTV、ALTV，使用 KS Statistic、Mutual Information、Wasserstein Distance，並附上 Median、出現率與 Examination-level Bootstrap 區間。
""")

code(r"""
main = ["ASTV", "DP", "MLTV", "MSTV", "ALTV"]
s = sep.loc[main]
table3 = pd.DataFrame({
    "Median Normal": s.normal_median, "Median Pathologic": s.path_median,
    ">0 比例 Normal (%)": 100 * (1 - s.normal_zero_frac), ">0 比例 Pathologic (%)": 100 * (1 - s.path_zero_frac),
    "KS": s.KS, "KS 95% CI (exam bootstrap)": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(s.KS_ci_lo, s.KS_ci_hi)],
    "KS 去掉最大兩個 Exam": s.KS_wo_top2,
    "MI (nats)": s.MI_knn_nats,
    "Wasserstein（原始單位）": s.wasserstein_raw, "Wasserstein ÷ Normal SD": s.wasserstein_normalSD,
    "單位": [p2.UNITS[f] for f in main],
})
fmt = {"Median Normal": "{:.4g}", "Median Pathologic": "{:.4g}", ">0 比例 Normal (%)": "{:.1f}",
       ">0 比例 Pathologic (%)": "{:.1f}", "KS": "{:.3f}", "KS 去掉最大兩個 Exam": "{:.3f}",
       "MI (nats)": "{:.3f}", "Wasserstein（原始單位）": "{:.4g}", "Wasserstein ÷ Normal SD": "{:.2f}"}
display(table3.assign(**{c: table3[c].map(f.format) for c, f in fmt.items()}))
h_label = -sum(q * np.log(q) for q in (df.label.mean(), 1 - df.label.mean()))
print(f"Label entropy H(label) = {h_label:.3f} nats（MI 的上限）")
print("完整 21 個特徵的結果：EDA/outputs/separability.csv")
""")

md(r"""
**怎麼讀這張表**

- **KS Statistic**：兩群 ECDF 的最大垂直差距，介於 0 與 1。KS 越大，代表這「單一特徵」的兩群分布差距越大。
- **Mutual Information（MI）**：特徵與 Label 之間統計依賴性的估計值（kNN 估計，單位 nats）。上限是 Label entropy 0.32 nats；它是估計值，會受樣本數影響。
- **Wasserstein Distance**：把一個分布搬成另一個分布的平均距離，帶有原始單位。ASTV 的 21.3（百分點）與 DP 的 0.0015（次／秒）不能直接比大小，所以不拿來做跨特徵的重要度排名。即使除以 Normal 的標準差，DP 的 5.57 也主要反映「Normal 幾乎全是 0、標準差極小」，仍要小心解讀。

**數據解讀**

- ASTV 最能分開兩群，也最穩定：KS 0.712，區間 [0.558, 0.835]；去掉最大的兩次檢查後仍有 0.663。
- DP 的 KS 是 0.597，但區間很寬 [0.292, 0.776]，去掉最大兩次檢查後降到 0.413。它的強度有一部分來自少數檢查。
- MLTV 的 KS 是 0.544，方向是 Pathologic 較低（中位數 8.1 對 0）。
- MSTV 與 ALTV 的 KS 較低（0.358、0.224），但 MSTV 的 MI（0.126）與 ASTV（0.129）接近。原因見下一小節：MSTV 是雙向偏離，單一門檻分不開，不代表沒有資訊。
""")

md(r"""
### 3.1 KS 不等於 Accuracy

**目的**：用 ASTV 實際算一次，避免把 KS = 0.71 誤讀成「71% 準確率」。
""")

code(r"""
n, p = df[df.label == 0], df[df.label == 1]
xs = np.unique(df.ASTV)
Fn = np.array([(n.ASTV <= x).mean() for x in xs]); Fp = np.array([(p.ASTV <= x).mean() for x in xs])
k = int(np.argmax(np.abs(Fn - Fp)))
print(f"KS 發生在 ASTV = {xs[k]:.0f}：Normal 有 {Fn[k]:.1%} 小於等於它，Pathologic 只有 {Fp[k]:.1%}，差距 = {Fn[k]-Fp[k]:.3f}")

flag = df.ASTV >= 60                                # 假想規則：ASTV >= 60 就示警
acc_rule = (flag == (df.label == 1)).mean()
print(f"規則 ASTV >= 60：示警 {int((flag & (df.label == 0)).sum())} 筆 Normal、{int((flag & (df.label == 1)).sum())} 筆 Pathologic；"
      f"Accuracy = {acc_rule:.1%}；全部猜 Normal 的 Accuracy = {(df.label == 0).mean():.1%}")
""")

md(r"""
**解讀**：KS = 0.71 的意思是「在 ASTV = 59 這個位置，兩群的累積比例差了 71 個百分點」。它不是準確率。用 ASTV ≥ 60 當規則時，Accuracy 是 84.1%，反而低於全部猜 Normal 的 90.3%，因為 Normal 多很多，158 筆誤報就超過了抓到的 91 筆。

**研究意義**：單一特徵再強，也不能直接當偵測器。這也是為什麼本組的主要指標是 PR-AUC，而不是 Accuracy。
""")

md(r"""
### 3.2 MSTV 為什麼 KS 低卻仍值得保留

**目的**：說明 Figure 1 中 Pathologic MSTV 為何在兩側各有集中。先選出 Pathologic，再依「有沒有延長減速（DP > 0）」做描述性分組。
""")

code(r"""
sub = p2.phenotype_table(df)
display(sub[["n_segments", "n_exams", "ASTV", "MSTV", "ALTV", "MLTV", "Width", "Mean", "share_ASTV_ge_60"]].round(3))
print(f"MSTV：單向 AUC = {sep.loc['MSTV', 'AUC_path_gt_normal']:.2f}；"
      f"與 Normal 平均的絕對偏離 |z| 的 AUC = {sep.loc['MSTV', 'AUC_abs_dev']:.2f}")
""")

md(r"""
**解讀**：Normal 的 MSTV 中位數是 1.2。Pathologic 之中，沒有延長減速的 39 筆只有 0.3，有延長減速的 65 筆是 2.7，兩組往相反方向偏離 Normal。所以「MSTV 越高越異常」的單向 AUC 只有 0.59，改看「離 Normal 多遠」則是 0.80。ASTV 在兩組都偏高（ASTV ≥ 60 的比例 95% 與 83%，Normal 為 16%）。

**限制**：這個分組先用了 Pathologic 標籤，再依 DP 切開，是對 Training Data 的描述，**不是臨床分型**。DP-present 只來自 12 次檢查，樣本很集中。
""")

md(r"""
## 4. Feature Redundancy

**目的**：檢查 21 個候選特徵之間有沒有重複資訊。使用 Spearman Correlation（看排序關係，不假設線性），並直接驗證 `Width = Max − Min`。
""")

code(r"""
sp_all, pe_all, sp_norm, pairs, vif, cond = p2.redundancy(df)
FIG_SAME["fig4_spearman_heatmap"] = regenerate("fig4_spearman_heatmap", lambda: p2.fig_heatmap(sp_all))
display(Image(filename=str(FIG / "fig4_spearman_heatmap.png"), width=640))

resid = (df.Width - (df.Max - df.Min)).abs().max()
assert resid == 0, "Width != Max - Min"
print(f"Width = Max - Min：{len(df)} 筆全部成立，最大誤差 = {resid}；21 個特徵的共變異數秩 = {cond['rank_all21']}")
display(pairs.head(8)[["feature_a", "feature_b", "spearman_train_all", "spearman_train_normal"]].round(3).reset_index(drop=True))
print("Variability 群內：ASTV–MSTV ρ = %.2f，ALTV–MLTV ρ = %.2f" % (sp_all.loc["ASTV", "MSTV"], sp_all.loc["ALTV", "MLTV"]))
""")

md(r"""
**Figure 4**　21 個特徵的 Spearman ρ（Training Portion）。橘色為正相關、藍色為負相關，|ρ| ≥ 0.7 的格子標出數值。

**數據解讀**

- `Width = Max − Min` 在 1076 筆全部成立，誤差為 0。這是恆等式，所以 Width 可以由 Min、Max 完全還原，21 個特徵實際只有 20 個自由度。
- Mode、Mean、Median 高度相關（ρ = 0.90–0.97），但**不是完全相同**，移除其中任何一個都會損失一些資訊。
- 四個 Variability 特徵彼此不重複（ASTV–MSTV ρ = −0.36；ALTV–MLTV ρ = −0.02）。

**限制**：Spearman 只描述兩兩之間的排序關係。高相關不代表移除後 Autoencoder 的表現一定不受影響，這要由 P5 的消融實驗確認。
""")

md(r"""
### 4.1 依冗餘程度精簡特徵（PRUNED18）

**目的**：把上面的觀察變成一條事先固定的規則，供 H3 使用。規則只用 **Normal** 的 Training Data，不看異常標籤：先移除 Width，再把其餘 Histogram 特徵依 |ρ| ≥ 0.8 分群，每群保留最有代表性的一個。
""")

code(r"""
p18 = p2.prune_histogram(sp_norm, drop_width_first=True)
kept18 = [f for f in FEATURES if f not in p18["dropped"]]
r2 = p2.linear_r2_from_kept(df[df.label == 0], kept18, p18["dropped"])
p17 = p2.prune_histogram(sp_norm)                 # 舊規則，僅作選做對照
SETS = p2.feature_sets(p18["dropped"], p17["dropped"])
print("分群結果：", p18["clusters"])
print("移除：", p18["dropped"], "→ 保留", len(kept18), "個特徵")
print("被移除特徵可由保留特徵線性解釋的變異比例 R²（Normal-train）：", r2)
display(pd.DataFrame({"n_features": {k: len(v) for k, v in SETS.items()},
                      "移除的特徵": {k: ", ".join(f for f in FEATURES if f not in v) or "—" for k, v in SETS.items()}}))
""")

md(r"""
**解讀**：規則移除 Width、Mode、Mean，留下 18 個特徵（PRUNED18）。Width 的 R² = 1.0，是無損移除。Mode 與 Mean 的 R² 是 0.93 與 0.94，代表大部分資訊仍在，但不是全部。

**限制**：PRUNED18 是啟發式（Heuristic）挑選，不是無資訊損失的精簡。較早的 PRUNED17 還多移除了 Min，而 Width 不在時 Min 無法還原，所以只保留為選做對照，不作為主結果。
""")

md(r"""
## 5. Exploratory Hypotheses

**目的**：把 Section 2–4 的觀察整理成兩個可以被 P5 用 Autoencoder 檢驗的假說。以下都是**待驗證的假說，不是結論**。

### H1 — Variability Features

> ASTV、MSTV、ALTV、MLTV 這一群特徵，可能提供其他特徵補不上的異常偵測資訊。

- **來自哪個觀察**：ASTV 是最能分開兩群的單一特徵（KS 0.71，Figure 1、Section 3）。MLTV 的 KS 為 0.54。MSTV 雙向偏離 Normal（Section 3.2）。四個特徵彼此相關性低（Section 4）。
- **為什麼提出**：沒有延長減速的 Pathologic 片段（39 筆）在 DP 上與 Normal 無法區分，主要靠 Variability 特徵才看得出差異。如果拿掉這一群，預期這類片段最容易被漏掉。
- **P5 怎麼驗證**：比較 `ALL21` 與 `NO_VARIABILITY17`（移除這四個特徵），其他條件完全相同。預期方向：移除後 Validation PR-AUC 下降。若差距在 Seed 變動範圍內，則 H1 不被支持。

### H3 — Redundancy-guided Feature Reduction

> 移除確定重複（Width）與高度相關（Mode、Mean）的特徵後，能否在減少輸入維度的同時維持相近的偵測表現？

- **來自哪個觀察**：`Width = Max − Min` 完全成立；Mode／Mean／Median 的 ρ = 0.90–0.97（Section 4）。
- **為什麼提出**：重複的輸入會讓 Autoencoder 的重建誤差重複計算同一份資訊。但高相關不保證可以安全移除，所以需要實驗。
- **P5 怎麼驗證**：比較 `ALL21`、`DROP_WIDTH20`（只移除 Width，無損對照）、`PRUNED18`。若 PRUNED18 明顯低於 ALL21，代表 Mode／Mean 帶有模型用得到的資訊，H3 不被支持。

**補充（不列為主要假說）**：我們原本猜想把 DL、DS、DP 加總會是更好的特徵。EDA 不支持：加總後 KS 從 0.60 降到 0.40，因為輕度減速在 Normal 就有 45%。這個轉換不再往下做。

**揭露**：H1 是看過 Training Label 之後提出的。H3 的規則只用 Normal 資料。
""")

code(r"""
print(f"DEC_SUM = DL+DS+DP 的 KS = {sep.loc['DEC_SUM', 'KS']:.2f}；DP 單獨的 KS = {sep.loc['DP', 'KS']:.2f}")
for name in ["ALL21", "DROP_WIDTH20", "PRUNED18", "NO_VARIABILITY17"]:
    print(f"{name:17s} ({len(SETS[name]):2d}): {SETS[name]}")
print("機器可讀版本：EDA/outputs/feature_sets.json")
""")

md(r"""
**說明**：以上四組 Feature Set 是 P2 建議的實驗定義，**尚未與 P3／P5 確認**。P3 目前的程式固定使用 21 個特徵，要由 P3 決定如何加入 Feature Set 選項；P2 不修改 P3 的程式。
""")

md(r"""
## 6. Hypothesis Verification Status

**目的**：避免「提出假說後從未回頭檢視」。下表記錄每個假說由誰、用什麼方式驗證，以及目前的真實狀態。

| Hypothesis | Supporting EDA Evidence（Train-only） | P5 Ablation | Evaluation | Status |
|---|---|---|---|---|
| **H1** Variability Features 提供互補資訊 | ASTV KS 0.71 [0.56, 0.83]；MLTV KS 0.54；MSTV 雙向偏離（\|z\| AUC 0.80 對單向 0.59）；群內 ρ 低 | `ALL21` 對 `NO_VARIABILITY17` | Validation PR-AUC（主）、AUROC（輔）；5 seeds；另報 DP-present／DP-absent 的 Recall | **Pending P5 Verification** |
| **H3** Redundancy-guided Reduction 維持相近表現 | Width = Max − Min（誤差 0）；Mode／Mean／Median ρ 0.90–0.97；PRUNED18 的 R² 0.93／0.94 | `ALL21` 對 `DROP_WIDTH20` 對 `PRUNED18`（`PRUNED17` 選做） | 同上 | **Pending P5 Verification** |

**固定的實驗條件**：P1 的 Split、P3 的 Autoencoder 架構與訓練設定、Seeds、Scaler 只用 Normal-train fit、同一套 Threshold Protocol。每次只改 Feature Set。

**目前狀態說明**

- P5 尚未回報任何 Feature Ablation 結果，所以兩個假說都是 **Pending P5 Verification**。
- P3 已公布的是 `ALL21` 的 Validation Baseline（暫定），不是對 H1／H3 的驗證。
- P2 做過一個 Train-only 的 Mahalanobis Probe 來挑選假說。它使用了 Training Label，也不是 P3 的 Autoencoder，**不能當作驗證結果**，本 Notebook 與 §4 都不引用它的數字。
- 本 Notebook 不載入 Test。P5 的正式結果會寫在 §7–§8，之後再回來更新這張表與 §4 的交叉引用。

**給 P5 的提醒**：課程要求至少兩個 Controlled Ablation，其中至少一個要是 Anomaly Detection-specific。上面兩個 Feature Ablation 不能取代它。P3 已實作 Scoring Function 的比較（`--score mse|cosine|mahalanobis`），可以作為這一項。
""")

md(r"""
## 7. Paper Report §4（英文，交 I1）

**目的**：提供可直接放進 ACL 報告的 §4 英文內文。下面的英文是由 `report/p2_section4.tex` 轉出的同一份文字；LaTeX 版含 Figure、Table 與引用指令，BibTeX 在 `report/p2_refs.bib`。

---

### 4 Exploratory Feature Analysis

> 中文說明：第一段交代資料範圍與統計方法，讓讀者知道所有數字只來自 Training Portion，以及為什麼用 Examination-level Bootstrap。

**Analysis protocol.** """ + P["Analysis protocol."] + r"""

> 中文說明：第二段是主要發現。用 Figure 1 與 Table 1 說明哪些特徵分得開、哪些只是看起來強，並解釋 MSTV 的雙向偏離。

**Distribution contrast and separability.** """ + P["Distribution contrast and separability."] + r"""

**Figure 1.** """ + fig_cap + r"""（圖檔：`EDA/figures/fig1_variability_contrast.pdf`，即 Section 2 的 Figure 1；tex 內的路徑以 Repository Root 為基準。）

""" + tab_md + r"""

**Table 1.** """ + tab_cap + r"""

> 中文說明：第三段說明冗餘。重點是區分「完全重複」（Width）與「高度相關但不相同」（Mode、Mean），並誠實說明精簡規則不是無損的。

**Feature redundancy.** """ + P["Feature redundancy."] + r"""

> 中文說明：第四段把觀察轉成假說，並寫明實驗控制條件。兩個假說都寫成待 §7–§8 檢驗，不寫成結論。

**Exploratory hypotheses.** """ + P["Exploratory hypotheses."] + r"""

---

**引用對應**：Dataset 定義引用 UCI（Campos and Bernardes, 2000）。文獻背景引用 Huang et al. (2021)，其研究是三分類且沒有 Exam 分組切分，只作背景，不作為本組數據的證據。KS、MI、Median 等數值全部來自本組的 Figure 1 與 Table 1。
""")

md(r"""
### 7.1 一致性檢查：Notebook、CSV、LaTeX、Figures

**目的**：用程式確認 §4 英文內文中的每個數字，都等於這份 Notebook 剛剛重新算出來的結果；同時比對 `EDA/outputs/` 的 CSV 與 `EDA/figures/` 的圖檔。
""")

code(r"""
tex = (ROOT / "report" / "p2_section4.tex").read_text(encoding="utf-8")
flat = re.sub(r"\s+", "", tex)
f2 = lambda v: f"{v:.2f}"
d2 = lambda v: f"{v:.2f}".replace("0.", ".", 1)            # 表格用 .71 的寫法
pct = lambda v, k=0: f"{100 * v:.{k}f}" + r"\%"
S = sep; sub_p, sub_a = sub.loc["Pathologic, DP-present (DP > 0)"], sub.loc["Pathologic, DP-absent (DP = 0)"]
pr = pres.set_index(["feature", "class"]).share_with_event
mmm = [abs(sp_all.loc[a, b]) for a, b in (("Mode", "Mean"), ("Mode", "Median"), ("Mean", "Median"))]

claims = {
    "樣本數": f"{len(n)}Normaland{len(p)}Pathologicsegmentsfrom{df.exam.nunique()}examinations",
    "Pathologic 的 Exam 數": f"only{p.exam.nunique()}examinations",
    "最大兩個 Exam": f"contribute{int(per_exam.iloc[:2].sum())}ofthem",
    "ASTV median": f"median{S.normal_median.ASTV:g}vs.\\{S.path_median.ASTV:g}",
    "ASTV KS 與 CI": f"KS{f2(S.KS.ASTV)},95\\%CI{f2(S.KS_ci_lo.ASTV)}--{f2(S.KS_ci_hi.ASTV)}",
    "MLTV = 0 比例": f"{pct(S.path_zero_frac.MLTV)}ofPathologicsegmentshave$\\mathrm{{MLTV}}=0$vs.\\{pct(S.normal_zero_frac.MLTV)}",
    "DP 出現率": f"{pct(pr['DP', 'Pathologic'], 1)}ofPathologicbut{pct(pr['DP', 'Normal'], 1)}ofNormal",
    "KS 去掉最大兩個 Exam": f"dropsto{f2(S.KS_wo_top2.DP)}and{f2(S.KS_wo_top2.Mean)}",
    "DP-present 子群": f"{int(sub_p.n_segments)}segmentsfrom{int(sub_p.n_exams)}examinations)haveamedianMSTVof{sub_p.MSTV:g}",
    "DP-absent 子群": f"{int(sub_a.n_segments)}segmentsfrom{int(sub_a.n_exams)}examinations)of{sub_a.MSTV:g},against{sub.loc['Normal'].MSTV:g}forNormal",
    "MSTV AUC": f"only{f2(S.AUC_path_gt_normal.MSTV)}whiletheAUCofitsabsolutedeviationfromtheNormalmeanis{f2(S.AUC_abs_dev.MSTV)}",
    "ASTV >= 60": f"{pct(sub_p.share_ASTV_ge_60)}and{pct(sub_a.share_ASTV_ge_60)}ofthetwogroupsvs.\\{pct(sub.loc['Normal'].share_ASTV_ge_60)}",
    "Wasserstein (SD)": f"ASTV:{S.wasserstein_normalSD.ASTV:.1f}SD;Mean:{S.wasserstein_normalSD.Mean:.1f}SD",
    "Pathologic ALTV = 0": f"{pct(S.path_zero_frac.ALTV)}ofPathologicALTVvaluesarezero",
    "Label entropy": f"labelentropy{f2(h_label)}",
    "共變異數秩": f"hasrank{cond['rank_all21']}",
    "Mode/Mean/Median ρ": f"$\\rho={f2(min(mmm))}$--${f2(max(mmm))}$",
    "ASTV–MSTV、ALTV–MLTV ρ": f"ASTV--MSTV$\\rho={f2(sp_all.loc['ASTV', 'MSTV'])}$;ALTV--MLTV$\\rho={f2(sp_all.loc['ALTV', 'MLTV'])}$",
    "PRUNED18 移除與剩餘": f"Itremoves{','.join(p18['dropped'][:-1])}and{p18['dropped'][-1]},leaving{len(kept18)}features",
    "Mode、Mean 的 R²": f"{pct(r2['Mode'])}and{pct(r2['Mean'])}ofthevarianceofModeandMean",
    "DEC_SUM 對 DP": f"KS{f2(S.KS.DEC_SUM)}vs.\\{f2(S.KS.DP)}",
    "DL 在 Normal 的出現率": f"Normalsegments({pct(pr['DL', 'Normal'])})",
}
scale = {"DP": 1000, "AC": 1000}                            # 表格中事件率以每 1000 秒表示
for f in ["ASTV", "DP", "Mean", "MLTV", "AC", "MSTV", "ALTV", "LB"]:
    k = scale.get(f, 1)
    med = lambda v: f"{round(k * v, 1):g}"
    claims[f"Table 1 · {f}"] = (f"&{med(S.normal_median[f])}&{med(S.path_median[f])}&{d2(S.KS[f])}[{d2(S.KS_ci_lo[f])},{d2(S.KS_ci_hi[f])}]"
                               f"&{d2(S.AUC_path_gt_normal[f])}&{d2(S.AUC_abs_dev[f])}&{d2(S.MI_knn_nats[f])}\\\\")
check = pd.DataFrame({"由本次執行算出的字串": claims, "出現在 p2_section4.tex": {k: v in flat for k, v in claims.items()}})
display(check)
assert check["出現在 p2_section4.tex"].all(), "LaTeX 與重新計算的結果不一致"

# CSV：重新計算的結果 vs. Repository 中的 EDA/outputs/separability.csv、event_presence.csv
csv = pd.read_csv(OUT / "separability.csv", index_col=0)
num = [c for c in sep.columns if c in csv.columns and pd.api.types.is_numeric_dtype(sep[c])]
csv_ok = np.allclose(sep.loc[csv.index, num].astype(float), csv[num].astype(float), rtol=2e-4, atol=1e-9, equal_nan=True)
pres_csv = pd.read_csv(OUT / "event_presence.csv")
pres_ok = np.allclose(pres.select_dtypes("number"), pres_csv.select_dtypes("number"), atol=1e-4)
sets_ok = json.loads((OUT / "feature_sets.json").read_text()) == SETS
print(f"separability.csv 一致（{len(num)} 欄 × {len(csv)} 列）：{csv_ok}；event_presence.csv 一致：{pres_ok}；feature_sets.json 一致：{sets_ok}")
print("重新產生的 PNG 與既有檔案逐位元組相同：", FIG_SAME)
assert csv_ok and pres_ok and sets_ok and all(FIG_SAME.values())

# Train-only：這份 Notebook 從 data/ 讀過的檔案只能有 train.csv
data_reads = sorted({r for r in READ_LOG if r.startswith("data/")})
print("從 data/ 讀取的檔案：", data_reads)
assert data_reads == ["data/splits/train.csv"]
print("全部檢查通過。")
""")

md(r"""
**解讀**：上表每一列都是由這次執行重新算出的數字組成的字串，並確認它出現在 `report/p2_section4.tex`。CSV、Feature Set 定義與圖檔也與 Repository 內的檔案一致。讀檔紀錄顯示，本 Notebook 從 `data/` 只讀了 `train.csv`。

**限制**：這個檢查只保證「數字一致」，不保證 ACL 版面。正式樣式檔下的排版要由 I1 整合時確認。
""")

md(r"""
## 8. References & Final Checklist

### References

1. Campos, D. and Bernardes, J. (2000). *Cardiotocography*. UCI Machine Learning Repository. DOI 10.24432/C51S4N. https://archive.ics.uci.edu/dataset/193/cardiotocography — Dataset 與特徵定義。
2. Huang, L., Jiang, Z., Cai, R., Li, L., Chen, Q., Hong, J., Hao, Z. and Wei, H. (2021). Investigating the interpretability of fetal status assessment using antepartum cardiotocographic records. *BMC Medical Informatics and Decision Making*, 21, 355. https://doi.org/10.1186/s12911-021-01714-4 — 文獻背景（三分類，非本組切分）。
3. SciPy documentation, `scipy.stats.ks_2samp`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html — Two-sample KS Statistic 的定義與實作。

本 Notebook 的所有統計數值都是本組 Training Portion 的計算結果，不是上述文獻的結果。

### 相關檔案

| 檔案 | 用途 |
|---|---|
| `EDA/p2_eda.py` | 所有統計與圖的程式（`python EDA/p2_eda.py`） |
| `EDA/outputs/` | 統計表（`separability.csv`、`event_presence.csv`、`feature_sets.json` 等） |
| `EDA/figures/` | 圖（PNG 與 PDF） |
| `report/p2_section4.tex`、`report/p2_refs.bib` | §4 LaTeX 與 BibTeX，交 I1 |
| `EDA/docs/p2_handoff.md` | 給 P3／P5 的 Feature Set 與消融規格 |
| `EDA/P2_Feature_Analysis.ipynb` | 較早的完整研究 Notebook（保留） |

### Final Checklist

只有實際通過的項目才打勾。

- [x] 至少一張 Normal／Pathologic 分布對照圖。（Figure 1、Figure 2）
- [x] 每張圖都有文字解讀。（Figure 1、2、4 與 Appendix 的圖）
- [x] 量化 Feature Separability。（Section 3：KS、MI、Wasserstein、Bootstrap CI）
- [x] 檢查 Feature Redundancy。（Section 4：Spearman、Width = Max − Min）
- [x] 提出 H1、H3 及可執行的 Ablation。（Section 5–6；Feature Set 定義尚待 P3／P5 確認）
- [ ] P5 已完成相應驗證，或明確標註 Pending 並持續追蹤至正式交付。（**未完成**：目前標註為 Pending P5 Verification；P5 尚未回報結果，正式交付前需回來更新 Section 6 與 §4。）
- [x] §4 英文內文完成。（Section 7 與 `report/p2_section4.tex`；ACL 樣式下的版面待 I1 確認）
- [x] EDA 只使用 Training Portion。（Section 7.1 的讀檔紀錄）
- [x] Notebook 可以由 Clean Kernel → Run All 正常執行。（以 `jupyter nbconvert --execute` 驗證）
- [x] 結果和 LaTeX／CSV／Figures 一致。（Section 7.1 的自動檢查）
""")

md(r"""
## Appendix — 補充圖

**目的**：收錄主要敘事沒有用到、但可供報告或簡報選用的圖。每張圖附一段簡短解讀。
""")

code(r"""
FIG_SAME["fig3_separability_ranking"] = regenerate("fig3_separability_ranking",
    lambda: p2.fig_ranking(sep))
display(Image(filename=str(FIG / "fig3_separability_ranking.png"), width=460))
""")

md(r"""
**Figure A1（`fig3_separability_ranking`）**　21 個特徵與 3 個加總欄位的 KS，橫線是 Examination-level Bootstrap 95% CI，叉號是去掉最大兩次 Pathologic 檢查後的 KS。ASTV 排名最高且區間最窄。DP 與 Mean 的點與叉號距離很遠（0.60→0.41、0.58→0.37），代表它們的強度依賴少數檢查。這張圖只是單一特徵的排序，不是特徵重要度的定論。
""")

code(r"""
FIG_SAME["fig5_pathologic_phenotypes"] = regenerate("fig5_pathologic_phenotypes", lambda: p2.fig_phenotypes(df))
display(Image(filename=str(FIG / "fig5_pathologic_phenotypes.png"), width=900))
""")

md(r"""
**Figure A2（`fig5_pathologic_phenotypes`）**　Pathologic 依 DP 分成的兩個描述性子群。DP-absent 的點集中在低 MSTV、高 ASTV；DP-present 的點 MSTV 較高、Histogram Mean 較低。兩組在 MSTV 上落在 Normal 的兩側，對應 Section 3.2 的中位數 0.3 與 2.7。分組用到了 Pathologic 標籤，不是臨床分型。
""")

code(r"""
FIG_SAME["figS1_all_features_ecdf"] = regenerate("figS1_all_features_ecdf", lambda: p2.fig_all_ecdf(df, sep))
display(Image(filename=str(FIG / "figS1_all_features_ecdf.png"), width=980))
assert all(FIG_SAME.values()), FIG_SAME
""")

md(r"""
**Figure A3（`figS1_all_features_ecdf`）**　全部 21 個特徵的 ECDF，各小圖標題為 KS。它是 Figure 1 的完整版，用來確認主文沒有只挑好看的特徵：Nzeros、DS、Max、Nmax 等特徵的兩條曲線幾乎重疊（KS < 0.2）。同樣只描述 Training Portion 的單一特徵分布。
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = ROOT / "EDA" / "P2_Section4_Deliverable.ipynb"
nbf.write(nb, str(out))
print("wrote", out, len(cells), "cells")
