"""P2 - Exploratory Feature Analysis for the CTG anomaly-detection project (HW2 section 4).

Run from the repository root:

    python EDA/p2_eda.py

Everything here reads ONLY the training portion through P1's loader
(`load_split("train", ...)`). Validation and test files are never opened.

Outputs
    EDA/outputs/*.csv, *.json   statistics tables
    EDA/figures/*.png, *.pdf    figures

Two ways the training data is used (do not mix them up):
  * descriptive EDA  -> all 1076 training rows, Normal and Pathologic labels visible;
  * anything that imitates a one-class model (scaling, covariance, redundancy
    pruning) -> fitted on training Normal rows only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import ks_2samp, rankdata, wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "data"))
from load_data import FEATURES, load_split  # noqa: E402  (P1's shared loader, unmodified)

OUT = ROOT / "EDA" / "outputs"
FIG = ROOT / "EDA" / "figures"
SEED = 42
N_BOOT = 2000

FAMILY = {
    "baseline": ["LB"],
    "events": ["AC", "FM", "UC", "DL", "DS", "DP"],
    "variability": ["ASTV", "MSTV", "ALTV", "MLTV"],
    "histogram": ["Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean", "Median",
                  "Variance", "Tendency"],
}
FAMILY_OF = {f: fam for fam, fs in FAMILY.items() for f in fs}
# Event counts divided by segment length: zero-inflated, few distinct count levels.
EVENT_RATES = ["AC", "FM", "UC", "DL", "DS", "DP"]
# Small-cardinality ordinal variables (treated as discrete for mutual information).
DISCRETE = ["Nmax", "Nzeros", "Tendency"]
UNITS = {
    "LB": "bpm", "AC": "1/s", "FM": "1/s", "UC": "1/s", "DL": "1/s", "DS": "1/s", "DP": "1/s",
    "ASTV": "% of time", "MSTV": "unit not stated by UCI", "ALTV": "% of time",
    "MLTV": "unit not stated by UCI", "Width": "bpm", "Min": "bpm", "Max": "bpm",
    "Nmax": "count", "Nzeros": "count", "Mode": "bpm", "Mean": "bpm", "Median": "bpm",
    "Variance": "unit not stated by UCI", "Tendency": "-1/0/1",
}

C_NORMAL, C_PATH = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("bo", ["#104281", "#2a78d6", "#f0efec", "#eb6834", "#9c3a12"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 8,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- data
def load_train() -> pd.DataFrame:
    """Training portion only: 21 features + label + exam (row_id as index)."""
    X, y, exam = load_split("train", return_exam=True)
    df = X.copy()
    df["label"] = y
    df["exam"] = exam
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate derived features for H2 (all are simple, label-free formulas)."""
    out = df.copy()
    out["DEC_SUM"] = out["DL"] + out["DS"] + out["DP"]      # all decelerations per second
    out["DEC_SEVERE"] = out["DS"] + out["DP"]              # severe + prolonged decelerations per second
    out["ACC_DEC"] = out["AC"] - out["DEC_SUM"]            # accelerations minus decelerations per second
    return out


DERIVED = ["DEC_SUM", "DEC_SEVERE", "ACC_DEC"]


# --------------------------------------------------------------------------- audit
def audit(df: pd.DataFrame) -> dict:
    X = df[FEATURES]
    top_share = X.apply(lambda s: s.value_counts(normalize=True).iloc[0])
    path = df[df.label == 1]
    per_exam = path.exam.value_counts()
    res = {
        "n_rows": int(len(df)),
        "n_normal": int((df.label == 0).sum()),
        "n_pathologic": int((df.label == 1).sum()),
        "anomaly_rate": float(df.label.mean()),
        "n_exams": int(df.exam.nunique()),
        "n_exams_with_pathologic": int(path.exam.nunique()),
        "pathologic_share_top2_exams": float(per_exam.iloc[:2].sum() / len(path)),
        "pathologic_per_exam_top5": {k: int(v) for k, v in per_exam.head(5).items()},
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "constant_features": [c for c in FEATURES if X[c].nunique() == 1],
        "near_constant_features(top value share > 0.95)":
            {c: round(float(v), 4) for c, v in top_share.items() if v > 0.95},
        "zero_fraction": {c: round(float((X[c] == 0).mean()), 4) for c in FEATURES},
        "n_unique": {c: int(X[c].nunique()) for c in FEATURES},
        "width_equals_max_minus_min": bool((df.Width == df.Max - df.Min).all()),
        "width_max_abs_residual": float((df.Width - (df.Max - df.Min)).abs().max()),
    }
    return res


# --------------------------------------------------------------------------- separability
def auc_path_gt_normal(x: np.ndarray, y: np.ndarray) -> float:
    """P(x_path > x_normal) + 0.5 P(tie): rank-based, scale-free, valid with ties."""
    r = rankdata(x)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def binary_mi(a: np.ndarray, b: np.ndarray) -> float:
    return float(mutual_info_score(a, b))  # nats, exact plug-in for two discrete variables


def separability(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = df.label.to_numpy()
    n, p = df[df.label == 0], df[df.label == 1]
    mu0, sd0 = n[cols].mean(), n[cols].std().replace(0, 1)  # one-class style scaling

    # kNN mutual information (Kraskov/Ross estimator), averaged over seeds because the
    # estimator adds random jitter; discrete mask for the ordinal histogram features.
    mask = [c in DISCRETE for c in cols]
    mi = np.mean([mutual_info_classif(df[cols], y, discrete_features=mask, n_neighbors=5,
                                      random_state=s) for s in range(20)], axis=0)
    h_y = -sum(q * np.log(q) for q in (y.mean(), 1 - y.mean()))

    rows = []
    for j, c in enumerate(cols):
        a, b = n[c].to_numpy(), p[c].to_numpy()
        ks = ks_2samp(a, b)
        auc = auc_path_gt_normal(df[c].to_numpy(), y)
        q = lambda v, k: float(np.percentile(v, k))  # noqa: E731
        rows.append({
            "feature": c,
            "family": FAMILY_OF.get(c, "derived"),
            "type": ("event rate (zero-inflated)" if c in EVENT_RATES else
                     "ordinal discrete" if c in DISCRETE else
                     "derived" if c in DERIVED else "continuous / integer-valued"),
            "normal_mean": a.mean(), "path_mean": b.mean(),
            "normal_median": np.median(a), "path_median": np.median(b),
            "normal_IQR": q(a, 75) - q(a, 25), "path_IQR": q(b, 75) - q(b, 25),
            "normal_zero_frac": (a == 0).mean(), "path_zero_frac": (b == 0).mean(),
            "KS": ks.statistic,
            "AUC_path_gt_normal": auc,
            "AUC_sep": max(auc, 1 - auc),
            "cliffs_delta": 2 * auc - 1,
            # two-sided view, closer to what a one-class detector uses: AUROC of the
            # absolute z-score w.r.t. the Normal training mean / SD
            "AUC_abs_dev": float(roc_auc_score(y, np.abs((df[c] - mu0[c]) / sd0[c]))),
            "direction": "higher in Pathologic" if auc > 0.5 else "lower in Pathologic",
            "wasserstein_raw": wasserstein_distance(a, b),
            "wasserstein_normalSD": wasserstein_distance((a - mu0[c]) / sd0[c], (b - mu0[c]) / sd0[c]),
            "MI_knn_nats": mi[j],
            "MI_knn_frac_of_H(label)": mi[j] / h_y,
            "MI_presence_nats": binary_mi(df[c].to_numpy() > 0, y) if c in EVENT_RATES else np.nan,
        })
    return pd.DataFrame(rows).set_index("feature")


def exam_bootstrap(df: pd.DataFrame, cols: list[str], n_boot: int = N_BOOT, seed: int = SEED):
    """Cluster bootstrap: resample whole exams, because segments of one exam are dependent."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("exam")]
    ks = np.full((n_boot, len(cols)), np.nan)
    auc = np.full((n_boot, len(cols)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i] for i in idx])
        y = s.label.to_numpy()
        if y.sum() < 2:
            continue
        for j, c in enumerate(cols):
            x = s[c].to_numpy()
            ks[b, j] = ks_2samp(x[y == 0], x[y == 1]).statistic
            auc[b, j] = auc_path_gt_normal(x, y)
    ci = lambda m: np.nanpercentile(m, [2.5, 97.5], axis=0)  # noqa: E731
    (kl, kh), (al, ah) = ci(ks), ci(auc)
    return pd.DataFrame({"KS_ci_lo": kl, "KS_ci_hi": kh, "AUC_ci_lo": al, "AUC_ci_hi": ah}, index=cols)


def row_bootstrap_ks(df: pd.DataFrame, cols: list[str], n_boot: int = 500, seed: int = SEED):
    """Naive row-level bootstrap, only to show how much it understates uncertainty."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(cols)))
    Xv, y = df[cols].to_numpy(), df.label.to_numpy()
    for b in range(n_boot):
        i = rng.integers(0, len(df), len(df))
        for j in range(len(cols)):
            out[b, j] = ks_2samp(Xv[i, j][y[i] == 0], Xv[i, j][y[i] == 1]).statistic
    lo, hi = np.percentile(out, [2.5, 97.5], axis=0)
    return pd.DataFrame({"KS_rowboot_lo": lo, "KS_rowboot_hi": hi}, index=cols)


def leave_top_exams_out(df: pd.DataFrame, cols: list[str], k: int = 2) -> pd.Series:
    """KS after dropping the k exams that contribute most Pathologic segments."""
    top = df[df.label == 1].exam.value_counts().index[:k]
    s = df[~df.exam.isin(top)]
    return pd.Series({c: ks_2samp(s.loc[s.label == 0, c], s.loc[s.label == 1, c]).statistic
                      for c in cols}, name=f"KS_without_top{k}_path_exams")


def presence_table(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Share of segments with at least one event, per class, with exam-bootstrap CI."""
    rng = np.random.default_rng(seed)
    cols = ["AC", "DL", "DS", "DP", "FM", "UC"]
    groups = [g for _, g in df.groupby("exam")]
    boot = np.empty((n_boot, len(cols), 2))
    for b in range(n_boot):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for k in (0, 1):
            boot[b, :, k] = (s.loc[s.label == k, cols] > 0).mean().to_numpy()
    rows = []
    for j, c in enumerate(cols):
        for k, name in ((0, "Normal"), (1, "Pathologic")):
            lo, hi = np.nanpercentile(boot[:, j, k], [2.5, 97.5])
            rows.append({"feature": c, "class": name,
                         "n": int((df.label == k).sum()),
                         "n_with_event": int(((df.label == k) & (df[c] > 0)).sum()),
                         "share_with_event": float((df.loc[df.label == k, c] > 0).mean()),
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def phenotype_table(df: pd.DataFrame) -> pd.DataFrame:
    """Medians of Normal vs. two label-free sub-groups of Pathologic (DP > 0 or DP = 0)."""
    cols = ["LB", "AC", "DL", "DP", "ASTV", "MSTV", "ALTV", "MLTV", "Width", "Min", "Mean", "Median", "Variance"]
    grp = np.where(df.label == 0, "Normal",
                   np.where(df.DP > 0, "Pathologic, DP > 0 (decelerative)", "Pathologic, DP = 0 (flat)"))
    t = df.groupby(grp)[cols].median()
    t.insert(0, "n_exams", df.groupby(grp).exam.nunique())
    t.insert(0, "n_segments", df.groupby(grp).size())
    t["share_ASTV_ge_60"] = (df.ASTV >= 60).groupby(grp).mean()
    return t


# --------------------------------------------------------------------------- redundancy
def redundancy(df: pd.DataFrame):
    normal = df[df.label == 0]
    sp_all = df[FEATURES].corr(method="spearman")
    pe_all = df[FEATURES].corr(method="pearson")
    sp_norm = normal[FEATURES].corr(method="spearman")
    rows = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            rows.append({"feature_a": a, "feature_b": b,
                         "spearman_train_all": sp_all.loc[a, b], "pearson_train_all": pe_all.loc[a, b],
                         "spearman_train_normal": sp_norm.loc[a, b],
                         "same_family": FAMILY_OF[a] == FAMILY_OF[b]})
    pairs = pd.DataFrame(rows)
    pairs["abs_spearman"] = pairs.spearman_train_all.abs()
    pairs = pairs.sort_values("abs_spearman", ascending=False).drop(columns="abs_spearman")

    # Variance inflation factors on Normal training rows (what a one-class model sees).
    Z = (normal[FEATURES] - normal[FEATURES].mean()) / normal[FEATURES].std().replace(0, 1)
    vif = {}
    for c in FEATURES:
        others = [f for f in FEATURES if f != c]
        beta, *_ = np.linalg.lstsq(Z[others].to_numpy(), Z[c].to_numpy(), rcond=None)
        r2 = 1 - ((Z[c] - Z[others].to_numpy() @ beta) ** 2).sum() / (Z[c] ** 2).sum()
        vif[c] = float(1 / max(1 - r2, 1e-12))
    eig = np.linalg.eigvalsh(Z.cov().to_numpy())
    cond = {"cov_min_eigenvalue_all21": float(eig.min()), "cov_max_eigenvalue_all21": float(eig.max()),
            "rank_all21": int(np.linalg.matrix_rank(Z.to_numpy()))}
    return sp_all, pe_all, sp_norm, pairs, pd.Series(vif, name="VIF_normal_train"), cond


def prune_histogram(sp_norm: pd.DataFrame, thr: float = 0.8) -> dict:
    """Label-free pruning rule for H-redundancy, fitted on Normal training rows only.

    Histogram features are clustered (complete linkage on 1 - |Spearman|, cut at
    1 - thr, so every pair inside a cluster has |rho| >= thr). Each cluster keeps its
    medoid (largest mean |rho| to the other members; ties -> first in FEATURES order).
    Width is always dropped because Width = Max - Min exactly.
    """
    from scipy.cluster.hierarchy import fcluster
    hist = FAMILY["histogram"]
    d = 1 - sp_norm.loc[hist, hist].abs().to_numpy()
    np.fill_diagonal(d, 0)
    lab = fcluster(linkage(squareform(d, checks=False), "complete"), 1 - thr, "distance")
    clusters, keep = [], []
    for k in np.unique(lab):
        members = [h for h, l in zip(hist, lab) if l == k]
        clusters.append(members)
        if len(members) == 1:
            keep.append(members[0])
        else:
            sub = sp_norm.loc[members, members].abs()
            keep.append(((sub.sum() - 1) / (len(members) - 1)).idxmax())
    keep = [h for h in hist if h in keep and h != "Width"]
    return {"threshold_abs_spearman": thr, "clusters": clusters, "kept": keep,
            "dropped": [h for h in hist if h not in keep]}


# --------------------------------------------------------------------------- group probe
def group_probe(df: pd.DataFrame, sets: dict[str, list[str]], n_repeats: int = 5) -> pd.DataFrame:
    """Exploratory, model-light probe of feature-GROUP separability (not the P3 model).

    Mahalanobis distance to the Normal training distribution (z-score + Ledoit-Wolf
    covariance, fitted on Normal rows of the fitting folds only), scored on held-out
    exams with exam-grouped 5-fold CV inside the training portion.
    """
    y, g = df.label.to_numpy(), df.exam.to_numpy()
    rows = []
    for name, cols in sets.items():
        X = df[cols].to_numpy(float)
        aucs, aps = [], []
        for r in range(n_repeats):
            score = np.empty(len(df))
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
            for fit, held in cv.split(X, y, g):
                Xn = X[fit][y[fit] == 0]
                mu, sd = Xn.mean(0), Xn.std(0, ddof=1)
                sd[sd == 0] = 1
                lw = LedoitWolf().fit((Xn - mu) / sd)
                score[held] = lw.mahalanobis((X[held] - mu) / sd)
            aucs.append(roc_auc_score(y, score))
            aps.append(average_precision_score(y, score))
        rows.append({"feature_set": name, "n_features": len(cols),
                     "AUROC_mean": np.mean(aucs), "AUROC_sd": np.std(aucs, ddof=1),
                     "PR_AUC_mean": np.mean(aps), "PR_AUC_sd": np.std(aps, ddof=1),
                     "features": " ".join(cols)})
    out = pd.DataFrame(rows).set_index("feature_set")
    out.attrs["base_rate"] = float(y.mean())
    return out


# --------------------------------------------------------------------------- figures
def _ecdf(ax, x, color, ls, label=None):
    xs = np.sort(x)
    ax.step(np.concatenate([[xs[0]], xs]), np.concatenate([[0], np.arange(1, len(xs) + 1) / len(xs)]),
            where="post", color=color, lw=1.6, ls=ls, label=label)


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def fig_variability(df, sep):
    n, p = df[df.label == 0], df[df.label == 1]
    feats = ["ASTV", "ALTV", "MSTV", "MLTV"]
    xlabel = {"ASTV": "ASTV (% of time)", "ALTV": "ALTV (% of time)",
              "MSTV": "MSTV (unit not stated)", "MLTV": "MLTV (unit not stated)"}
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.7), sharey="row")
    for j, f in enumerate(feats):
        hi = np.percentile(df[f], 99.5)
        bins = np.linspace(df[f].min(), hi, 26)
        ax = axes[0, j]
        ax.hist(np.clip(n[f], None, hi), bins=bins, weights=np.full(len(n), 1 / len(n)), color=C_NORMAL,
                alpha=0.55, edgecolor=SURFACE, lw=0.4)
        ax.hist(np.clip(p[f], None, hi), bins=bins, weights=np.full(len(p), 1 / len(p)), histtype="step",
                color=C_PATH, lw=1.6)
        ax.set_title(f"{f}   KS = {sep.loc[f, 'KS']:.2f}", loc="left", color=INK)
        ax = axes[1, j]
        _ecdf(ax, n[f].to_numpy(), C_NORMAL, "-")
        _ecdf(ax, p[f].to_numpy(), C_PATH, "--")
        ax.set_xlim(axes[0, j].get_xlim())
        ax.set_xlabel(xlabel[f], fontsize=7)
    axes[0, 0].set_ylabel("share of class\n(per bin)")
    axes[1, 0].set_ylabel("cumulative share\nof class (ECDF)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_NORMAL, alpha=0.55),
               plt.Line2D([], [], color=C_PATH, lw=1.6, ls="--")]
    fig.legend(handles, [f"Normal (n = {len(n)})", f"Pathologic (n = {len(p)})"], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    _save(fig, "fig1_variability_contrast")


def fig_events(df, pres, sep):
    n, p = df[df.label == 0], df[df.label == 1]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    feats = ["AC", "DL", "DS", "DP"]
    for k, (cls, col, off) in enumerate((("Normal", C_NORMAL, -0.19), ("Pathologic", C_PATH, 0.19))):
        t = pres[pres["class"] == cls].set_index("feature").loc[feats]
        xs = np.arange(len(feats)) + off
        ax.bar(xs, 100 * t.share_with_event, width=0.34, color=col, label=cls,
               hatch=None if k == 0 else "///", edgecolor=SURFACE, lw=0.6)
        ax.errorbar(xs, 100 * t.share_with_event, yerr=[100 * (t.share_with_event - t.ci_lo),
                                                         100 * (t.ci_hi - t.share_with_event)],
                    fmt="none", ecolor=INK2, lw=0.9, capsize=2)
        for x, v, h in zip(xs, t.share_with_event, t.ci_hi):
            ax.text(x, 100 * h + 2, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(feats)), ["AC\naccel.", "DL\nlight", "DS\nsevere", "DP\nprolonged"],
                  fontsize=7)
    ax.set_xlabel("accelerations / decelerations by type", fontsize=7)
    ax.set_ylabel("% of segments with ≥ 1 event")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Event presence", loc="left")
    for ax, f, title, lab in ((axes[1], "AC", "(b) Acceleration rate", "AC (events per 1000 s)"),
                              (axes[2], "DEC_SUM", "(c) Deceleration rate", "DL + DS + DP (events per 1000 s)")):
        _ecdf(ax, 1000 * n[f].to_numpy(), C_NORMAL, "-")
        _ecdf(ax, 1000 * p[f].to_numpy(), C_PATH, "--")
        ax.set_xlabel(lab, fontsize=7)
        ax.set_title(f"{title}   KS = {sep.loc[f, 'KS']:.2f}", loc="left")
        ax.set_ylim(0, 1.02)
    axes[1].set_ylabel("cumulative share of class")
    handles = [plt.Line2D([], [], color=C_NORMAL, lw=1.6), plt.Line2D([], [], color=C_PATH, lw=1.6, ls="--")]
    fig.legend(handles, [f"Normal (n = {len(n)})", f"Pathologic (n = {len(p)})"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, "fig2_events_contrast")


def fig_ranking(sep):
    s = sep.loc[FEATURES].sort_values("KS")
    d = sep.loc[DERIVED].sort_values("KS")
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [len(s), len(d)], "hspace": 0.08})
    for ax, t in ((axes[0], s), (axes[1], d)):
        ypos = np.arange(len(t))
        ax.hlines(ypos, t.KS_ci_lo, t.KS_ci_hi, color=C_NORMAL, lw=1.4, alpha=0.55)
        ax.plot(t.KS, ypos, "o", color=C_NORMAL, ms=4.5, mec=SURFACE, mew=0.6)
        ax.plot(t.KS_wo_top2, ypos, "x", color=INK2, ms=4, mew=0.9)
        ax.set_yticks(ypos, [f"{f}  ({t.loc[f, 'family']})" if ax is axes[0] else f for f in t.index], fontsize=7)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1)
    axes[1].set_xlabel("two-sample KS statistic, Normal vs. Pathologic (train)")
    axes[1].set_ylabel("derived", fontsize=7)
    handles = [plt.Line2D([], [], color=C_NORMAL, marker="o", ms=4.5, lw=1.4),
               plt.Line2D([], [], color=INK2, marker="x", ms=4, lw=0)]
    axes[0].legend(handles, ["KS with 95% exam-bootstrap CI", "KS without the 2 largest\nPathologic exams"],
                   loc="lower right", fontsize=6.5)
    _save(fig, "fig3_separability_ranking")


def fig_heatmap(sp_all):
    d = 1 - sp_all.abs().to_numpy()
    np.fill_diagonal(d, 0)
    order = [FEATURES[i] for i in leaves_list(linkage(squareform(d, checks=False), "average"))]
    m = sp_all.loc[order, order]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(m, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.grid(False)
    for i in range(len(order)):
        for j in range(len(order)):
            v = m.iat[i, j]
            if i != j and abs(v) >= 0.7:
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=5.5,
                        color="white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Spearman ρ (training portion)")
    cb.outline.set_visible(False)
    _save(fig, "fig4_spearman_heatmap")


def fig_all_ecdf(df, sep):
    n, p = df[df.label == 0], df[df.label == 1]
    order = sep.loc[FEATURES].sort_values("KS", ascending=False).index
    fig, axes = plt.subplots(3, 7, figsize=(11, 5), sharey=True)
    for ax, f in zip(axes.ravel(), order):
        _ecdf(ax, n[f].to_numpy(), C_NORMAL, "-")
        _ecdf(ax, p[f].to_numpy(), C_PATH, "--")
        ax.set_title(f"{f}  KS={sep.loc[f, 'KS']:.2f}", loc="left", fontsize=8)
        ax.set_xlabel(UNITS[f], fontsize=6.5)
        ax.tick_params(labelsize=6.5)
        if f in EVENT_RATES:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_fontsize(6)
    for ax in axes[:, 0]:
        ax.set_ylabel("ECDF")
    handles = [plt.Line2D([], [], color=C_NORMAL, lw=1.6), plt.Line2D([], [], color=C_PATH, lw=1.6, ls="--")]
    fig.legend(handles, [f"Normal (n = {len(n)})", f"Pathologic (n = {len(p)})"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    _save(fig, "figS1_all_features_ecdf")


def fig_phenotypes(df):
    n, p = df[df.label == 0], df[df.label == 1]
    pd_, pf = p[p.DP > 0], p[p.DP == 0]
    rng = np.random.default_rng(SEED)
    jit = lambda v, s: v + rng.uniform(-s, s, len(v))  # noqa: E731  (display jitter only)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, (fx, fy, lx, ly, sx, sy) in zip(axes, (
            ("MSTV", "ASTV", "MSTV (mean short-term variability)", "ASTV (% of time with abnormal STV)", 0.04, 0.4),
            ("Variance", "Mean", "Variance (FHR histogram variance)", "Mean (FHR histogram mean, bpm)", 0.4, 0.4))):
        ax.scatter(jit(n[fx], sx), jit(n[fy], sy), s=7, color=C_NORMAL, alpha=0.3, lw=0, label=f"Normal (n = {len(n)})")
        ax.scatter(jit(pd_[fx], sx), jit(pd_[fy], sy), s=26, marker="^", color=C_PATH, edgecolor=SURFACE, lw=0.5,
                   label=f"Pathologic, DP > 0 (n = {len(pd_)})")
        ax.scatter(jit(pf[fx], sx), jit(pf[fy], sy), s=24, marker="s", facecolor="none", edgecolor=C_PATH, lw=1.1,
                   label=f"Pathologic, DP = 0 (n = {len(pf)})")
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
    axes[0].set_xlim(0, 5)
    axes[1].set_xlim(-3, 200)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), markerscale=1.4)
    fig.tight_layout()
    _save(fig, "fig5_pathologic_phenotypes")


# --------------------------------------------------------------------------- main
def run(verbose: bool = True) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = add_derived(load_train())
    cols = FEATURES + DERIVED

    aud = audit(df)
    (OUT / "audit_train.json").write_text(json.dumps(aud, indent=2))

    sep = separability(df, cols)
    sep = sep.join(exam_bootstrap(df, cols)).join(row_bootstrap_ks(df, cols))
    sep["KS_wo_top2"] = leave_top_exams_out(df, cols, 2)
    sep["unit"] = [UNITS.get(c, "1/s") for c in sep.index]
    sep.sort_values("KS", ascending=False).to_csv(OUT / "separability.csv", float_format="%.5g")

    pres = presence_table(df)
    pres.to_csv(OUT / "event_presence.csv", index=False, float_format="%.4f")

    pheno = phenotype_table(df)
    pheno.to_csv(OUT / "pathologic_phenotypes.csv", float_format="%.4g")

    sp_all, pe_all, sp_norm, pairs, vif, cond = redundancy(df)
    sp_all.to_csv(OUT / "spearman_train_all.csv", float_format="%.4f")
    pe_all.to_csv(OUT / "pearson_train_all.csv", float_format="%.4f")
    sp_norm.to_csv(OUT / "spearman_train_normal.csv", float_format="%.4f")
    pairs.to_csv(OUT / "redundancy_pairs.csv", index=False, float_format="%.4f")
    vif.to_csv(OUT / "vif_normal_train.csv", float_format="%.3f")
    pruning = prune_histogram(sp_norm)
    pruning["covariance_conditioning_normal_train"] = cond
    (OUT / "histogram_pruning.json").write_text(json.dumps(pruning, indent=2))

    hist_kept = pruning["kept"]
    non_hist = [f for f in FEATURES if FAMILY_OF[f] != "histogram"]
    acdec = ["AC", "DL", "DS", "DP"]
    sets = {
        "ALL21": FEATURES,
        "variability only (ASTV MSTV ALTV MLTV)": FAMILY["variability"],
        "events only (AC FM UC DL DS DP)": FAMILY["events"],
        "accel/decel only (AC DL DS DP)": acdec,
        "histogram only (10)": FAMILY["histogram"],
        "baseline only (LB)": ["LB"],
        "ALL21 - variability": [f for f in FEATURES if f not in FAMILY["variability"]],
        "ALL21 - accel/decel": [f for f in FEATURES if f not in acdec],
        "ALL21 - histogram": non_hist,
        "ALL21 - Width": [f for f in FEATURES if f != "Width"],
        "ALL21 with pruned histogram": non_hist + hist_kept,
        "ALL21 + DEC_SUM": FEATURES + ["DEC_SUM"],
        "ALL21 + ACC_DEC": FEATURES + ["ACC_DEC"],
        "ALL21, DL DS DP -> DEC_SUM": [f for f in FEATURES if f not in ("DL", "DS", "DP")] + ["DEC_SUM"],
        "ALL21, AC DL DS DP -> ACC_DEC": [f for f in FEATURES if f not in acdec] + ["ACC_DEC"],
    }
    probe = group_probe(df, sets)
    probe.to_csv(OUT / "group_probe.csv", float_format="%.4f")

    fig_variability(df, sep)
    fig_events(df, pres, sep)
    fig_ranking(sep)
    fig_heatmap(sp_all)
    fig_all_ecdf(df, sep)
    fig_phenotypes(df)

    if verbose:
        pd.set_option("display.width", 250, "display.max_columns", 30)
        print(json.dumps(aud, indent=1))
        show = ["family", "normal_median", "path_median", "KS", "KS_ci_lo", "KS_ci_hi", "KS_wo_top2",
                "AUC_path_gt_normal", "AUC_abs_dev", "wasserstein_normalSD", "MI_knn_nats"]
        print(sep.sort_values("KS", ascending=False)[show].round(3).to_string())
        print(pres.round(3).to_string())
        print(pheno.round(3).T.to_string())
        print(pairs.head(15).round(3).to_string())
        print(vif.round(1).to_string())
        print(json.dumps(pruning, indent=1))
        print(f"base rate = {probe.attrs['base_rate']:.4f}")
        print(probe.drop(columns="features").round(3).to_string())
    return {"df": df, "audit": aud, "sep": sep, "presence": pres, "pairs": pairs, "vif": vif,
            "pruning": pruning, "probe": probe, "phenotypes": pheno, "spearman": sp_all}


if __name__ == "__main__":
    run()
