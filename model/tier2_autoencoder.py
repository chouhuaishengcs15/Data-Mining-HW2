"""Tier 2 Primary Method: Autoencoder trained on Normal CTG segments only.

Pipeline: features -> z-score (fit on train normals) -> optional clip -> AE -> score s(x) -> tau.

Usage (from the repo root):
    python model/tier2_autoencoder.py                      # tune on val only
    python model/tier2_autoencoder.py --final              # also score test (run once at the end)
    python model/tier2_autoencoder.py --score mse          # ablation: scoring function
    python model/tier2_autoencoder.py --contamination 0.05 # ablation: anomalies in training
    python model/tier2_autoencoder.py --threshold percentile
    python model/tier2_autoencoder.py --feature-set PRUNED18   # P2 feature sets (EDA/outputs/feature_sets.json)
    python model/tier2_autoencoder.py --clip 5 --cov ledoit_wolf

Outputs go to results/tier2/<run name>/:
    config.json       settings of this run
    metrics.csv       one row per seed and split
    summary.csv       mean / std over seeds
    scores.csv        per-sample score, tau, prediction (all seeds)
    feature_errors.csv  per-feature squared reconstruction error (for explanation)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (average_precision_score, f1_score, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "data"))
from load_data import FEATURES, fit_preprocess, load_split  # noqa: E402


def feature_columns(name):
    """Feature names of a P2 feature set (EDA/outputs/feature_sets.json)."""
    sets = json.loads((ROOT / "EDA" / "outputs" / "feature_sets.json").read_text())
    sets["PRUNED17"] = sets.get("PRUNED17_optional", [])
    if name not in sets:
        raise SystemExit(f"unknown feature set {name}; choose from {sorted(sets)}")
    return sets[name]


# ---------------------------------------------------------------- model
class AutoEncoder(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        enc, dims = [], [d_in] + list(hidden)
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        enc = enc[:-1]                      # linear bottleneck
        dec, rdims = [], dims[::-1]
        for a, b in zip(rdims[:-1], rdims[1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        dec = dec[:-1]                      # linear output (z-scored data)
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(X, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoEncoder(X.shape[1], args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Xt = torch.tensor(X, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(args.epochs):
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), args.batch_size):
            xb = Xt[perm[i:i + args.batch_size]]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def reconstruct(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)


# ---------------------------------------------------------------- scoring s(x)
class Scorer:
    """s(x) from x and its reconstruction x_hat. Fit only uses train errors."""

    def __init__(self, kind, cov="empirical"):
        self.kind, self.cov = kind, cov

    def fit(self, X, Xh):
        if self.kind == "mahalanobis":
            E = X - Xh
            self.mu = E.mean(0)
            if self.cov == "ledoit_wolf":      # shrinkage: stabler when cov is ill-conditioned
                cov = LedoitWolf().fit(E).covariance_
            else:
                cov = np.cov(E, rowvar=False) + 1e-6 * np.eye(E.shape[1])
            self.prec = np.linalg.inv(cov)
        return self

    def __call__(self, X, Xh):
        E = X - Xh
        if self.kind == "mse":
            return (E ** 2).mean(1)
        if self.kind == "cosine":
            num = (X * Xh).sum(1)
            den = np.linalg.norm(X, axis=1) * np.linalg.norm(Xh, axis=1) + 1e-12
            return 1 - num / den
        if self.kind == "mahalanobis":
            D = E - self.mu
            return np.sqrt(np.einsum("ij,jk,ik->i", D, self.prec, D))
        raise ValueError(self.kind)


# ---------------------------------------------------------------- threshold tau
def choose_tau(args, s_train_normal, s_val, y_val):
    if args.threshold in ("best_f1", "best_f2"):   # uses val labels (allowed for tau only)
        b2 = 1.0 if args.threshold == "best_f1" else 4.0   # beta^2; F2 weights recall higher
        p, r, t = precision_recall_curve(y_val, s_val)
        p, r = p[:-1], r[:-1]
        fb = (1 + b2) * p * r / np.clip(b2 * p + r, 1e-12, None)
        return float(t[np.argmax(fb)])
    if args.threshold == "percentile":         # label-free: q-th pct of train normal scores
        return float(np.percentile(s_train_normal, args.percentile))
    if args.threshold == "budget":             # flag the top budget% of val segments
        return float(np.percentile(s_val, 100 - args.budget))
    raise ValueError(args.threshold)


def metrics(y, s, tau):
    pred = (s >= tau).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "pr_auc": average_precision_score(y, s),
        "roc_auc": roc_auc_score(y, s),
        "f1": f1_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "fpr": fp / max(int((y == 0).sum()), 1),
        "n_alerts": int(pred.sum()),
    }


# ---------------------------------------------------------------- main
def build_train(args, seed):
    """Train normals, plus optional train anomalies for the contamination ablation."""
    X_n, _ = load_split("train", use_normal_only=True)
    if args.contamination <= 0:
        return X_n, 0
    X_all, y_all = load_split("train")
    X_a = X_all[y_all == 1]
    want = int(round(args.contamination * len(X_n) / (1 - args.contamination)))
    k = min(want, len(X_a))
    if k < want:
        print(f"  [warn] only {len(X_a)} train anomalies, using {k} instead of {want}")
    X_a = X_a.sample(n=k, random_state=seed)
    return pd.concat([X_n, X_a]), k


def run(args):
    cols = feature_columns(args.feature_set)
    idx = [FEATURES.index(f) for f in cols]
    name = args.name or (f"{args.score}_{args.threshold}_c{args.contamination:g}"
                         f"_h{'-'.join(map(str, args.hidden))}_{args.feature_set}"
                         + (f"_clip{args.clip:g}" if args.clip > 0 else "")
                         + ("_lw" if args.cov == "ledoit_wolf" else ""))
    out = ROOT / "results" / "tier2" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    splits = ["val", "test"] if args.final else ["val"]
    data = {sp: load_split(sp) for sp in splits}
    rows, score_rows, err_rows = [], [], []

    for seed in args.seeds:
        X_tr_raw, n_contam = build_train(args, seed)
        zscore = fit_preprocess(X_tr_raw)      # per-column, so selecting after is the same as refitting

        def prep(X_df):
            Z = zscore(X_df)[:, idx]
            return np.clip(Z, -args.clip, args.clip) if args.clip > 0 else Z

        X_tr = prep(X_tr_raw)

        t0 = time.perf_counter()
        model = train_ae(X_tr, args, seed)
        train_s = time.perf_counter() - t0

        Xh_tr = reconstruct(model, X_tr)
        scorer = Scorer(args.score, args.cov).fit(X_tr, Xh_tr)
        s_tr = scorer(X_tr, Xh_tr)

        X_va = prep(data["val"][0])
        s_va = scorer(X_va, reconstruct(model, X_va))
        tau = choose_tau(args, s_tr, s_va, data["val"][1])

        for sp in splits:
            X_df, y = data[sp]
            X = prep(X_df)
            t0 = time.perf_counter()
            Xh = reconstruct(model, X)
            s = scorer(X, Xh)
            ms = (time.perf_counter() - t0) * 1000 / len(X)
            m = metrics(y, s, tau)
            rows.append({"seed": seed, "split": sp, "tau": tau, **m,
                         "train_s": train_s, "ms_per_sample": ms, "n_contam": n_contam})
            score_rows.append(pd.DataFrame({
                "row_id": X_df.index, "seed": seed, "split": sp, "label": y,
                "score": s, "tau": tau, "pred": (s >= tau).astype(int)}))
            err = pd.DataFrame((X - Xh) ** 2, columns=cols)
            err.insert(0, "row_id", X_df.index)
            err.insert(1, "seed", seed)
            err.insert(2, "split", sp)
            err_rows.append(err)

        v = rows[-len(splits)]
        print(f"seed {seed}: val PR-AUC {v['pr_auc']:.3f}  ROC-AUC {v['roc_auc']:.3f}  "
              f"F1 {v['f1']:.3f}  F2 {v['f2']:.3f}  P {v['precision']:.3f}  R {v['recall']:.3f}  "
              f"tau {tau:.4f}  train {train_s:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    cols = ["pr_auc", "roc_auc", "f1", "f2", "precision", "recall", "fpr",
            "n_alerts", "train_s", "ms_per_sample"]
    summ = df.groupby("split")[cols].agg(["mean", "std"])
    summ.to_csv(out / "summary.csv")
    pd.concat(score_rows).to_csv(out / "scores.csv", index=False)
    pd.concat(err_rows).to_csv(out / "feature_errors.csv", index=False)

    print(f"\n== {name}  ({len(args.seeds)} seeds, mean ± std)")
    for sp in splits:
        r = summ.loc[sp]
        print(f"{sp:>4}: " + "  ".join(
            f"{c} {r[(c, 'mean')]:.3f}±{r[(c, 'std')]:.3f}"
            for c in ["pr_auc", "roc_auc", "f1", "f2", "precision", "recall"]))
    print(f"saved to {out.relative_to(ROOT)}")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, nargs="+", default=[16, 8],
                   help="encoder layer sizes; last one is the bottleneck")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--score", choices=["mse", "cosine", "mahalanobis"], default="mahalanobis")
    p.add_argument("--threshold", choices=["best_f1", "best_f2", "percentile", "budget"], default="best_f1")
    p.add_argument("--percentile", type=float, default=95, help="for --threshold percentile")
    p.add_argument("--budget", type=float, default=10, help="%% of segments flagged, for --threshold budget")
    p.add_argument("--contamination", type=float, default=0.0,
                   help="fraction of train anomalies mixed into training (ablation)")
    p.add_argument("--feature-set", default="DROP_WIDTH20",
                   help="P2 feature set: DROP_WIDTH20 (default, Width = Max - Min removed), "
                        "ALL21, PRUNED18, NO_VARIABILITY17, PRUNED17")
    p.add_argument("--clip", type=float, default=0,
                   help="clip z-scores to [-clip, clip]; 0 = off")
    p.add_argument("--cov", choices=["empirical", "ledoit_wolf"], default="empirical",
                   help="covariance estimate for the Mahalanobis score")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--final", action="store_true", help="also evaluate on test (do this once)")
    p.add_argument("--name", default=None, help="output folder name")
    return p.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(1)
    run(parse())
