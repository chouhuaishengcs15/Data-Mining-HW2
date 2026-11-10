"""Grid search for the Tier 2 autoencoder, on val only (test is never touched).

Usage (from the repo root):
    python model/hp_search.py
    python model/hp_search.py --score mse

Output: results/tier2/hp_search_<score>.csv, ranked by mean val PR-AUC.
"""
import argparse
import contextlib
import io
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import tier2_autoencoder as t2  # noqa: E402

GRID = {
    "hidden": [[8, 4], [16, 4], [16, 8], [32, 8], [32, 16], [32, 16, 8]],
    "epochs": [50, 100, 200, 400],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score", default="mahalanobis", choices=["mse", "cosine", "mahalanobis"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = p.parse_args()

    rows = []
    for hidden, epochs in itertools.product(GRID["hidden"], GRID["epochs"]):
        name = f"hp/{a.score}_h{'-'.join(map(str, hidden))}_e{epochs}"
        args = argparse.Namespace(
            hidden=hidden, epochs=epochs, batch_size=64, lr=1e-3, weight_decay=1e-5,
            score=a.score, threshold="best_f1", percentile=95, budget=10,
            contamination=0.0, seeds=a.seeds, final=False, name=name)
        with contextlib.redirect_stdout(io.StringIO()):
            t2.run(args)
        m = pd.read_csv(t2.ROOT / "results" / "tier2" / name / "metrics.csv")
        row = {"hidden": "-".join(map(str, hidden)), "epochs": epochs}
        for c in ["pr_auc", "roc_auc", "f1"]:
            row[f"{c}_mean"], row[f"{c}_std"] = m[c].mean(), m[c].std()
        row["train_s"] = m["train_s"].mean()
        rows.append(row)
        print(f"h={row['hidden']:>8} e={epochs:>3}  PR-AUC {row['pr_auc_mean']:.3f}±{row['pr_auc_std']:.3f}"
              f"  ROC-AUC {row['roc_auc_mean']:.3f}  F1 {row['f1_mean']:.3f}", flush=True)

    df = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    out = t2.ROOT / "results" / "tier2" / f"hp_search_{a.score}.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved to {out.relative_to(t2.ROOT)}\n")
    print(df.head(5).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
