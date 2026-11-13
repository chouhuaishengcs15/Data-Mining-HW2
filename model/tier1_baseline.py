"""Evaluate a single-feature CTG baseline using the shared splits."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             fbeta_score, precision_recall_curve, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from load_data import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="Also evaluate test")
    args = parser.parse_args()
    out = ROOT / "results" / "tier1" / "astv"
    out.mkdir(parents=True, exist_ok=True)

    val, y_val = load_split("val")
    p, r, thresholds = precision_recall_curve(y_val, val["ASTV"])
    f2 = 5 * p[:-1] * r[:-1] / np.clip(4 * p[:-1] + r[:-1], 1e-12, None)
    tau = thresholds[np.argmax(f2)]
    splits = {"val": (val, y_val)}
    if args.final:
        splits["test"] = load_split("test")

    metrics, predictions = [], []
    for split, (features, labels) in splits.items():
        score = features["ASTV"].to_numpy()
        pred = score >= tau
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        metrics.append(dict(split=split, tau=tau,
                            ap=average_precision_score(labels, score),
                            f2=fbeta_score(labels, pred, beta=2),
                            roc_auc=roc_auc_score(labels, score),
                            tp=tp, fp=fp, fn=fn, tn=tn,
                            recall=tp / (tp + fn)))
        predictions.append(pd.DataFrame(dict(row_id=features.index, seed=pd.NA,
                                             split=split, label=labels, score=score,
                                             tau=tau, pred=pred.astype(int))))
    suffix = "" if args.final else "_val"
    frame = pd.DataFrame(metrics)
    frame.to_csv(out / f"metrics{suffix}.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        out / f"scores{suffix}.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
