"""Run supervised CTG XGBoost on the shared P1 splits."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score, confusion_matrix, fbeta_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument("--final", action="store_true", help="Evaluate the held-out test split")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "data"))
from load_data import load_split

out = repo / "results" / "tier3" / "xgboost"
out.mkdir(parents=True, exist_ok=True)
seeds = range(5)
features = [c for c in load_split("train")[0].columns if c != "Width"]
train, y_train, train_exams = load_split("train", return_exam=True)
val, y_val, val_exams = load_split("val", return_exam=True)
assert not set(train_exams).intersection(val_exams)
train = train[features].to_numpy(dtype=np.float32)
val = val[features].to_numpy(dtype=np.float32)
weight = float(np.count_nonzero(y_train == 0) / np.count_nonzero(y_train == 1))

search, candidates = [], {}
for idx, (depth, rate, balance) in enumerate(itertools.product(
    (2, 4), (0.03, 0.1), (1.0, weight)
)):
    params = dict(max_depth=depth, learning_rate=rate, scale_pos_weight=balance,
                  n_estimators=200, objective="binary:logistic", tree_method="hist",
                  subsample=0.8, colsample_bytree=0.8, n_jobs=1)
    candidates[idx] = []
    for seed in seeds:
        model = xgb.XGBClassifier(**params, random_state=seed)
        start = perf_counter()
        model.fit(train, y_train)
        train_s = perf_counter() - start
        scores = model.predict_proba(val)[:, 1]
        p, r, thresholds = precision_recall_curve(y_val, scores)
        f2 = 5 * p[:-1] * r[:-1] / (4 * p[:-1] + r[:-1] + 1e-12)
        tau = float(thresholds[np.argmax(f2)])
        ap = average_precision_score(y_val, scores)
        search.append(dict(config=idx, seed=seed, val_ap=ap, tau=tau, **params))
        candidates[idx].append((model, tau, train_s, scores))

search = pd.DataFrame(search)
search.to_csv(out / "hp_search.csv", index=False)
winner = int(search.groupby("config").val_ap.mean().idxmax())
selection = dict(config=winner, params=candidates[winner][0][0].get_params(),
                 features=features, seeds=list(seeds), validation_selection="mean AP",
                 threshold_selection="per-seed maximum validation F2; lowest threshold on ties",
                 source_commit=subprocess.check_output(
                     ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
                 ).strip(), xgboost=xgb.__version__, sklearn=sklearn.__version__)
(out / "tuning.json").write_text(json.dumps(selection, indent=2) + "\n")

if not args.final:
    print(f"Validation search complete. Selected config {winner}.")
    sys.exit(0)

test, y_test, test_exams = load_split("test", return_exam=True)
assert not set(test_exams).intersection(set(train_exams) | set(val_exams))
assert test.index.is_unique and len(test) == 378 and len(set(test_exams)) == 75
X_test = test[features].to_numpy(dtype=np.float32)

rows, predictions = [], []
for seed, (model, tau, train_s, val_scores) in enumerate(candidates[winner]):
    test_scores = model.predict_proba(X_test)[:, 1]
    repeats = 100
    start = perf_counter()
    for _ in range(repeats):
        model.predict_proba(X_test)
    latency = (perf_counter() - start) * 1000 / (repeats * len(test))
    pred = test_scores >= tau
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    rows.append(dict(seed=seed, tau=tau, val_ap=average_precision_score(y_val, val_scores),
                     ap=average_precision_score(y_test, test_scores),
                     f2=fbeta_score(y_test, pred, beta=2), roc_auc=roc_auc_score(y_test, test_scores),
                     precision=precision_score(y_test, pred, zero_division=0),
                     recall=recall_score(y_test, pred), fpr=fp/(fp+tn),
                     tp=tp, fp=fp, fn=fn, tn=tn, train_s=train_s, batch_ms_per_row=latency))
    predictions.append(pd.DataFrame(dict(row_id=test.index, seed=seed, split="test",
                                         label=y_test, score=test_scores, tau=tau,
                                         pred=pred.astype(int))))
metrics = pd.DataFrame(rows)
metrics.to_csv(out / "metrics.csv", index=False)
pd.concat(predictions, ignore_index=True).to_csv(out / "scores.csv", index=False)
metrics.drop(columns="seed").agg(["mean", "std"]).T.to_csv(out / "summary.csv")
print(metrics.to_string(index=False))
print(metrics.drop(columns="seed").agg(["mean", "std"]).T)
