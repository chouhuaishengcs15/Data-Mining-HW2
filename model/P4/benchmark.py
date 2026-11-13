"""CPU timing on in-memory validation rows, including score and alert."""
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from threadpoolctl import threadpool_limits

p = argparse.ArgumentParser()
p.add_argument("tier", type=int, choices=[1, 2, 3])
args = p.parse_args()
repo = Path(__file__).resolve().parents[2]
out = repo / "results" / "p4"
out.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(repo / "model"), str(repo / "data")]
from load_data import load_split

def timed(call):
    start = perf_counter_ns()
    result = call()
    return result, (perf_counter_ns() - start) / 1e9

def threshold(scores):
    precision, recall, thresholds = precision_recall_curve(y_val, scores)
    f2 = 5 * precision[:-1] * recall[:-1] / np.clip(
        4 * precision[:-1] + recall[:-1], 1e-12, None)
    return thresholds[np.argmax(f2)]

val, y_val = load_split("val")
val_rows = val.to_numpy(dtype=np.float64)
records = []

def measure(seed, train_s, calibration_s, predict, rows):
    for row in rows:
        predict(row)
    repeats = 20
    _, elapsed = timed(lambda: predict_all(predict, rows, repeats))
    records.append(dict(tier=args.tier, seed=seed, train_s=train_s,
                        threshold_s=calibration_s,
                        single_row_ms=elapsed * 1000 / (repeats * len(rows))))

def predict_all(predict, rows, repeats):
    for _ in range(repeats):
        for row in rows:
            predict(row)

with threadpool_limits(limits=1):
    if args.tier == 1:
        astv = val.columns.get_loc("ASTV")
        scores = val_rows[:, astv]
        tau, calibration = timed(lambda: threshold(scores))
        assert tau == 59
        measure(None, 0, calibration, lambda row: row[astv] >= tau, val_rows)
    elif args.tier == 2:
        import torch
        import tier2_autoencoder as ae

        torch.set_num_threads(1)
        cfg = SimpleNamespace(**json.loads((repo / "results/tier2"
            / "final_mahalanobis_best_f2_DROP_WIDTH20/config.json").read_text()))
        train, _ = load_split("train", use_normal_only=True)
        cols = ae.feature_columns(cfg.feature_set)
        idx = [val.columns.get_loc(c) for c in cols]
        mu = train.mean().to_numpy()[idx]
        sd = train.std().replace(0, 1).to_numpy()[idx]
        X = (train.to_numpy()[:, idx] - mu) / sd
        rows = val_rows[:, idx]
        zval = (rows - mu) / sd
        # Warm up PyTorch's optimizer initialization before timing.
        ae.train_ae(X, SimpleNamespace(**{**vars(cfg), "epochs": 1}), 0)

        def fit(seed):
            model = ae.train_ae(X, cfg, seed)
            scorer = ae.Scorer(cfg.score, cfg.cov).fit(X, ae.reconstruct(model, X))
            return model, scorer

        for seed in cfg.seeds:
            (model, scorer), train_s = timed(lambda: fit(seed))
            tau, calibration = timed(lambda: threshold(
                scorer(zval, ae.reconstruct(model, zval))))

            def predict(row):
                z = ((row - mu) / sd)[None, :]
                return scorer(z, ae.reconstruct(model, z))[0] >= tau

            measure(seed, train_s, calibration, predict, rows)
    else:
        import xgboost as xgb

        tier3 = repo / "results/tier3/xgboost"
        cfg = json.loads((tier3 / "tuning.json").read_text())
        winner = pd.read_csv(tier3 / "hp_search.csv")
        winner = winner.loc[winner.config == cfg["config"]]
        train, labels = load_split("train")
        X = train[cfg["features"]].to_numpy(dtype=np.float32)
        rows = val[cfg["features"]].to_numpy(dtype=np.float32)
        params = {k: v for k, v in cfg["params"].items()
                  if v is not None and k != "random_state"}
        params["n_estimators"] = int(winner.n_estimators.iloc[0])
        xgb.XGBClassifier(**{**params, "n_estimators": 1}).fit(X, labels)
        for seed in cfg["seeds"]:
            model = xgb.XGBClassifier(**params, random_state=seed)
            _, train_s = timed(lambda: model.fit(X, labels))
            tau, calibration = timed(lambda: threshold(model.predict_proba(rows)[:, 1]))
            assert np.isclose(tau, winner.loc[winner.seed == seed, "tau"].iloc[0])

            def predict(row):
                return model.predict_proba(row[None, :])[0, 1] >= tau

            measure(seed, train_s, calibration, predict, rows)

frame = pd.DataFrame(records)
frame.to_csv(out / f"timing_tier{args.tier}.csv", index=False)
print(platform.platform())
print(frame.to_string(index=False))

files = [out / f"timing_tier{tier}.csv" for tier in (1, 2, 3)]
if all(path.is_file() for path in files):
    combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    columns = ["train_s", "threshold_s", "single_row_ms"]
    combined.groupby("tier")[columns].agg(["mean", "std"]).to_csv(out / "timing_summary.csv")
