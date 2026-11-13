"""Estimate Tier 3 minus Tier 2 test AP with paired exam resampling."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

parser = argparse.ArgumentParser()
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
out = repo / "results" / "p4"
out.mkdir(parents=True, exist_ok=True)
test = pd.read_csv(repo / "data/splits/test.csv").set_index("row_id")
tier2 = repo / "results/tier2/final_mahalanobis_best_f2_DROP_WIDTH20/scores.csv"
scores = []
for path in (tier2, repo / "results/tier3/xgboost/scores.csv"):
    frame = pd.read_csv(path).query("split == 'test'")
    assert set(frame.seed) == set(range(5)) and set(frame.row_id) == set(test.index)
    assert np.array_equal(frame.label, test.loc[frame.row_id, "label"])
    scores.append(frame.pivot(index="seed", columns="row_id", values="score")
                  .reindex(index=range(5), columns=test.index).to_numpy())
assert np.isfinite(scores).all()
assert len(test) == 378 and test.exam.nunique() == 75
labels = test.label.to_numpy()
exam_codes, exams = pd.factorize(test.exam)

def gap(weights=None):
    ap = [np.mean([average_precision_score(labels, row, sample_weight=weights)
                   for row in model_scores]) for model_scores in scores]
    return float(ap[1] - ap[0])

rng = np.random.default_rng(20261114)
draws = 10000
gaps = []
skipped = 0
for _ in range(draws):
    sampled = rng.integers(len(exams), size=len(exams))
    weights = np.bincount(sampled, minlength=len(exams))[exam_codes]
    if np.unique(labels[weights > 0]).size < 2:
        skipped += 1
        continue
    gaps.append(gap(weights))

result = dict(comparison="Tier 3 minus Tier 2, mean of five seed-level AP differences",
              test_rows=len(test), exams=len(exams), draws=draws, skipped=skipped,
              rng_seed=20261114, observed_gap=gap(),
              percentile_95_ci=np.quantile(gaps, [0.025, 0.975]).tolist())
(out / "bootstrap_ap.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
