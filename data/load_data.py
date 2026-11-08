"""Shared data loader for CTG. Every method should load data through this file.

Example:
    import sys; sys.path.append("data")
    from load_data import load_split, fit_preprocess, FEATURES

    X_tr, y_tr = load_split("train", use_normal_only=True)   # one-class models
    X_va, y_va = load_split("val")
    X_te, y_te = load_split("test")

    prep = fit_preprocess(X_tr)          # fit on TRAIN only
    X_tr, X_va, X_te = prep(X_tr), prep(X_va), prep(X_te)

Rules: never fit anything (scaler, feature selection, model weights) on val
or test. Val anomalies may only be used for choosing the threshold tau and
hyperparameters. Test is evaluated once at the end.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "splits"

FEATURES = ["LB", "AC", "FM", "UC", "DL", "DS", "DP",
            "ASTV", "MSTV", "ALTV", "MLTV",
            "Width", "Min", "Max", "Nmax", "Nzeros",
            "Mode", "Mean", "Median", "Variance", "Tendency"]


def load_split(name, use_normal_only=False, return_exam=False):
    """Return (X, y) for 'train' | 'val' | 'test'.

    X: DataFrame of the 21 features, index = row_id. y: 0 = Normal, 1 = Pathologic.
    return_exam=True also returns the exam id of each row (for grouped CV).
    """
    df = pd.read_csv(SPLITS / f"{name}.csv", index_col="row_id")
    if use_normal_only:
        df = df[df["label"] == 0]
    out = (df[FEATURES], df["label"].to_numpy())
    return out + (df["exam"].to_numpy(),) if return_exam else out


def load_suspect(split=None):
    """Suspect (NSP 2) segments, optionally restricted to one split."""
    df = pd.read_csv(SPLITS / "suspect.csv", index_col="row_id")
    if split is not None:
        df = df[df["split"] == split]
    return df[FEATURES]


def fit_preprocess(X_train):
    """Default preprocessing: z-score with train mean / std (CTG has no NaN)."""
    mu = X_train.mean()
    sd = X_train.std().replace(0, 1)

    def transform(X):
        return ((X[FEATURES] - mu) / sd).to_numpy(dtype=np.float64)

    return transform
