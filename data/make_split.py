"""Create the shared train/val/test split for the UCI Cardiotocography (CTG) data.

Usage (from repo root):
    python data/make_split.py

Input:
    data/raw/CTG.xls        original UCI file (sheet "Raw Data")

Outputs (data/splits/):
    split.csv               index of every segment: row_id, exam, date, nsp, label, split
    train.csv / val.csv / test.csv
                            Normal + Pathologic segments with the 21 features
    suspect.csv             Suspect segments (not used for training / main metrics)
    stats.csv               counts per split

Design (see data/README.md):
- Anomaly = NSP 3 (Pathologic), label 1. Normal = NSP 1, label 0.
  Suspect (NSP 2) is a clinical grey zone; it is kept out of the main task
  and saved separately for error analysis.
- Several segments come from the same CTG exam (FileName). The split is done
  per exam, so no exam appears in two splits (prevents leakage).
- Exams are stratified by "has any Pathologic segment" and cut 60 / 20 / 20
  by segment count, with a fixed seed.
- Features: the 21 standard UCI features. Event counts (AC, FM, UC, DL, DS, DP)
  are converted to events per second by dividing by the segment length (e - b),
  exactly as in the UCI "Data" sheet. Expert pattern columns (A ... SUSP, CLASS)
  are excluded because they are derived from the expert labelling.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "CTG.xls"
OUT = ROOT / "splits"
SEED = 42
RATIOS = (0.6, 0.2, 0.2)

RATE_FEATURES = ["AC", "FM", "UC", "DL", "DS", "DP"]
FEATURES = ["LB", "AC", "FM", "UC", "DL", "DS", "DP",
            "ASTV", "MSTV", "ALTV", "MLTV",
            "Width", "Min", "Max", "Nmax", "Nzeros",
            "Mode", "Mean", "Median", "Variance", "Tendency"]


def load_raw():
    """Return one row per CTG segment with exam info, 21 features and NSP."""
    raw = pd.read_excel(RAW, sheet_name="Raw Data")
    raw = raw[raw["FileName"].notna() & raw["NSP"].notna()].reset_index(drop=True)
    assert len(raw) == 2126

    df = pd.DataFrame({
        "row_id": np.arange(len(raw)),
        "exam": raw["FileName"].astype(str),
        "segment": raw["SegFile"].astype(str),
        "date": pd.to_datetime(raw["Date"]).dt.date,
    })
    duration = raw["e"] - raw["b"]  # seconds
    for c in FEATURES:
        df[c] = raw[c] / duration if c in RATE_FEATURES else raw[c]
    df["nsp"] = raw["NSP"].astype(int)
    return df


def assign_split(df):
    """Exam-level stratified split; returns a Series exam -> split name."""
    rng = np.random.default_rng(SEED)
    exams = df.groupby("exam").agg(n=("row_id", "size"),
                                   has_path=("nsp", lambda s: (s == 3).any()))
    result = {}
    for _, grp in exams.groupby("has_path"):
        order = rng.permutation(np.array(grp.index))  # copy; never shuffle the Index in place
        cum = grp["n"].reindex(order).cumsum().to_numpy() / grp["n"].sum()
        for exam, c in zip(order, cum):
            if c <= RATIOS[0] + 1e-9:
                result[exam] = "train"
            elif c <= RATIOS[0] + RATIOS[1] + 1e-9:
                result[exam] = "val"
            else:
                result[exam] = "test"
    return pd.Series(result)


def main():
    df = load_raw()
    df["split"] = df["exam"].map(assign_split(df))
    df["label"] = df["nsp"].map({1: 0, 2: -1, 3: 1})  # -1 = Suspect (excluded)

    OUT.mkdir(exist_ok=True)
    df[["row_id", "exam", "segment", "date", "nsp", "label", "split"]].to_csv(
        OUT / "split.csv", index=False)

    cols = ["row_id", "exam", "label"] + FEATURES
    main_rows = df[df["label"] >= 0]
    for name in ["train", "val", "test"]:
        main_rows[main_rows["split"] == name][cols].to_csv(OUT / f"{name}.csv", index=False)
    df[df["label"] == -1][["row_id", "exam", "split"] + FEATURES].to_csv(
        OUT / "suspect.csv", index=False)

    stats = df.groupby("split").agg(
        exams=("exam", "nunique"),
        normal=("nsp", lambda s: (s == 1).sum()),
        pathologic=("nsp", lambda s: (s == 3).sum()),
        suspect=("nsp", lambda s: (s == 2).sum()),
    ).reindex(["train", "val", "test"])
    stats.loc["all"] = [df["exam"].nunique(), (df.nsp == 1).sum(),
                        (df.nsp == 3).sum(), (df.nsp == 2).sum()]
    stats["n_main"] = stats["normal"] + stats["pathologic"]
    stats["anomaly_rate_%"] = (100 * stats["pathologic"] / stats["n_main"]).round(2)
    stats.to_csv(OUT / "stats.csv")
    print(stats.to_string())

    # sanity check: no exam in two splits
    assert df.groupby("exam")["split"].nunique().max() == 1


if __name__ == "__main__":
    main()
