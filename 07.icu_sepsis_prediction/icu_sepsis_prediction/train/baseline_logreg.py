from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train.metrics import compute_metrics


SEED = 42


def load_windows_from_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_windows_from_db() -> pd.DataFrame:
    """
    Reads from icu.windows (JSONB features) if you used load_db.py.
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="labor_db",
        user="admin",
        password="admin",
    )
    q = "SELECT patient_id, t_hour, y, features FROM icu.windows;"
    df = pd.read_sql(q, conn)
    conn.close()

    feats = pd.json_normalize(df["features"])
    out = pd.concat([df.drop(columns=["features"]), feats], axis=1)
    return out


def split_by_patient(df: pd.DataFrame, test_size: float = 0.2):
    patients = df["patient_id"].astype(str).unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(patients)
    n_test = max(1, int(len(patients) * test_size))
    test_ids = set(patients[:n_test])

    df_train = df[~df["patient_id"].isin(test_ids)].reset_index(drop=True)
    df_test = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)
    return df_train, df_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, choices=["file", "db"], default="file")
    ap.add_argument("--windows_path", type=str, default="data/processed/windows.parquet")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    if args.source == "db":
        df = load_windows_from_db()
    else:
        df = load_windows_from_file(Path(args.windows_path).expanduser().resolve())

    if df.empty:
        raise RuntimeError("windows dataset is empty. Run etl.make_windows first.")

    # Features / target
    y = df["y"].astype(int).values
    X = df.drop(columns=["patient_id", "t_hour", "y"], errors="ignore")

    # split by patient_id
    df_train, df_test = split_by_patient(df, test_size=args.test_size)

    y_train = df_train["y"].astype(int).values
    X_train = df_train.drop(columns=["patient_id", "t_hour", "y"], errors="ignore")

    y_test = df_test["y"].astype(int).values
    X_test = df_test.drop(columns=["patient_id", "t_hour", "y"], errors="ignore")

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        ]
    )

    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, probs, threshold=args.threshold)

    print("\n=== Logistic Regression Baseline ===")
    print("Patients(train/test):", df_train["patient_id"].nunique(), "/", df_test["patient_id"].nunique())
    print("Rows(train/test):", len(df_train), "/", len(df_test))
    print("AUROC:", m.auroc)
    print("AUPRC:", m.auprc)
    print("Confusion (tn fp fn tp):", m.tn, m.fp, m.fn, m.tp)

    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    dump(pipe, artifacts / "logreg.joblib")
    (artifacts / "metrics_logreg.json").write_text(json.dumps(m.to_dict(), indent=2), encoding="utf-8")
    print("\n✅ Saved:", artifacts / "logreg.joblib")
    print("✅ Saved:", artifacts / "metrics_logreg.json")


if __name__ == "__main__":
    main()
