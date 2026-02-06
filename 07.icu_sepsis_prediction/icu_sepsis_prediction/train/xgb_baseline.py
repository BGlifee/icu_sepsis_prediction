from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from train.baseline_logreg import split_by_patient, load_windows_from_file

SEED = 42


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auroc": auroc,
        "auprc": auprc,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_path", type=str, default="data/processed/windows.parquet")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--num_boost_round", type=int, default=5000)
    args = ap.parse_args()

    import xgboost as xgb

    df = load_windows_from_file(Path(args.windows_path).expanduser().resolve())
    if df.empty:
        raise RuntimeError("windows dataset is empty. Run etl.make_windows first.")

    df_train, df_test = split_by_patient(df, test_size=args.test_size)

    y_train = df_train["y"].astype(int).values
    X_train = df_train.drop(columns=["patient_id", "t_hour", "y"], errors="ignore")

    y_test = df_test["y"].astype(int).values
    X_test = df_test.drop(columns=["patient_id", "t_hour", "y"], errors="ignore")

    # drop columns that are all-missing in TRAIN (prevents imputer from dropping them implicitly)
    all_missing = X_train.isna().all(axis=0)
    drop_cols = list(X_train.columns[all_missing])
    if drop_cols:
        print(f"Dropping all-missing columns in train: {drop_cols}")
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols)


    # impute
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)

    # handle imbalance
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    dtrain = xgb.DMatrix(X_train_imp, label=y_train, feature_names=list(X_train.columns))
    dtest = xgb.DMatrix(X_test_imp, label=y_test, feature_names=list(X_train.columns))

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "eta": 0.02,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "seed": SEED,
        "scale_pos_weight": scale_pos_weight,
    }

    evals = [(dtrain, "train"), (dtest, "test")]
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False,
    )

    probs = booster.predict(dtest)
    metrics = compute_metrics(y_test, probs, threshold=args.threshold)

    print("\n=== XGBoost Baseline (native xgb.train) ===")
    print("Patients(train/test):", df_train["patient_id"].nunique(), "/", df_test["patient_id"].nunique())
    print("Rows(train/test):", len(df_train), "/", len(df_test))
    print("AUROC:", metrics["auroc"])
    print("AUPRC:", metrics["auprc"])
    print("Confusion (tn fp fn tp):", metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"])
    print("Best iteration:", getattr(booster, "best_iteration", None))

    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    # save model + imputer
    booster.save_model(str(artifacts / "xgb.json"))
    dump({"imputer": imp, "feature_names": list(X_train.columns)}, artifacts / "xgb_imputer.joblib")
    (artifacts / "metrics_xgb.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # feature importance (gain)
    score = booster.get_score(importance_type="gain")
    fi = (
        pd.DataFrame({"feature": list(score.keys()), "gain": list(score.values())})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    fi.to_csv(artifacts / "feature_importance.csv", index=False)

    print("\n✅ Saved:", artifacts / "xgb.json")
    print("✅ Saved:", artifacts / "xgb_imputer.joblib")
    print("✅ Saved:", artifacts / "metrics_xgb.json")
    print("✅ Saved:", artifacts / "feature_importance.csv")


if __name__ == "__main__":
    main()
