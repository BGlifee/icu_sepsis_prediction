from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

# Reuse your existing helpers (make sure these imports work in your repo)
from train.baseline_logreg import split_by_patient, load_windows_from_file

SEED = 42


# -----------------------------
# Metrics
# -----------------------------
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


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def drop_all_missing_train_cols(X_train: pd.DataFrame, X_other: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Drop columns that are entirely missing in X_train, apply same drops to X_other."""
    all_missing = X_train.isna().all(axis=0)
    drop_cols = list(X_train.columns[all_missing])
    if drop_cols:
        X_train = X_train.drop(columns=drop_cols)
        X_other = X_other.drop(columns=drop_cols)
    return X_train, X_other, drop_cols


def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


# -----------------------------
# CV evaluation (patient-level)
# -----------------------------
def evaluate_groupkfold_xgb(
    df: pd.DataFrame,
    base_params: dict,
    n_splits: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    threshold: float,
    seed: int = SEED,
) -> dict:
    import xgboost as xgb

    groups = df["patient_id"].astype(str).values
    y_all = df["y"].astype(int).values
    X_all = df.drop(columns=["patient_id", "t_hour", "y"], errors="ignore")

    gkf = GroupKFold(n_splits=n_splits)
    fold_rows: List[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), start=1):
        X_tr = X_all.iloc[tr_idx].copy()
        y_tr = y_all[tr_idx]
        X_va = X_all.iloc[va_idx].copy()
        y_va = y_all[va_idx]

        # Drop train all-missing columns
        X_tr, X_va, drop_cols = drop_all_missing_train_cols(X_tr, X_va)

        # Impute
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)
        X_va_imp = imp.transform(X_va)

        # fold imbalance
        spw = compute_scale_pos_weight(y_tr)

        params = dict(base_params)
        params["seed"] = int(seed + fold)
        params["scale_pos_weight"] = float(spw)

        dtr = xgb.DMatrix(X_tr_imp, label=y_tr, feature_names=list(X_tr.columns))
        dva = xgb.DMatrix(X_va_imp, label=y_va, feature_names=list(X_tr.columns))

        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=num_boost_round,
            evals=[(dtr, "train"), (dva, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        prob_va = booster.predict(dva)
        m = compute_metrics(y_va, prob_va, threshold=threshold)

        fold_rows.append(
            {
                "fold": fold,
                "n_train_rows": int(len(tr_idx)),
                "n_valid_rows": int(len(va_idx)),
                "n_train_patients": int(pd.Series(groups[tr_idx]).nunique()),
                "n_valid_patients": int(pd.Series(groups[va_idx]).nunique()),
                "drop_cols_count": int(len(drop_cols)),
                "best_iteration": int(getattr(booster, "best_iteration", -1)),
                **m,
            }
        )

    df_folds = pd.DataFrame(fold_rows)

    summary = {
        "n_splits": int(n_splits),
        "threshold": float(threshold),
        "auroc_mean": float(df_folds["auroc"].mean()),
        "auroc_std": float(df_folds["auroc"].std(ddof=1)) if n_splits > 1 else float("nan"),
        "auprc_mean": float(df_folds["auprc"].mean()),
        "auprc_std": float(df_folds["auprc"].std(ddof=1)) if n_splits > 1 else float("nan"),
        "folds": df_folds.to_dict(orient="records"),
    }
    return summary


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_path", type=str, default="data/processed/windows.parquet")
    ap.add_argument("--long_path", type=str, default="data/processed/long.parquet")

    # split + eval
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5)

    # xgb train
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--num_boost_round", type=int, default=5000)
    ap.add_argument("--horizon_h", type=int, default=6)

    # CV evaluation
    ap.add_argument("--do_cv", action="store_true", help="Run GroupKFold CV on TRAIN patients and save metrics.")
    ap.add_argument("--cv_splits", type=int, default=5)

    # output dirs
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--powerbi_dir", type=str, default="powerbi")

    args = ap.parse_args()

    import xgboost as xgb

    windows_path = Path(args.windows_path).expanduser().resolve()
    df = load_windows_from_file(windows_path)
    if df.empty:
        raise RuntimeError("windows dataset is empty. Run etl.make_windows first.")

    # patient-holdout split
    df_train, df_test = split_by_patient(df, test_size=args.test_size)

    # Optional: CV on TRAIN (patient-level)
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "eta": 0.02,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
    }

    artifacts = ensure_dir(Path(args.artifacts_dir))
    powerbi_dir = ensure_dir(Path(args.powerbi_dir))

    cv_summary = None
    if args.do_cv:
        cv_summary = evaluate_groupkfold_xgb(
            df=df_train,
            base_params=base_params,
            n_splits=int(args.cv_splits),
            num_boost_round=int(args.num_boost_round),
            early_stopping_rounds=int(args.early_stopping_rounds),
            threshold=float(args.threshold),
            seed=SEED,
        )
        # save CV results (json + csv)
        (artifacts / "metrics_xgb_cv.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
        df_cv = pd.DataFrame(cv_summary["folds"])
        df_cv.to_csv(artifacts / "metrics_xgb_cv_folds.csv", index=False)
        print("\n=== GroupKFold CV (TRAIN patients only) ===")
        print("AUROC mean±std:", cv_summary["auroc_mean"], "±", cv_summary["auroc_std"])
        print("AUPRC mean±std:", cv_summary["auprc_mean"], "±", cv_summary["auprc_std"])
        print("✅ Saved:", artifacts / "metrics_xgb_cv.json")
        print("✅ Saved:", artifacts / "metrics_xgb_cv_folds.csv")

    # Prepare train/test matrices
    y_train = df_train["y"].astype(int).values
    X_train = df_train.drop(columns=["patient_id", "t_hour", "y"], errors="ignore")

    y_test = df_test["y"].astype(int).values
    X_test = df_test.drop(columns=["patient_id", "t_hour", "y"], errors="ignore")

    # Drop all-missing columns in train and apply to test
    X_train, X_test, drop_cols = drop_all_missing_train_cols(X_train, X_test)
    if drop_cols:
        print(f"Dropping all-missing columns in train (applied to test too): {drop_cols}")

    # Impute
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)

    # Imbalance weight (TRAIN 기준)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    dtrain = xgb.DMatrix(X_train_imp, label=y_train, feature_names=list(X_train.columns))
    dtest = xgb.DMatrix(X_test_imp, label=y_test, feature_names=list(X_train.columns))

    params = dict(base_params)
    params["seed"] = SEED
    params["scale_pos_weight"] = float(scale_pos_weight)

    # NOTE: early stopping should use a VALID set, not TEST.
    # Here, we do not early-stop on test. We train with early stopping disabled OR
    # (recommended) early stop via CV above, then train fixed rounds.
    # For simplicity: if do_cv, we can use average best_iteration, else train full rounds without early stop.
    num_round = int(args.num_boost_round)
    es_rounds = int(args.early_stopping_rounds)

    best_iter = None
    if cv_summary is not None:
        best_iters = [r.get("best_iteration", -1) for r in cv_summary["folds"] if r.get("best_iteration", -1) >= 0]
        if best_iters:
            # xgb.best_iteration is 0-based; add 1 for num_boost_round
            best_iter = int(round(float(np.mean(best_iters)))) + 1
            num_round = max(50, best_iter)
            print(f"\nUsing num_boost_round from CV avg best_iteration: {num_round}")

    # Train final model
    # If we don't have a proper valid set here, don't early-stop on test.
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train")],
        verbose_eval=False,
    )

    # Predict + metrics on HOLDOUT TEST
    probs = booster.predict(dtest)
    metrics = compute_metrics(y_test, probs, threshold=float(args.threshold))

    print("\n=== XGBoost Baseline (Patient-holdout TEST) ===")
    print("Patients(train/test):", df_train["patient_id"].nunique(), "/", df_test["patient_id"].nunique())
    print("Rows(train/test):", len(df_train), "/", len(df_test))
    print("AUROC:", metrics["auroc"])
    print("AUPRC:", metrics["auprc"])
    print("Confusion (tn fp fn tp):", metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"])

    # Save artifacts
    booster.save_model(str(artifacts / "xgb.json"))
    dump({"imputer": imp, "feature_names": list(X_train.columns), "dropped_cols": drop_cols}, artifacts / "xgb_imputer.joblib")
    (artifacts / "metrics_xgb.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (artifacts / "xgb_features.json").write_text(json.dumps(list(X_train.columns), indent=2), encoding="utf-8")

    # Also save a single-row CSV summary (nice for quick checks / BI cards)
    df_metrics_row = pd.DataFrame(
        [{
            "model_name": "xgboost",
            "split": "patient_holdout_test",
            "threshold": metrics["threshold"],
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
            "n_train_patients": int(df_train["patient_id"].nunique()),
            "n_test_patients": int(df_test["patient_id"].nunique()),
            "n_train_rows": int(len(df_train)),
            "n_test_rows": int(len(df_test)),
            "num_boost_round": int(num_round),
            "scale_pos_weight": float(scale_pos_weight),
            "dropped_cols_count": int(len(drop_cols)),
            "horizon_h": int(args.horizon_h),
        }]
    )
    df_metrics_row.to_csv(artifacts / "metrics_xgb_summary.csv", index=False)

    print("\n✅ Saved:", artifacts / "xgb.json")
    print("✅ Saved:", artifacts / "xgb_imputer.joblib")
    print("✅ Saved:", artifacts / "metrics_xgb.json")
    print("✅ Saved:", artifacts / "xgb_features.json")
    print("✅ Saved:", artifacts / "metrics_xgb_summary.csv")

    # -----------------------------
    # Export for Power BI (CSV)
    # -----------------------------
    # 1) enriched predictions (risk + selected aggregated vitals)
    vital_prefixes = ["HR", "MAP", "Resp", "O2Sat", "Temp"]
    vital_stats = ["__last", "__mean", "__std", "__min", "__max"]
    vital_cols = [f"{p}{s}" for p in vital_prefixes for s in vital_stats if f"{p}{s}" in df_test.columns]

    df_pred_enriched = df_test[["patient_id", "t_hour"] + vital_cols].copy()
    df_pred_enriched["actual_label"] = y_test
    df_pred_enriched["risk_score"] = probs
    df_pred_enriched["predicted_label"] = (probs >= float(args.threshold)).astype(int)
    df_pred_enriched["threshold"] = float(args.threshold)
    df_pred_enriched["model_name"] = "xgboost"
    df_pred_enriched["horizon_h"] = int(args.horizon_h)

    pred_path = powerbi_dir / "fact_predictions_enriched.csv"
    df_pred_enriched.to_csv(pred_path, index=False)
    print("\n✅ Saved:", pred_path)

    # 1-b) metrics summary for BI cards
    bi_metrics_path = powerbi_dir / "fact_model_metrics.csv"
    df_metrics_row.to_csv(bi_metrics_path, index=False)
    print("✅ Saved:", bi_metrics_path)

    # 2) raw long + risk overlay (hourly vitals + risk_score at matching hours)
    long_path = Path(args.long_path).expanduser().resolve()
    if long_path.exists():
        if long_path.suffix.lower() == ".parquet":
            df_long = pd.read_parquet(long_path)
        else:
            df_long = pd.read_csv(long_path)

        df_long_with_risk = df_long.merge(
            df_pred_enriched[["patient_id", "t_hour", "risk_score", "predicted_label", "threshold", "horizon_h"]],
            left_on=["patient_id", "hour"],
            right_on=["patient_id", "t_hour"],
            how="left",
        ).drop(columns=["t_hour"])

        out_overlay = powerbi_dir / "fact_long_with_risk.csv"
        df_long_with_risk.to_csv(out_overlay, index=False)
        print("✅ Saved:", out_overlay)
    else:
        print("⚠ long file not found. Skip overlay export:", long_path)


if __name__ == "__main__":
    main()