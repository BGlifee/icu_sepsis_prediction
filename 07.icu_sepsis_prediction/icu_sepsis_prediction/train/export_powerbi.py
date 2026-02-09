from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import numpy as np


# =========================
# Paths
# =========================
LONG_PATH = Path("data/processed/long.parquet")
WINDOWS_PATH = Path("data/processed/windows.parquet")
FEATURE_IMPORTANCE_PATH = Path("artifacts/feature_importance.csv")
METRICS_PATH = Path("artifacts/metrics_xgb.json")

OUT_DIR = Path("powerbi")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. DimPatient
# =========================
def export_dim_patient(df_long: pd.DataFrame) -> None:
    dim_patient = (
        df_long[["patient_id"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    dim_patient.to_csv(OUT_DIR / "dim_patient.csv", index=False)
    print("✅ dim_patient.csv saved")


# =========================
# 2. FactVitals (time series)
# =========================
def export_fact_vitals(df_long: pd.DataFrame) -> None:
    vitals_cols = [
        "patient_id",
        "hour",
        "SepsisLabel",
    ]

    # 나머지 컬럼은 vitals/labs
    feature_cols = [
        c for c in df_long.columns
        if c not in vitals_cols
        and c not in ["patient_id", "hour"]
    ]

    fact_vitals = df_long[
        ["patient_id", "hour", "SepsisLabel"] + feature_cols
    ].sort_values(["patient_id", "hour"])

    fact_vitals.to_csv(OUT_DIR / "fact_vitals.csv", index=False)
    print("✅ fact_vitals.csv saved")


# =========================
# 3. FactPredictions
# =========================
def export_fact_predictions(df_windows: pd.DataFrame) -> None:
    fact_pred = df_windows[
        ["patient_id", "t_hour", "y"]
    ].copy()

    # risk score column name (xgb / logreg 둘 다 대응)
    risk_cols = [c for c in df_windows.columns if "prob" in c or "risk" in c]
    if risk_cols:
        fact_pred["risk_score"] = df_windows[risk_cols[0]]
    else:
        fact_pred["risk_score"] = np.nan

    fact_pred.rename(columns={"y": "actual_label"}, inplace=True)
    fact_pred["predicted_label"] = (fact_pred["risk_score"] >= 0.5).astype(int)
    fact_pred["threshold"] = 0.5
    fact_pred["model_name"] = "xgboost"
    fact_pred["horizon_h"] = 6

    fact_pred.sort_values(["patient_id", "t_hour"]).to_csv(
        OUT_DIR / "fact_predictions.csv", index=False
    )
    print("✅ fact_predictions.csv saved")


# =========================
# 4. FactFeatureContribution
# =========================
def export_feature_importance() -> None:
    fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    fi["model_name"] = "xgboost"
    fi["window_h"] = 24
    fi["horizon_h"] = 6

    fi.rename(columns={"gain": "importance_gain"}, inplace=True)
    fi.to_csv(OUT_DIR / "fact_feature_contribution.csv", index=False)
    print("✅ fact_feature_contribution.csv saved")


# =========================
# 5. Patient Snapshot (explainable view)
# =========================
def export_patient_snapshot(df_windows: pd.DataFrame) -> None:
    # 최근 시점 기준 snapshot
    snap = df_windows.sort_values("t_hour").groupby("patient_id").tail(1)

    cols = ["patient_id", "t_hour", "y"]
    feature_cols = [c for c in snap.columns if "__" in c]

    snapshot = snap[cols + feature_cols].copy()
    snapshot.rename(columns={"y": "actual_label"}, inplace=True)

    snapshot.to_csv(OUT_DIR / "fact_patient_snapshot.csv", index=False)
    print("✅ fact_patient_snapshot.csv saved")


# =========================
# Main
# =========================
def main():
    print("Loading data...")
    df_long = pd.read_parquet(LONG_PATH)
    df_windows = pd.read_parquet(WINDOWS_PATH)

    export_dim_patient(df_long)
    export_fact_vitals(df_long)
    export_fact_predictions(df_windows)
    export_feature_importance()
    export_patient_snapshot(df_windows)

    print("\n🎉 Power BI export complete!")
    print("📁 Output folder:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
