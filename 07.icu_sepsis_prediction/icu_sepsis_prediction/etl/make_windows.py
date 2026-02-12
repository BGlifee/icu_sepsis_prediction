from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


EXCLUDE_COLS = {"patient_id", "hour", "SepsisLabel", "ICULOS"}


def load_long(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def window_features(win: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """
    Compute per-feature aggregates within the observation window.
    Returns flat dict: <col>__last, __mean, __std, __min, __max
    """
    feats: dict[str, float] = {}
    for c in feature_cols:
        s = pd.to_numeric(win[c], errors="coerce")
        # last non-null
        last = s.dropna().iloc[-1] if s.notna().any() else np.nan
        feats[f"{c}__last"] = float(last) if pd.notna(last) else np.nan
        feats[f"{c}__mean"] = float(s.mean()) if s.notna().any() else np.nan
        feats[f"{c}__std"] = float(s.std(ddof=0)) if s.notna().any() else np.nan
        feats[f"{c}__min"] = float(s.min()) if s.notna().any() else np.nan
        feats[f"{c}__max"] = float(s.max()) if s.notna().any() else np.nan
    return feats


def build_windows(
    df_long: pd.DataFrame,
    W: int = 24,
    H: int = 6,
    step: int = 1,
) -> pd.DataFrame:
    """
    For each patient and time t:
      - Observation window: [t-W+1 ... t]
      - Label: y(t)=1 if any SepsisLabel==1 in (t+1 ... t+H)
    Features: last/mean/std/min/max for each variable inside observation window.

    IMPORTANT: No leakage (label uses only future SepsisLabel).
    """
    # Ensure columns
    if "patient_id" not in df_long.columns or "hour" not in df_long.columns:
        raise ValueError("df_long must contain patient_id and hour columns.")
    if "SepsisLabel" not in df_long.columns:
        df_long = df_long.copy()
        df_long["SepsisLabel"] = 0

    df_long = df_long.sort_values(["patient_id", "hour"]).reset_index(drop=True)

    # Feature columns: numeric vitals/labs
    feature_cols = [c for c in df_long.columns if c not in EXCLUDE_COLS]

    rows = []
    for pid, g in tqdm(df_long.groupby("patient_id", sort=False), desc="Building windows"):
        g = g.sort_values("hour")
        hours = g["hour"].values
        max_t = int(np.nanmax(hours)) if len(hours) else 0

        # create a mapping hour -> row index (assume hourly; if missing hours, we still slice by row positions)
        # We'll use row-based indexing but require at least W rows before t.
        y_series = pd.to_numeric(g["SepsisLabel"], errors="coerce").fillna(0).astype(int).values

        for end_idx in range(W - 1, len(g) - H, step):
            t_hour = int(g.iloc[end_idx]["hour"])

            obs = g.iloc[end_idx - W + 1 : end_idx + 1]
            fut = y_series[end_idx + 1 : end_idx + 1 + H]
            y = int(np.any(fut == 1))

            feats = window_features(obs, feature_cols)
            rows.append(
                {
                    "patient_id": pid,
                    "t_hour": t_hour,
                    "y": y,
                    **feats,
                }
            )

    out = pd.DataFrame(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_path", type=str, default="data/processed/long.parquet", help="Long table path")
    ap.add_argument("--W", type=int, default=24, help="Observation window hours")
    ap.add_argument("--H", type=int, default=6, help="Prediction horizon hours")
    ap.add_argument("--step", type=int, default=1, help="Step size in hours")
    ap.add_argument("--out_path", type=str, default="data/processed/windows.parquet", help="Output windows path")
    ap.add_argument("--meta_path", type=str, default="data/processed/metadata.json", help="Metadata JSON path")
    args = ap.parse_args()

    long_path = Path(args.long_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    meta_path = Path(args.meta_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_long = load_long(long_path)
    windows = build_windows(df_long, W=args.W, H=args.H, step=args.step)

    if out_path.suffix.lower() == ".parquet":
        windows.to_parquet(out_path, index=False)
    else:
        windows.to_csv(out_path, index=False)

    meta = {
        "W": args.W,
        "H": args.H,
        "step": args.step,
        "n_windows": int(len(windows)),
        "n_patients": int(windows["patient_id"].nunique()) if len(windows) else 0,
        "n_features": int(windows.shape[1] - 3) if len(windows) else 0,  # minus patient_id, t_hour, y
        "feature_columns": [c for c in windows.columns if c not in ("patient_id", "t_hour", "y")],
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n✅ Saved windows:", out_path)
    print("Shape:", windows.shape)
    print("Saved metadata:", meta_path)
    print(windows.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
