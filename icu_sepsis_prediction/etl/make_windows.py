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


def _safe_float(x) -> float:
    return float(x) if pd.notna(x) else np.nan


def _slope(x: np.ndarray) -> float:
    """Linear slope over time index (0..n-1). Returns nan if <2 points."""
    if x.size < 2:
        return np.nan
    t = np.arange(x.size, dtype=float)
    try:
        m = np.polyfit(t, x.astype(float), 1)[0]
        return float(m)
    except Exception:
        return np.nan


def _recent_mean(s: pd.Series, k: int) -> float:
    s2 = s.dropna()
    if len(s2) == 0:
        return np.nan
    return float(s2.iloc[-k:].mean())


def window_features(
    win: pd.DataFrame,
    feature_cols: list[str],
    feature_set: str = "v1",
) -> dict[str, float]:
    """
    v1: <col>__last, __mean, __std, __min, __max
    v2: v1 + first/delta/range/slope/cv/last3_mean/last6_mean/missing_ratio
        + abnormal duration features for key vitals (HR/MAP/Temp/O2Sat/Resp)
    """
    feats: dict[str, float] = {}

    # thresholds for abnormal duration (count of hours in window)
    abnormal_rules = {
        "HR": [("gt", 100)],
        "MAP": [("lt", 65)],
        "Temp": [("gt", 38)],
        "O2Sat": [("lt", 92)],
        "Resp": [("gt", 22)],
    }

    # ---------- per-column aggregates ----------
    for c in feature_cols:
        s = pd.to_numeric(win[c], errors="coerce")
        has = s.notna().any()

        last = s.dropna().iloc[-1] if has else np.nan
        mean = s.mean() if has else np.nan
        std = s.std(ddof=0) if has else np.nan
        minv = s.min() if has else np.nan
        maxv = s.max() if has else np.nan

        # v1
        feats[f"{c}__last"] = _safe_float(last)
        feats[f"{c}__mean"] = _safe_float(mean)
        feats[f"{c}__std"] = _safe_float(std)
        feats[f"{c}__min"] = _safe_float(minv)
        feats[f"{c}__max"] = _safe_float(maxv)

        if feature_set != "v2":
            continue

        # v2 extras
        first = s.dropna().iloc[0] if has else np.nan
        feats[f"{c}__first"] = _safe_float(first)
        feats[f"{c}__delta"] = _safe_float(last - first) if pd.notna(last) and pd.notna(first) else np.nan
        feats[f"{c}__range"] = _safe_float(maxv - minv) if pd.notna(maxv) and pd.notna(minv) else np.nan

        s_nonnull = s.dropna().to_numpy()
        feats[f"{c}__slope"] = _slope(s_nonnull)
        feats[f"{c}__cv"] = float(std / mean) if pd.notna(std) and pd.notna(mean) and mean != 0 else np.nan

        feats[f"{c}__last3_mean"] = _recent_mean(s, 3)
        feats[f"{c}__last6_mean"] = _recent_mean(s, 6)
        feats[f"{c}__missing_ratio"] = float(s.isna().mean())

    # ---------- abnormal duration (only once) ----------
    if feature_set == "v2":
        for vital, rules in abnormal_rules.items():
            # Prefer exact match (HR), otherwise prefix match (HR_*)
            if vital in win.columns:
                s_v = pd.to_numeric(win[vital], errors="coerce")
            else:
                candidates = [c for c in win.columns if (c == vital or c.startswith(f"{vital}_"))]
                if not candidates:
                    continue
                s_v = pd.to_numeric(win[candidates[0]], errors="coerce")

            for op, thr in rules:
                if op == "gt":
                    feats[f"{vital}__hours_gt_{thr}"] = float((s_v > thr).sum(skipna=True))
                elif op == "lt":
                    feats[f"{vital}__hours_lt_{thr}"] = float((s_v < thr).sum(skipna=True))

    return feats


def build_windows(
    df_long: pd.DataFrame,
    W: int = 24,
    H: int = 6,
    step: int = 1,
    feature_set: str = "v1",
) -> pd.DataFrame:
    """
    For each patient and time t:
      - Observation window: [t-W+1 ... t]
      - Label: y(t)=1 if any SepsisLabel==1 in (t+1 ... t+H)
    IMPORTANT: No leakage (label uses only future SepsisLabel).
    """
    if "patient_id" not in df_long.columns or "hour" not in df_long.columns:
        raise ValueError("df_long must contain patient_id and hour columns.")

    df_long = df_long.copy()

    # Ensure label exists
    if "SepsisLabel" not in df_long.columns:
        df_long["SepsisLabel"] = 0

    # Ensure hour is numeric & sorted
    df_long["hour"] = pd.to_numeric(df_long["hour"], errors="coerce")
    df_long = df_long.dropna(subset=["hour"])
    df_long["hour"] = df_long["hour"].astype(int)

    df_long = df_long.sort_values(["patient_id", "hour"]).reset_index(drop=True)

    # Feature columns: everything except excludes
    candidate_cols = [c for c in df_long.columns if c not in EXCLUDE_COLS]

    # Keep only columns that can become numeric (skip strings that turn all-NaN)
    feature_cols: list[str] = []
    for c in candidate_cols:
        s = pd.to_numeric(df_long[c], errors="coerce")
        if s.notna().any():
            feature_cols.append(c)

    rows = []
    for pid, g in tqdm(df_long.groupby("patient_id", sort=False), desc="Building windows"):
        g = g.sort_values("hour")
        y_series = pd.to_numeric(g["SepsisLabel"], errors="coerce").fillna(0).astype(int).values

        # need at least W rows history and H rows future
        for end_idx in range(W - 1, len(g) - H, step):
            t_hour = int(g.iloc[end_idx]["hour"])

            obs = g.iloc[end_idx - W + 1 : end_idx + 1]
            fut = y_series[end_idx + 1 : end_idx + 1 + H]
            y = int(np.any(fut == 1))

            feats = window_features(obs, feature_cols, feature_set=feature_set)
            rows.append(
                {
                    "patient_id": pid,
                    "t_hour": t_hour,
                    "y": y,
                    **feats,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_path", type=str, default="data/processed/long.parquet", help="Long table path")
    ap.add_argument("--W", type=int, default=24, help="Observation window hours")
    ap.add_argument("--H", type=int, default=6, help="Prediction horizon hours")
    ap.add_argument("--step", type=int, default=1, help="Step size in hours")
    ap.add_argument("--out_path", type=str, default="data/processed/windows.parquet", help="Output windows path")
    ap.add_argument("--meta_path", type=str, default="data/processed/metadata.json", help="Metadata JSON path")
    ap.add_argument(
        "--feature_set",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="v1: last/mean/std/min/max, v2: adds trend/delta/range/recent/abnormal/missingness",
    )
    args = ap.parse_args()

    long_path = Path(args.long_path).expanduser().resolve()

    out_path = Path(args.out_path).expanduser().resolve()
    meta_path = Path(args.meta_path).expanduser().resolve()

    # make outputs feature_set-specific if generic names are used
    if out_path.name == "windows.parquet":
        out_path = out_path.with_name(f"windows_{args.feature_set}.parquet")
    if meta_path.name == "metadata.json":
        meta_path = meta_path.with_name(f"metadata_{args.feature_set}.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    df_long = load_long(long_path)
    windows = build_windows(df_long, W=args.W, H=args.H, step=args.step, feature_set=args.feature_set)

    # Save windows
    if out_path.suffix.lower() == ".parquet":
        windows.to_parquet(out_path, index=False)
    else:
        windows.to_csv(out_path, index=False)

    meta = {
        "feature_set": args.feature_set,
        "long_path": str(long_path),
        "out_path": str(out_path),
        "W": int(args.W),
        "H": int(args.H),
        "step": int(args.step),
        "n_windows": int(len(windows)),
        "n_patients": int(windows["patient_id"].nunique()) if len(windows) else 0,
        "n_features": int(windows.shape[1] - 3) if len(windows) else 0,
        "feature_columns": [c for c in windows.columns if c not in ("patient_id", "t_hour", "y")],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n✅ Saved windows:", out_path)
    print("Shape:", windows.shape)
    print("✅ Saved metadata:", meta_path)
    print(windows.head(3).to_string(index=False))


if __name__ == "__main__":
    main()