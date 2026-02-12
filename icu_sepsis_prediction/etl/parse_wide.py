from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def infer_hour(df: pd.DataFrame) -> pd.Series:
    """
    PhysioNet 2019 style:
    - if ICULOS exists, use it (typically 1..)
    - else use row index + 1
    """
    if "ICULOS" in df.columns:
        return (
             pd.to_numeric(df["ICULOS"], errors="coerce")
                .ffill()
                .fillna(0)
                .astype(int)
)

    return pd.Series(range(1, len(df) + 1), index=df.index, dtype=int)


def read_patient_file(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, sep="|")
    df.insert(0, "patient_id", fp.stem)
    df.insert(1, "hour", infer_hour(df))
    # make sure SepsisLabel exists (if missing, set all 0)
    if "SepsisLabel" not in df.columns:
        df["SepsisLabel"] = 0
    # coerce label to int 0/1
    df["SepsisLabel"] = pd.to_numeric(df["SepsisLabel"], errors="coerce").fillna(0).astype(int)
    return df


def parse_all(input_dir: Path) -> pd.DataFrame:
    files = sorted(list(input_dir.glob("*.psv")))
    if not files:
        raise FileNotFoundError(
            f"No .psv files found in: {input_dir}\n"
            f"Expected PhysioNet 2019 style patient files like p000001.psv"
        )

    dfs: list[pd.DataFrame] = []
    for fp in tqdm(files, desc="Reading patient .psv files"):
        dfs.append(read_patient_file(fp))

    out = pd.concat(dfs, ignore_index=True)
    # Ensure minimal columns
    if "hour" not in out.columns:
        out["hour"] = out.groupby("patient_id").cumcount() + 1

    # Sort
    out = out.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Folder containing patient .psv files")
    ap.add_argument(
        "--out_path",
        type=str,
        default="data/processed/long.parquet",
        help="Output path (.parquet recommended; falls back to csv if you change extension)",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = parse_all(input_dir)

    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print("\n✅ Saved long table:", out_path)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns)[:20], "..." if len(df.columns) > 20 else "")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
