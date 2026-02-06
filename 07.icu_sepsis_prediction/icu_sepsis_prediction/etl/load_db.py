from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras


def connect():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="labor_db",
        user="admin",
        password="admin",
    )


def run_sql_file(conn, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def load_windows(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def upsert_windows(conn, df: pd.DataFrame, dry_run: bool = False) -> None:
    required = {"patient_id", "t_hour", "y"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"windows must contain columns: {required}")

    feature_cols = [c for c in df.columns if c not in ("patient_id", "t_hour", "y")]
    rows = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        features = {c: (None if pd.isna(d[c]) else float(d[c])) for c in feature_cols}
        rows.append((str(d["patient_id"]), int(d["t_hour"]), int(d["y"]), json.dumps(features)))

    if dry_run:
        print(f"[dry_run] would upsert {len(rows)} rows into icu.windows")
        return

    q = """
    INSERT INTO icu.windows (patient_id, t_hour, y, features)
    VALUES %s
    ON CONFLICT (patient_id, t_hour) DO UPDATE
      SET y = EXCLUDED.y,
          features = EXCLUDED.features;
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, q, rows, page_size=2000)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_path", type=str, default="data/processed/windows.parquet")
    ap.add_argument("--schema_sql", type=str, default="sql/schema.sql")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    windows_path = Path(args.windows_path).expanduser().resolve()
    schema_sql = Path(args.schema_sql).expanduser().resolve()

    df = load_windows(windows_path)
    print("Loaded windows:", df.shape)

    conn = connect()
    run_sql_file(conn, schema_sql)
    upsert_windows(conn, df, dry_run=args.dry_run)
    conn.close()

    print("✅ DB load done.")


if __name__ == "__main__":
    main()
