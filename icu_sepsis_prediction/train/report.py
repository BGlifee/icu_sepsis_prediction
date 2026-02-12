from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_confusion_from_metrics(metrics_path: Path, out_path: Path) -> None:
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]

    # simple 2x2 heatmap-like plot without seaborn
    mat = [[tn, fp], [fn, tp]]

    fig, ax = plt.subplots()
    ax.imshow(mat)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title("Confusion Matrix")

    # annotate
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i][j]), ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(fi_path: Path, out_path: Path, top_k: int = 20) -> None:
    fi = pd.read_csv(fi_path)
    fi = fi.sort_values("gain", ascending=False).head(top_k)
    # plot as horizontal bar (matplotlib default color)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi["feature"][::-1], fi["gain"][::-1])
    ax.set_title(f"Top {top_k} Feature Importance (gain)")
    ax.set_xlabel("gain")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_path", type=str, default="artifacts/metrics_xgb.json")
    ap.add_argument("--fi_path", type=str, default="artifacts/feature_importance.csv")
    ap.add_argument("--out_dir", type=str, default="artifacts/plots")
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_from_metrics(Path(args.metrics_path), out_dir / "confusion_matrix.png")
    plot_feature_importance(Path(args.fi_path), out_dir / "feature_importance_topk.png", top_k=args.top_k)

    print("✅ Saved plots to:", out_dir.resolve())


if __name__ == "__main__":
    main()
