from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class Metrics:
    auroc: float
    auprc: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "threshold": self.threshold,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return Metrics(auroc=auroc, auprc=auprc, threshold=float(threshold), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
