"""Reward Model (DIRECTIVE-034)

لماذا نموذج بسيط؟
- الهدف هو نقطة بداية قابلة للتدريب محلياً على بيانات human feedback بدون GPU.
- نستخدم انحداراً خطياً Ridge عبر numpy فقط لتجنب تبعيات ثقيلة.

المدخلات المتوقعة:
- prompt_embedding: list[float]
- variation_embedding: list[float]
- metadata (اختياري)
- target: humanScore مُطبع إلى 0-1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import json
import numpy as np


@dataclass
class FitReport:
    mae: float
    rmse: float
    correlation: float


class RewardModel:
    def __init__(self, l2: float = 1e-2):
        self.l2 = float(l2)
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    @staticmethod
    def _to_matrix(pairs: Iterable[Tuple[List[float], List[float]]]) -> np.ndarray:
        rows = []
        for p_emb, v_emb in pairs:
            x = np.asarray(list(p_emb) + list(v_emb), dtype=np.float32)
            rows.append(x)
        return np.stack(rows, axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Ridge regression closed-form."""
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Add bias via separate b
        n, d = X.shape
        XtX = (X.T @ X) / max(1, n)
        Xty = (X.T @ y) / max(1, n)

        reg = self.l2 * np.eye(d, dtype=np.float32)
        self.w = np.linalg.solve(XtX + reg, Xty)
        # b as mean residual
        self.b = float(y.mean() - (X @ self.w).mean())

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model is not trained")
        scores = X @ self.w + self.b
        return np.clip(scores, 0.0, 1.0)

    def save(self, path: str) -> None:
        if self.w is None:
            raise RuntimeError("Model is not trained")
        np.savez(path, w=self.w, b=np.asarray([self.b], dtype=np.float32), l2=np.asarray([self.l2], dtype=np.float32))

    @classmethod
    def load(cls, path: str) -> "RewardModel":
        data = np.load(path)
        model = cls(l2=float(data["l2"][0]))
        model.w = data["w"].astype(np.float32)
        model.b = float(data["b"][0])
        return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> FitReport:
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    if y_true.size < 2:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.isnan(corr):
            corr = 0.0

    return FitReport(mae=mae, rmse=rmse, correlation=corr)


def load_training_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a minimal training dataset.

    Expected JSON format (list of objects):
    {
      "prompt_embedding": [...],
      "variation_embedding": [...],
      "humanScore": 1-5
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    pairs = [(it["prompt_embedding"], it["variation_embedding"]) for it in items]
    X = RewardModel._to_matrix(pairs)
    y = np.asarray([(float(it["humanScore"]) - 1.0) / 4.0 for it in items], dtype=np.float32)
    return X, y
