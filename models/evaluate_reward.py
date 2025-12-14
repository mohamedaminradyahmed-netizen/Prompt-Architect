"""Evaluate Reward Model (DIRECTIVE-034)

Usage:
  python3 models/evaluate_reward.py --model reward_model.npz --input test.json

Outputs MAE/RMSE/correlation.
"""

from __future__ import annotations

import argparse

from reward_model import RewardModel, compute_metrics, load_training_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model .npz")
    ap.add_argument("--input", required=True, help="Path to JSON eval data")
    args = ap.parse_args()

    X, y = load_training_json(args.input)
    model = RewardModel.load(args.model)
    preds = model.predict(X)

    report = compute_metrics(y, preds)
    print(f"mae={report.mae:.6f}")
    print(f"rmse={report.rmse:.6f}")
    print(f"correlation={report.correlation:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
