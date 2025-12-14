"""Train Reward Model (DIRECTIVE-034)

Usage:
  python3 models/train_reward.py --input data.json --output reward_model.npz

This trains a lightweight Ridge regression reward model using numpy.
"""

from __future__ import annotations

import argparse

from reward_model import RewardModel, compute_metrics, load_training_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSON training data")
    ap.add_argument("--output", required=True, help="Path to write model .npz")
    ap.add_argument("--l2", type=float, default=1e-2)
    args = ap.parse_args()

    X, y = load_training_json(args.input)
    model = RewardModel(l2=args.l2)
    model.fit(X, y)

    preds = model.predict(X)
    report = compute_metrics(y, preds)

    model.save(args.output)

    print(f"trained_on={len(y)}")
    print(f"mae={report.mae:.6f}")
    print(f"rmse={report.rmse:.6f}")
    print(f"correlation={report.correlation:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
