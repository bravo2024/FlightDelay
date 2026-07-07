"""train.py - Flight delay prediction training pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import argparse
import json
from src.data import load_flights
from src.model import train_gradient_boosting, evaluate
from src.core import temporal_split, StandardScaler, compute_metrics


def main():
    parser = argparse.ArgumentParser(description="FlightDelay training pipeline")
    parser.add_argument("--csv", type=str, default=None, help="Path to flights CSV")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (overrides --csv if both are set)")
    parser.add_argument("--n", type=int, default=100000, help="Sample size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    csv_path = None if args.synthetic else args.csv
    data = load_flights(csv_path=csv_path, sample_n=args.n, seed=args.seed)
    print(f"Loaded: {data['n_samples']:,} flights, delay rate: {data['delay_rate']:.3f}")

    # Split
    X_train, X_test, y_train, y_test = temporal_split(
        data["X"], data["y"], test_size=0.25
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"Split: train={len(X_train):,}, test={len(X_test):,}")

    # Train
    print("\nTraining Gradient Boosting...")
    gb_result = train_gradient_boosting(X_train_s, y_train, n_estimators=200, seed=args.seed)
    gb_eval = evaluate(gb_result, X_test_s, y_test)

    print(f"\nTest Accuracy: {gb_eval['metrics']['accuracy']:.4f}")
    print(f"Test F1: {gb_eval['metrics']['f1']:.4f}")
    print(f"Test AUC: {gb_eval['metrics']['roc_auc']:.4f}")

    # Save
    Path("models").mkdir(exist_ok=True)
    with open("models/metrics.json", "w") as f:
        json.dump({k: v for k, v in gb_eval["metrics"].items() if k != "confusion_matrix"}, f, indent=2)
    print("\nSaved metrics -> models/metrics.json")


if __name__ == "__main__":
    main()
