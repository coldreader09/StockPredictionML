"""Walk-forward validation: several train/test splits instead of one."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

try:  # running as a script: python src/walk_forward.py
    from data_loader import load_price_data
    from train import prepare_data
except ModuleNotFoundError:  # imported from a notebook
    from src.data_loader import load_price_data
    from src.train import prepare_data

HORIZON = 5  # must match build_target's horizon


def make_model(y_train):
    pos = y_train.sum()
    neg = len(y_train) - pos
    return XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=neg / max(pos, 1),
        eval_metric="logloss", random_state=42,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prices.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=150, help="rows per test fold")
    args = parser.parse_args()

    X, y = prepare_data(load_price_data(args.data))
    # gap=HORIZON drops the rows whose 5-day label reaches into the test period
    tscv = TimeSeriesSplit(n_splits=args.folds, test_size=args.test_size, gap=HORIZON)

    accs, bases, precs = [], [], []
    for i, (tr, te) in enumerate(tscv.split(X), 1):
        model = make_model(y.iloc[tr]).fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        acc = accuracy_score(y.iloc[te], pred)
        base = max(y.iloc[te].mean(), 1 - y.iloc[te].mean())  # always-majority score on this fold
        prec = precision_score(y.iloc[te], pred, zero_division=0)
        accs.append(acc); bases.append(base); precs.append(prec)
        print(f"Fold {i}: train {len(tr):4d} | test {X.index[te[0]].date()} to {X.index[te[-1]].date()} "
              f"| acc {acc:.3f} | majority {base:.3f} | precision {prec:.3f}")

    print(f"\nMean accuracy  : {np.mean(accs):.3f}  (std across folds {np.std(accs):.3f})")
    print(f"Mean majority  : {np.mean(bases):.3f}")
    print(f"Mean precision : {np.mean(precs):.3f}")
    print(f"Test rows used : {args.folds * args.test_size}")


if __name__ == "__main__":
    main()
