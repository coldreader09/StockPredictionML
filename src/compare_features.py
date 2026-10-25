"""Compare feature sets with walk-forward validation (run from the repo root:
python src/compare_features.py).

For each feature set it retrains on an expanding window, tests on the next unseen block,
and every `horizon` days compares the model's top-K picks with all stocks and with a simple
20-day momentum rule. Reports the excess return, a t-statistic (|t| > 2 is the usual bar for
'probably not luck') and the rank IC (correlation between model score and actual returns).
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBClassifier

from features_v2 import build_panel_v2
from recommend import add_target, build_panel, load_universe

K = 5            # number of stocks picked per period
COST = 0.003     # assumed round-trip trading cost per period (0.3%); set to 0 to ignore costs
N_FOLDS = 4      # number of walk-forward test blocks
HORIZONS = (5, 20)

long = load_universe("data/universe.csv")


def make_model():
    return XGBClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.7,
        colsample_bytree=0.6, min_child_weight=50, reg_lambda=10,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")


def run(panel, cols, H, common_dates, mom_col):
    """Walk-forward test. Returns one row per non-overlapping holding period."""
    lab = add_target(panel.dropna(subset=cols))
    lab = lab[lab.index.isin(common_dates)]
    dates = np.sort(lab.index.unique())
    start = int(len(dates) * 0.5)
    edges = np.linspace(start, len(dates), N_FOLDS + 1).astype(int)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        train = lab[lab.index.isin(dates[: a - H])]          # gap of H days: no label overlap
        test = lab[lab.index.isin(dates[a:b])].copy()
        model = make_model().fit(train[cols], train["target"])
        test["score"] = model.predict_proba(test[cols])[:, 1]
        for d in dates[a:b][::H]:                             # non-overlapping periods
            day = test.loc[[d]]
            if len(day) < 2 * K:
                continue
            rows.append({
                "model": day.nlargest(K, "score")["fwd_return"].mean(),
                "momentum": day.nlargest(K, mom_col)["fwd_return"].mean(),
                "all": day["fwd_return"].mean(),
                "ic": spearmanr(day["score"], day["fwd_return"])[0],
            })
    return pd.DataFrame(rows)


def report(name, r):
    ex = r["model"] - r["all"]
    exn = r["model"] - COST - r["all"]
    print(f"{name:26s} n={len(r):3d}  vs all: {ex.mean():+.3%} (t={tstat(ex):+.2f})  "
          f"after costs: {exn.mean():+.3%} (t={tstat(exn):+.2f})  "
          f"IC={r['ic'].mean():+.4f}  hit={(ex > 0).mean():.0%}")


for H in HORIZONS:
    new = build_panel_v2(long, H)
    old = build_panel(long, H)
    # momentum baseline column for the old panel uses 'return_20d'; copy it onto the new panel
    mom_map = old.reset_index().set_index(["Date", "ticker"])["return_20d"]
    new["mom20"] = mom_map.reindex(pd.MultiIndex.from_arrays([new.index, new["ticker"]])).values

    old_cols = [c for c in old.columns if c not in ("fwd_return", "ticker")]
    new_cols = [c for c in new.columns if c not in ("fwd_return", "ticker", "mom20")]
    common = (new.dropna(subset=new_cols + ["mom20"]).index.unique()
              .intersection(old.dropna(subset=old_cols).index.unique()))

    groups = {
        "old 16 features": (old.assign(mom20=old["return_20d"]), old_cols),
        "momentum + risk (no rank)": (new, [c for c in new_cols if not c.endswith("_rk")
                                            and not c.startswith("mkt_") and c != "xs_dispersion_20d"]),
        "rank features only": (new, [c for c in new_cols if c.endswith("_rk")]),
        "all new features": (new, new_cols),
    }

    print(f"\n=== horizon {H}d | {long['ticker'].nunique()} stocks | {len(common)} common dates | "
          f"cost assumed {COST:.1%} per period ===")
    base = None
    for name, (p, c) in groups.items():
        r = run(p, c, H, common, "mom20")
        report(name, r)
        base = r
    ex = base["momentum"] - base["all"]
    exn = base["momentum"] - COST - base["all"]
    print(f"{'momentum top-%d (baseline)' % K:26s} n={len(base):3d}  vs all: {ex.mean():+.3%} "
          f"(t={tstat(ex):+.2f})  after costs: {exn.mean():+.3%} (t={tstat(exn):+.2f})")