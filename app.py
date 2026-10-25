<<<<<<< HEAD
"""Stock ranking model: data, features, backtest and the current ranking.

Run from the repo root:   streamlit run app.py
Needs data/universe.csv (build it with src/build_universe.py or src/recommend.py).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from features_v2 import build_panel_v2
from recommend import add_target, build_panel, make_model

st.set_page_config(page_title="Stock ranking model", layout="wide")

DATA_PATH = ROOT / "data" / "universe.csv"
ORIGINAL = "Original (16 features)"
EXTENDED = "Extended (price, risk, volume, rank features)"

BLUE, AMBER, GRAY, RED = "#1f4e79", "#b5651d", "#8a94a0", "#a23b3b"
STRATEGY_COLORS = alt.Scale(domain=["Model", "Momentum", "All stocks"], range=[BLUE, AMBER, GRAY])

# --------------------------------------------------------------------------------------
# Feature descriptions
# --------------------------------------------------------------------------------------
FEATURE_INFO = {
    # original feature set (src/features.py)
    "daily_return": ("Returns", "Today's close versus yesterday's close."),
    "return_5d": ("Returns", "Price change over the last 5 trading days."),
    "return_20d": ("Returns", "Price change over the last 20 trading days."),
    "return_lag_1": ("Returns", "Daily return one day ago."),
    "return_lag_2": ("Returns", "Daily return two days ago."),
    "return_lag_3": ("Returns", "Daily return three days ago."),
    "volatility_5d": ("Risk", "Standard deviation of daily returns over 5 days."),
    "volatility_20d": ("Risk", "Standard deviation of daily returns over 20 days."),
    "sma_ratio_5d": ("Trend", "Close divided by its 5-day average, minus 1."),
    "sma_ratio_20d": ("Trend", "Close divided by its 20-day average, minus 1."),
    "rsi_14": ("Oscillators", "Relative Strength Index, 14 days (0 to 100). High means recent gains dominate."),
    "macd": ("Oscillators", "12-day minus 26-day exponential average, divided by price."),
    "macd_hist": ("Oscillators", "MACD minus its own 9-day average, divided by price."),
    "bollinger_pctb": ("Trend", "Position inside the 20-day Bollinger band (0 = lower edge, 1 = upper edge)."),
    "volume_change": ("Volume", "Volume versus the previous day, as a fractional change."),
    "volume_ratio_20d": ("Volume", "Volume divided by its 20-day average."),
    # extended feature set (src/features_v2.py)
    "ret_1d": ("Returns", "Daily return."),
    "ret_5d": ("Returns", "Price change over 5 trading days."),
    "ret_20d": ("Returns", "Price change over 20 trading days."),
    "ret_60d": ("Returns", "Price change over 60 trading days."),
    "mom_120_5": ("Returns", "Return from 120 to 5 days ago (skips the most recent week, which tends to reverse)."),
    "mom_250_20": ("Returns", "Return from 250 to 20 days ago (the classic 12-1 month momentum)."),
    "vol_20d": ("Risk", "Standard deviation of daily returns over 20 days."),
    "vol_60d": ("Risk", "Standard deviation of daily returns over 60 days."),
    "vol_ratio": ("Risk", "20-day volatility divided by 60-day volatility."),
    "downside_vol_60d": ("Risk", "Standard deviation of negative daily returns over 60 days."),
    "skew_60d": ("Risk", "Skewness of daily returns over 60 days."),
    "max_dd_60d": ("Risk", "Close relative to its 60-day high (0 = at the high, -0.2 = 20% below)."),
    "beta_60d": ("Risk", "Sensitivity to the equal-weight market over 60 days."),
    "idio_vol_60d": ("Risk", "Volatility left after removing the market's effect, 60 days."),
    "dist_52w_high": ("Trend", "Close relative to its 52-week high."),
    "dist_52w_low": ("Trend", "Close relative to its 52-week low."),
    "sma_ratio_50d": ("Trend", "Close divided by its 50-day average, minus 1."),
    "sma_50_200": ("Trend", "50-day average divided by 200-day average, minus 1."),
    "volume_trend": ("Volume", "5-day average volume divided by 60-day average volume."),
    "log_dollar_vol": ("Volume", "Log of 20-day average traded value (a size and liquidity proxy)."),
    "amihud_20d": ("Volume", "Average absolute return per unit of traded value (illiquidity)."),
    "pv_corr_20d": ("Volume", "20-day correlation between returns and volume changes."),
    "ret_x_volratio": ("Volume", "Daily return multiplied by the volume ratio (big move on heavy volume)."),
    "atr_14_pct": ("Intraday (OHLC)", "14-day average true range as a share of price."),
    "overnight_ret": ("Intraday (OHLC)", "Today's open versus yesterday's close."),
    "intraday_ret": ("Intraday (OHLC)", "Today's close versus today's open."),
    "overnight_20d": ("Intraday (OHLC)", "20-day average overnight return."),
    "intraday_20d": ("Intraday (OHLC)", "20-day average intraday return."),
    "range_pct": ("Intraday (OHLC)", "High minus low, as a share of the close."),
    "close_loc": ("Intraday (OHLC)", "Where the close sits between the day's low and high, 5-day average."),
    "mkt_ret_5d": ("Market context", "5-day return of the equal-weight universe."),
    "mkt_ret_20d": ("Market context", "20-day return of the equal-weight universe."),
    "mkt_vol_20d": ("Market context", "20-day volatility of the equal-weight universe."),
    "mkt_breadth_50d": ("Market context", "Share of stocks trading above their 50-day average."),
    "xs_dispersion_20d": ("Market context", "Spread of 20-day returns across stocks."),
    "ret_20d_vs_mkt": ("Relative strength", "Stock's 20-day return minus the market's."),
    "ret_20d_vs_sector": ("Relative strength", "Stock's 20-day return minus its sector average."),
    "sector_ret_20d": ("Relative strength", "Average 20-day return of the stock's sector."),
    "vix": ("Market context", "India VIX level."),
    "vix_chg_5d": ("Market context", "5-day change in India VIX."),
    "vix_z_250": ("Market context", "India VIX relative to its 1-year range."),
    "nifty_ret_20d": ("Market context", "20-day Nifty 50 return."),
    "nifty_vs_200dma": ("Market context", "Nifty 50 relative to its 200-day average."),
}


def describe(name):
    if name in FEATURE_INFO:
        return FEATURE_INFO[name]
    if name.endswith("_rk"):
        base = name[:-3]
        return ("Cross-sectional rank",
                f"Percentile rank of {base} among all stocks on that date (0 = lowest, 1 = highest).")
    return ("Other", "")


# --------------------------------------------------------------------------------------
# Data and model pipeline (cached)
# --------------------------------------------------------------------------------------
def data_stamp():
    return DATA_PATH.stat().st_mtime if DATA_PATH.exists() else None


@st.cache_resource(show_spinner="Reading prices")
def read_prices(stamp):
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df.sort_index()


@st.cache_resource(show_spinner="Building features (the first run can take a minute)")
def get_panel(stamp, feature_set, horizon):
    long = read_prices(stamp)
    panel = build_panel_v2(long, horizon) if feature_set == EXTENDED else build_panel(long, horizon)
    cols = [c for c in panel.columns if c not in ("fwd_return", "ticker")]
    panel[cols] = panel[cols].astype("float32")
    labeled = add_target(panel.dropna(subset=cols))
    return panel, labeled, cols


def momentum_column(feature_set):
    return "ret_20d" if feature_set == EXTENDED else "return_20d"


def permutation_importance(model, X, y, repeats=3, max_rows=40000, seed=0):
    """Drop in test AUC when one feature is shuffled. Computed on held-out rows only."""
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    drops = {}
    for c in X.columns:
        vals = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            vals.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        drops[c] = float(np.mean(vals))
    return base, pd.Series(drops)


@st.cache_resource(show_spinner="Training on the early period and scoring the later one")
def get_trained(stamp, feature_set, horizon, test_fraction):
    _, labeled, cols = get_panel(stamp, feature_set, horizon)
    dates = np.sort(labeled.index.unique())
    cut = int(len(dates) * (1 - test_fraction))
    train = labeled[labeled.index.isin(dates[: cut - horizon])]  # gap: labels never overlap the test period
    test = labeled[labeled.index.isin(dates[cut:])].copy()

    model = make_model().fit(train[cols], train["target"])
    test["score"] = model.predict_proba(test[cols])[:, 1].astype("float32")

    gain = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(cols).fillna(0.0)
    gain = 100 * gain / gain.sum() if gain.sum() > 0 else gain
    auc_all = roc_auc_score(test["target"], test["score"])
    _, perm = permutation_importance(model, test[cols], test["target"])

    t = test[["score", "fwd_return"]].copy()
    t["excess"] = t["fwd_return"] - t.groupby(level=0)["fwd_return"].transform("mean")
    t["decile"] = np.ceil(t.groupby(level=0)["score"].rank(pct=True) * 10).clip(1, 10).astype(int)
    deciles = t.groupby("decile")["excess"].mean()

    return {
        "test": test, "gain": gain, "perm": perm, "auc": auc_all, "deciles": deciles,
        "train_range": (train.index.min(), train.index.max()),
        "test_range": (test.index.min(), test.index.max()),
        "n_train": len(train), "n_test": len(test),
    }


@st.cache_resource(show_spinner="Fitting on all history and scoring the latest day")
def get_latest(stamp, feature_set, horizon):
    panel, labeled, cols = get_panel(stamp, feature_set, horizon)
    model = make_model().fit(labeled[cols], labeled["target"])
    latest_date = panel.index.max()
    latest = panel[panel.index == latest_date].dropna(subset=cols).copy()
    latest["score"] = model.predict_proba(latest[cols])[:, 1]
    return latest_date, latest


def period_results(test, horizon, top_k, mom_col):
    """Every `horizon` days: top-K by model score, top-K by 20-day return, and all stocks."""
    groups = dict(tuple(test.groupby(level=0)))
    rows = []
    for d in np.sort(test.index.unique())[::horizon]:
        day = groups[d]
        if len(day) < 2 * top_k:
            continue
        top = day.nlargest(top_k, "score")
        mom = day.nlargest(top_k, mom_col)
        ic = spearmanr(day["score"], day["fwd_return"])[0] if day["score"].nunique() > 1 else np.nan
        rows.append({
            "date": d, "model": top["fwd_return"].mean(), "momentum": mom["fwd_return"].mean(),
            "all": day["fwd_return"].mean(), "precision": top["target"].mean(), "ic": ic,
        })
    return pd.DataFrame(rows).set_index("date")


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


# --------------------------------------------------------------------------------------
# Small display helpers (work across Streamlit versions)
# --------------------------------------------------------------------------------------
def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def show_table(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def pct(x, signed=True):
    return f"{x:+.2%}" if signed else f"{x:.2%}"


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.sidebar.title("Stock ranking model")
section = st.sidebar.radio("Section", ["Data", "Features", "Backtest", "Recommendations", "Stock explorer"])
st.sidebar.divider()
st.sidebar.subheader("Settings")

stamp = data_stamp()
if stamp is None:
    st.error("data/universe.csv was not found. Build it first with src/build_universe.py "
             "(or run src/recommend.py once), then reload this page.")
    st.stop()

prices = read_prices(stamp)
n_stocks = prices["ticker"].nunique()

feature_set = st.sidebar.radio("Feature set", [ORIGINAL, EXTENDED], index=1)
horizon = st.sidebar.selectbox("Holding period (trading days)", [5, 10, 20], index=0)
top_k = st.sidebar.slider("Stocks to pick", 3, max(3, min(20, n_stocks // 2)), min(5, max(3, n_stocks // 2)))
test_fraction = st.sidebar.slider("Share of history held out for testing", 0.1, 0.5, 0.2, 0.05)
cost = st.sidebar.number_input("Trading cost per period (%)", 0.0, 2.0, 0.30, 0.05) / 100
st.sidebar.caption("Educational project. Nothing here is investment advice.")

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
if section == "Data":
    st.header("Data")
    n_dates = prices.index.nunique()
    st.write(
        f"{n_stocks} stocks, {n_dates:,} trading days from {prices.index.min().date()} to "
        f"{prices.index.max().date()}, {len(prices):,} rows. "
        f"Columns: {', '.join(c for c in prices.columns if c != 'ticker')}."
    )

    coverage = prices.groupby(level=0)["ticker"].nunique().rename("stocks with data").reset_index()
    st.subheader("Stocks with data on each date")
    show_chart(
        alt.Chart(coverage).mark_line(color=BLUE).encode(
            x=alt.X("Date:T", title=None), y=alt.Y("stocks with data:Q", title="Stocks"),
        ).properties(height=220)
    )
    st.caption("A rising line means newer listings join the universe over time. Only stocks present on a "
               "date can be ranked that day.")

    st.subheader("First rows of the price file")
    show_table(prices.reset_index().head(10))

    st.subheader("How the model is set up")
    st.markdown(
        f"""
- **Label.** For each stock and date, the label is 1 if the stock's return over the next {horizon} trading
  days beat the median stock on that date, otherwise 0. The median is taken across all stocks on that date.
- **Features.** Numbers computed from past prices and volume only. See the Features section.
- **Model.** One XGBoost classifier trained on all stocks together. Its output is a score between 0 and 1.
- **Picking.** Each period, rank stocks by score and take the top {top_k}.
- **Testing.** Train on earlier dates and test on later ones. Training stops {horizon} trading days before
  the test period begins, so no training label reaches into the test period.
"""
    )

# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
elif section == "Features":
    st.header("Features")
    trained = get_trained(stamp, feature_set, horizon, test_fraction)
    gain, perm = trained["gain"], trained["perm"]

    st.write(
        f"{len(gain)} features. Importance is measured two ways, because they answer different questions. "
        f"Test AUC of the model is {trained['auc']:.3f} (0.500 is a coin flip)."
    )
    st.markdown(
        "- **Gain importance** shows how much the trees relied on a feature while fitting the training period.\n"
        "- **Test importance** shows how much AUC drops on the held-out period when that feature is shuffled. "
        "Values near zero mean the model does not use the feature in a way that helps on unseen data."
    )

    table = pd.DataFrame({
        "Feature": gain.index,
        "Group": [describe(c)[0] for c in gain.index],
        "Gain importance (%)": gain.values,
        "Test importance (AUC drop)": perm.reindex(gain.index).values,
        "What it measures": [describe(c)[1] for c in gain.index],
    })

    left, right = st.columns(2)
    top_gain = table.nlargest(15, "Gain importance (%)")
    top_perm = table.nlargest(15, "Test importance (AUC drop)")
    with left:
        st.subheader("Largest gain importance")
        show_chart(
            alt.Chart(top_gain).mark_bar(color=BLUE).encode(
                x=alt.X("Gain importance (%):Q", title="Share of total gain (%)"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature", "Group", alt.Tooltip("Gain importance (%):Q", format=".2f")],
            ).properties(height=380)
        )
    with right:
        st.subheader("Largest test importance")
        show_chart(
            alt.Chart(top_perm).mark_bar(color=AMBER).encode(
                x=alt.X("Test importance (AUC drop):Q", title="AUC drop when shuffled"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature", "Group", alt.Tooltip("Test importance (AUC drop):Q", format=".4f")],
            ).properties(height=380)
        )

    st.subheader("All features")
    sort_by = st.selectbox("Sort by", ["Gain importance (%)", "Test importance (AUC drop)", "Group", "Feature"])
    ascending = sort_by in ("Group", "Feature")
    table = table.sort_values(sort_by, ascending=ascending)
    show_table(
        table,
        column_config={
            "Gain importance (%)": st.column_config.ProgressColumn(
                "Gain importance (%)", format="%.2f", min_value=0.0,
                max_value=float(max(table["Gain importance (%)"].max(), 1e-9)),
            ),
            "Test importance (AUC drop)": st.column_config.NumberColumn(
                "Test importance (AUC drop)", format="%.5f"),
            "What it measures": st.column_config.TextColumn("What it measures", width="large"),
        },
    )
    st.caption(
        "A feature with high gain importance and near-zero test importance was probably used to fit noise "
        "in the training period. Negative test importance means shuffling the feature slightly helped, "
        "which is also a sign the feature carries no stable signal."
    )

# --------------------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------------------
elif section == "Backtest":
    st.header("Backtest")
    trained = get_trained(stamp, feature_set, horizon, test_fraction)
    res = period_results(trained["test"], horizon, top_k, momentum_column(feature_set))
    if res.empty:
        st.warning("No test dates have enough stocks for this setting. Lower the number of stocks to pick.")
        st.stop()

    tr0, tr1 = trained["train_range"]
    te0, te1 = trained["test_range"]
    st.write(
        f"Trained on {tr0.date()} to {tr1.date()} ({trained['n_train']:,} rows). Tested on {te0.date()} to "
        f"{te1.date()}, which the model never saw. Every {horizon} trading days the top {top_k} stocks were "
        f"picked, giving {len(res)} non-overlapping periods."
    )

    rows = []
    for name, col in [("Model (top by score)", "model"), ("Momentum (top by 20-day return)", "momentum"),
                      ("All stocks (average)", "all")]:
        r = res[col]
        row = {"Strategy": name, "Average return per period": pct(r.mean())}
        if col != "all":
            ex = r - res["all"]
            exn = ex - cost
            row.update({
                "Excess over all stocks": pct(ex.mean()),
                "t-stat": f"{tstat(ex):+.2f}",
                "Excess after costs": pct(exn.mean()),
                "t-stat after costs": f"{tstat(exn):+.2f}",
                "Periods beating all stocks": f"{(ex > 0).mean():.0%}",
            })
        rows.append(row)
    show_table(pd.DataFrame(rows).fillna(""))

    ex = res["model"] - res["all"]
    t = tstat(ex)
    if abs(t) < 2:
        word = "beat" if ex.mean() >= 0 else "trailed"
        verdict = (f"The model's picks {word} the all-stock average by {pct(abs(ex.mean()), signed=False)} per "
                   f"period, but with t = {t:+.2f} that is within what luck alone can produce. "
                   "A common bar is a t-stat above 2.")
    elif t >= 2:
        verdict = (f"The model's picks beat the all-stock average by {pct(ex.mean())} per period (t = {t:+.2f}). "
                   "Check survivorship bias, liquidity and costs before reading this as a real edge.")
    else:
        verdict = (f"The model's picks trailed the all-stock average by {pct(abs(ex.mean()), signed=False)} "
                   f"per period (t = {t:+.2f}).")
    st.write(verdict)
    st.write(
        f"Precision of the model's picks: {res['precision'].mean():.1%} (50% is random). "
        f"Average rank correlation between score and actual return (IC): {res['ic'].mean():+.4f} "
        f"(0 means no ranking skill)."
    )

    st.subheader("Growth of 1 unit")
    growth = pd.DataFrame({
        "Model": (1 + res["model"] - cost).cumprod(),
        "Momentum": (1 + res["momentum"] - cost).cumprod(),
        "All stocks": (1 + res["all"]).cumprod(),
    }).reset_index().melt("date", var_name="Strategy", value_name="Value")
    show_chart(
        alt.Chart(growth).mark_line().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("Value:Q", title="Value of 1 unit", scale=alt.Scale(zero=False)),
            color=alt.Color("Strategy:N", scale=STRATEGY_COLORS, legend=alt.Legend(orient="top", title=None)),
        ).properties(height=320)
    )
    st.caption(f"Model and momentum lines subtract a {cost:.2%} cost every period. The all-stocks line is a "
               f"benchmark with no costs. Periods are {horizon} trading days long and do not overlap.")

    left, right = st.columns(2)
    with left:
        st.subheader("Return by score decile")
        dec = trained["deciles"].rename("excess").reset_index()
        show_chart(
            alt.Chart(dec).mark_bar(color=BLUE).encode(
                x=alt.X("decile:O", title="Score decile (10 = highest scored)"),
                y=alt.Y("excess:Q", title=f"Average {horizon}-day return vs. all stocks", axis=alt.Axis(format="%")),
                tooltip=["decile", alt.Tooltip("excess:Q", format=".3%")],
            ).properties(height=300)
        )
        st.caption("If the model ranked well, bars would rise from left to right. A flat or jagged "
                   "pattern means the scores do not separate winners from losers.")
    with right:
        st.subheader("Model minus all stocks, by period")
        per = (res["model"] - res["all"]).rename("excess").reset_index()
        show_chart(
            alt.Chart(per).mark_bar().encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("excess:Q", title="Excess return", axis=alt.Axis(format="%")),
                color=alt.condition(alt.datum.excess > 0, alt.value(BLUE), alt.value(RED)),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("excess:Q", format=".2%")],
            ).properties(height=300)
        )

    with st.expander("Limits of this test"):
        st.markdown(
            "- **Survivorship bias.** The stock list is today's index membership. Companies that fell out or "
            "were delisted are missing, which flatters every strategy, momentum most of all.\n"
            "- **Few periods.** Even a few hundred periods is a small sample. One lucky month can move the average.\n"
            "- **Costs are approximate.** A flat cost per period ignores spreads, price impact and taxes, which "
            "are larger for small or thinly traded stocks.\n"
            "- **Fills.** Picks are assumed to trade at the same close used to compute the features. In "
            "practice you see that close and trade afterwards."
        )

# --------------------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------------------
elif section == "Recommendations":
    st.header("Recommendations")
    latest_date, latest = get_latest(stamp, feature_set, horizon)
    st.write(
        f"Stocks ranked on {latest_date.date()} by the model's probability of beating the median stock over "
        f"the next {horizon} trading days. The model was fit on all labeled history. See Backtest for how "
        "well this kind of ranking has worked on data it had not seen."
    )

    show_cols = [c for c in ["return_20d", "ret_20d", "ret_60d", "rsi_14", "volatility_20d", "vol_20d",
                             "dist_52w_high"] if c in latest.columns]
    top = latest.nlargest(top_k, "score")[["ticker", "score"] + show_cols].rename(
        columns={"ticker": "Stock", "score": "Probability"})
    top.insert(0, "Rank", range(1, len(top) + 1))
    show_table(
        top,
        column_config={"Probability": st.column_config.NumberColumn("Probability", format="%.3f")},
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Top 20 scores")
        chart_df = latest.nlargest(20, "score")[["ticker", "score"]]
        bars = alt.Chart(chart_df).mark_bar(color=BLUE).encode(
            x=alt.X("score:Q", title="Probability of beating the median", scale=alt.Scale(domain=[0.4, 0.7])),
            y=alt.Y("ticker:N", sort="-x", title=None),
            tooltip=["ticker", alt.Tooltip("score:Q", format=".3f")],
        )
        rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color=GRAY, strokeDash=[4, 3]).encode(x="x:Q")
        show_chart((bars + rule).properties(height=420))
        st.caption("The dashed line is 0.5, the score of a coin flip.")
    with right:
        st.subheader("Spread of scores across all stocks")
        hist = alt.Chart(latest[["score"]]).mark_bar(color=GRAY).encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Probability of beating the median"),
            y=alt.Y("count()", title="Stocks"),
        ).properties(height=420)
        show_chart(hist)
        st.caption(f"Scores range from {latest['score'].min():.2f} to {latest['score'].max():.2f}. A narrow band "
                   "around 0.5 means the model sees little difference between most stocks.")

# --------------------------------------------------------------------------------------
# Stock explorer
# --------------------------------------------------------------------------------------
else:
    st.header("Stock explorer")
    panel, _, cols = get_panel(stamp, feature_set, horizon)
    tickers = sorted(prices["ticker"].unique())
    ticker = st.selectbox("Stock", tickers, index=tickers.index("TCS.NS") if "TCS.NS" in tickers else 0)
    window = st.radio("Window", ["6 months", "1 year", "3 years", "All"], index=1, horizontal=True)

    g = prices[prices["ticker"] == ticker]
    end = g.index.max()
    start = {"6 months": end - pd.DateOffset(months=6), "1 year": end - pd.DateOffset(years=1),
             "3 years": end - pd.DateOffset(years=3), "All": g.index.min()}[window]
    gw = g[g.index >= start]

    st.subheader("Closing price")
    show_chart(
        alt.Chart(gw["close"].reset_index()).mark_line(color=BLUE).encode(
            x=alt.X("Date:T", title=None), y=alt.Y("close:Q", title="Close", scale=alt.Scale(zero=False)),
        ).properties(height=260)
    )

    st.subheader("A feature over time, against the median stock")
    feat = st.selectbox("Feature", cols, index=cols.index(momentum_column(feature_set)))
    sub = panel.loc[panel.index >= start, [feat, "ticker"]]
    median = sub.groupby(level=0)[feat].median().rename("Median stock")
    own = sub[sub["ticker"] == ticker][feat].rename(ticker)
    lines = pd.concat([own, median], axis=1).reset_index().melt("Date", var_name="Series", value_name="Value")
    show_chart(
        alt.Chart(lines.dropna()).mark_line().encode(
            x=alt.X("Date:T", title=None), y=alt.Y("Value:Q", title=feat, scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(domain=[ticker, "Median stock"], range=[BLUE, GRAY]),
                            legend=alt.Legend(orient="top", title=None)),
        ).properties(height=260)
    )
    st.caption(describe(feat)[1])

    st.subheader("Latest values compared with other stocks")
    last_date = panel.index.max()
    latest = panel[panel.index == last_date]
    row = latest[latest["ticker"] == ticker]
    if row.empty:
        st.info("This stock has no feature values on the latest date, usually because it has too little history.")
    else:
        ranks = latest[cols].rank(pct=True)
        mine = ranks[latest["ticker"].values == ticker].iloc[0]
        values = row[cols].iloc[0]
        out = pd.DataFrame({
            "Feature": cols,
            "Group": [describe(c)[0] for c in cols],
            "Value": values.values.astype(float),
            "Percentile among stocks": mine.reindex(cols).values.astype(float),
            "What it measures": [describe(c)[1] for c in cols],
        })
        st.caption(f"As of {last_date.date()}. A percentile of 0.90 means 90% of stocks have a lower value.")
        show_table(
            out,
            column_config={
                "Value": st.column_config.NumberColumn("Value", format="%.4f"),
                "Percentile among stocks": st.column_config.ProgressColumn(
                    "Percentile among stocks", format="%.2f", min_value=0.0, max_value=1.0),
                "What it measures": st.column_config.TextColumn("What it measures", width="large"),
            },
        )
=======
"""Stock ranking model: data, features, backtest and the current ranking.

Run from the repo root:   streamlit run app.py
Needs data/universe.csv (build it with src/build_universe.py or src/recommend.py).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from features_v2 import build_panel_v2
from recommend import add_target, build_panel, make_model

st.set_page_config(page_title="Stock ranking model", layout="wide")

DATA_PATH = ROOT / "data" / "universe.csv"
ORIGINAL = "Original (16 features)"
EXTENDED = "Extended (price, risk, volume, rank features)"

BLUE, AMBER, GRAY, RED = "#1f4e79", "#b5651d", "#8a94a0", "#a23b3b"
STRATEGY_COLORS = alt.Scale(domain=["Model", "Momentum", "All stocks"], range=[BLUE, AMBER, GRAY])

# --------------------------------------------------------------------------------------
# Feature descriptions
# --------------------------------------------------------------------------------------
FEATURE_INFO = {
    # original feature set (src/features.py)
    "daily_return": ("Returns", "Today's close versus yesterday's close."),
    "return_5d": ("Returns", "Price change over the last 5 trading days."),
    "return_20d": ("Returns", "Price change over the last 20 trading days."),
    "return_lag_1": ("Returns", "Daily return one day ago."),
    "return_lag_2": ("Returns", "Daily return two days ago."),
    "return_lag_3": ("Returns", "Daily return three days ago."),
    "volatility_5d": ("Risk", "Standard deviation of daily returns over 5 days."),
    "volatility_20d": ("Risk", "Standard deviation of daily returns over 20 days."),
    "sma_ratio_5d": ("Trend", "Close divided by its 5-day average, minus 1."),
    "sma_ratio_20d": ("Trend", "Close divided by its 20-day average, minus 1."),
    "rsi_14": ("Oscillators", "Relative Strength Index, 14 days (0 to 100). High means recent gains dominate."),
    "macd": ("Oscillators", "12-day minus 26-day exponential average, divided by price."),
    "macd_hist": ("Oscillators", "MACD minus its own 9-day average, divided by price."),
    "bollinger_pctb": ("Trend", "Position inside the 20-day Bollinger band (0 = lower edge, 1 = upper edge)."),
    "volume_change": ("Volume", "Volume versus the previous day, as a fractional change."),
    "volume_ratio_20d": ("Volume", "Volume divided by its 20-day average."),
    # extended feature set (src/features_v2.py)
    "ret_1d": ("Returns", "Daily return."),
    "ret_5d": ("Returns", "Price change over 5 trading days."),
    "ret_20d": ("Returns", "Price change over 20 trading days."),
    "ret_60d": ("Returns", "Price change over 60 trading days."),
    "mom_120_5": ("Returns", "Return from 120 to 5 days ago (skips the most recent week, which tends to reverse)."),
    "mom_250_20": ("Returns", "Return from 250 to 20 days ago (the classic 12-1 month momentum)."),
    "vol_20d": ("Risk", "Standard deviation of daily returns over 20 days."),
    "vol_60d": ("Risk", "Standard deviation of daily returns over 60 days."),
    "vol_ratio": ("Risk", "20-day volatility divided by 60-day volatility."),
    "downside_vol_60d": ("Risk", "Standard deviation of negative daily returns over 60 days."),
    "skew_60d": ("Risk", "Skewness of daily returns over 60 days."),
    "max_dd_60d": ("Risk", "Close relative to its 60-day high (0 = at the high, -0.2 = 20% below)."),
    "beta_60d": ("Risk", "Sensitivity to the equal-weight market over 60 days."),
    "idio_vol_60d": ("Risk", "Volatility left after removing the market's effect, 60 days."),
    "dist_52w_high": ("Trend", "Close relative to its 52-week high."),
    "dist_52w_low": ("Trend", "Close relative to its 52-week low."),
    "sma_ratio_50d": ("Trend", "Close divided by its 50-day average, minus 1."),
    "sma_50_200": ("Trend", "50-day average divided by 200-day average, minus 1."),
    "volume_trend": ("Volume", "5-day average volume divided by 60-day average volume."),
    "log_dollar_vol": ("Volume", "Log of 20-day average traded value (a size and liquidity proxy)."),
    "amihud_20d": ("Volume", "Average absolute return per unit of traded value (illiquidity)."),
    "pv_corr_20d": ("Volume", "20-day correlation between returns and volume changes."),
    "ret_x_volratio": ("Volume", "Daily return multiplied by the volume ratio (big move on heavy volume)."),
    "atr_14_pct": ("Intraday (OHLC)", "14-day average true range as a share of price."),
    "overnight_ret": ("Intraday (OHLC)", "Today's open versus yesterday's close."),
    "intraday_ret": ("Intraday (OHLC)", "Today's close versus today's open."),
    "overnight_20d": ("Intraday (OHLC)", "20-day average overnight return."),
    "intraday_20d": ("Intraday (OHLC)", "20-day average intraday return."),
    "range_pct": ("Intraday (OHLC)", "High minus low, as a share of the close."),
    "close_loc": ("Intraday (OHLC)", "Where the close sits between the day's low and high, 5-day average."),
    "mkt_ret_5d": ("Market context", "5-day return of the equal-weight universe."),
    "mkt_ret_20d": ("Market context", "20-day return of the equal-weight universe."),
    "mkt_vol_20d": ("Market context", "20-day volatility of the equal-weight universe."),
    "mkt_breadth_50d": ("Market context", "Share of stocks trading above their 50-day average."),
    "xs_dispersion_20d": ("Market context", "Spread of 20-day returns across stocks."),
    "ret_20d_vs_mkt": ("Relative strength", "Stock's 20-day return minus the market's."),
    "ret_20d_vs_sector": ("Relative strength", "Stock's 20-day return minus its sector average."),
    "sector_ret_20d": ("Relative strength", "Average 20-day return of the stock's sector."),
    "vix": ("Market context", "India VIX level."),
    "vix_chg_5d": ("Market context", "5-day change in India VIX."),
    "vix_z_250": ("Market context", "India VIX relative to its 1-year range."),
    "nifty_ret_20d": ("Market context", "20-day Nifty 50 return."),
    "nifty_vs_200dma": ("Market context", "Nifty 50 relative to its 200-day average."),
}


def describe(name):
    if name in FEATURE_INFO:
        return FEATURE_INFO[name]
    if name.endswith("_rk"):
        base = name[:-3]
        return ("Cross-sectional rank",
                f"Percentile rank of {base} among all stocks on that date (0 = lowest, 1 = highest).")
    return ("Other", "")


# --------------------------------------------------------------------------------------
# Data and model pipeline (cached)
# --------------------------------------------------------------------------------------
def data_stamp():
    return DATA_PATH.stat().st_mtime if DATA_PATH.exists() else None


@st.cache_resource(show_spinner="Reading prices")
def read_prices(stamp):
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df.sort_index()


@st.cache_resource(show_spinner="Building features (the first run can take a minute)")
def get_panel(stamp, feature_set, horizon):
    long = read_prices(stamp)
    panel = build_panel_v2(long, horizon) if feature_set == EXTENDED else build_panel(long, horizon)
    cols = [c for c in panel.columns if c not in ("fwd_return", "ticker")]
    panel[cols] = panel[cols].astype("float32")
    labeled = add_target(panel.dropna(subset=cols))
    return panel, labeled, cols


def momentum_column(feature_set):
    return "ret_20d" if feature_set == EXTENDED else "return_20d"


def permutation_importance(model, X, y, repeats=3, max_rows=40000, seed=0):
    """Drop in test AUC when one feature is shuffled. Computed on held-out rows only."""
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    drops = {}
    for c in X.columns:
        vals = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            vals.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        drops[c] = float(np.mean(vals))
    return base, pd.Series(drops)


@st.cache_resource(show_spinner="Training on the early period and scoring the later one")
def get_trained(stamp, feature_set, horizon, test_fraction):
    _, labeled, cols = get_panel(stamp, feature_set, horizon)
    dates = np.sort(labeled.index.unique())
    cut = int(len(dates) * (1 - test_fraction))
    train = labeled[labeled.index.isin(dates[: cut - horizon])]  # gap: labels never overlap the test period
    test = labeled[labeled.index.isin(dates[cut:])].copy()

    model = make_model().fit(train[cols], train["target"])
    test["score"] = model.predict_proba(test[cols])[:, 1].astype("float32")

    gain = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(cols).fillna(0.0)
    gain = 100 * gain / gain.sum() if gain.sum() > 0 else gain
    auc_all = roc_auc_score(test["target"], test["score"])
    _, perm = permutation_importance(model, test[cols], test["target"])

    t = test[["score", "fwd_return"]].copy()
    t["excess"] = t["fwd_return"] - t.groupby(level=0)["fwd_return"].transform("mean")
    t["decile"] = np.ceil(t.groupby(level=0)["score"].rank(pct=True) * 10).clip(1, 10).astype(int)
    deciles = t.groupby("decile")["excess"].mean()

    return {
        "test": test, "gain": gain, "perm": perm, "auc": auc_all, "deciles": deciles,
        "train_range": (train.index.min(), train.index.max()),
        "test_range": (test.index.min(), test.index.max()),
        "n_train": len(train), "n_test": len(test),
    }


@st.cache_resource(show_spinner="Fitting on all history and scoring the latest day")
def get_latest(stamp, feature_set, horizon):
    panel, labeled, cols = get_panel(stamp, feature_set, horizon)
    model = make_model().fit(labeled[cols], labeled["target"])
    latest_date = panel.index.max()
    latest = panel[panel.index == latest_date].dropna(subset=cols).copy()
    latest["score"] = model.predict_proba(latest[cols])[:, 1]
    return latest_date, latest


def period_results(test, horizon, top_k, mom_col):
    """Every `horizon` days: top-K by model score, top-K by 20-day return, and all stocks."""
    groups = dict(tuple(test.groupby(level=0)))
    rows = []
    for d in np.sort(test.index.unique())[::horizon]:
        day = groups[d]
        if len(day) < 2 * top_k:
            continue
        top = day.nlargest(top_k, "score")
        mom = day.nlargest(top_k, mom_col)
        ic = spearmanr(day["score"], day["fwd_return"])[0] if day["score"].nunique() > 1 else np.nan
        rows.append({
            "date": d, "model": top["fwd_return"].mean(), "momentum": mom["fwd_return"].mean(),
            "all": day["fwd_return"].mean(), "precision": top["target"].mean(), "ic": ic,
        })
    return pd.DataFrame(rows).set_index("date")


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


# --------------------------------------------------------------------------------------
# Small display helpers (work across Streamlit versions)
# --------------------------------------------------------------------------------------
def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def show_table(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def pct(x, signed=True):
    return f"{x:+.2%}" if signed else f"{x:.2%}"


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.sidebar.title("Stock ranking model")
section = st.sidebar.radio("Section", ["Data", "Features", "Backtest", "Recommendations", "Stock explorer"])
st.sidebar.divider()
st.sidebar.subheader("Settings")

stamp = data_stamp()
if stamp is None:
    st.error("data/universe.csv was not found. Build it first with src/build_universe.py "
             "(or run src/recommend.py once), then reload this page.")
    st.stop()

prices = read_prices(stamp)
n_stocks = prices["ticker"].nunique()

feature_set = st.sidebar.radio("Feature set", [ORIGINAL, EXTENDED], index=1)
horizon = st.sidebar.selectbox("Holding period (trading days)", [5, 10, 20], index=0)
top_k = st.sidebar.slider("Stocks to pick", 3, max(3, min(20, n_stocks // 2)), min(5, max(3, n_stocks // 2)))
test_fraction = st.sidebar.slider("Share of history held out for testing", 0.1, 0.5, 0.2, 0.05)
cost = st.sidebar.number_input("Trading cost per period (%)", 0.0, 2.0, 0.30, 0.05) / 100
st.sidebar.caption("Educational project. Nothing here is investment advice.")

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
if section == "Data":
    st.header("Data")
    n_dates = prices.index.nunique()
    st.write(
        f"{n_stocks} stocks, {n_dates:,} trading days from {prices.index.min().date()} to "
        f"{prices.index.max().date()}, {len(prices):,} rows. "
        f"Columns: {', '.join(c for c in prices.columns if c != 'ticker')}."
    )

    coverage = prices.groupby(level=0)["ticker"].nunique().rename("stocks with data").reset_index()
    st.subheader("Stocks with data on each date")
    show_chart(
        alt.Chart(coverage).mark_line(color=BLUE).encode(
            x=alt.X("Date:T", title=None), y=alt.Y("stocks with data:Q", title="Stocks"),
        ).properties(height=220)
    )
    st.caption("A rising line means newer listings join the universe over time. Only stocks present on a "
               "date can be ranked that day.")

    st.subheader("First rows of the price file")
    show_table(prices.reset_index().head(10))

    st.subheader("How the model is set up")
    st.markdown(
        f"""
- **Label.** For each stock and date, the label is 1 if the stock's return over the next {horizon} trading
  days beat the median stock on that date, otherwise 0. The median is taken across all stocks on that date.
- **Features.** Numbers computed from past prices and volume only. See the Features section.
- **Model.** One XGBoost classifier trained on all stocks together. Its output is a score between 0 and 1.
- **Picking.** Each period, rank stocks by score and take the top {top_k}.
- **Testing.** Train on earlier dates and test on later ones. Training stops {horizon} trading days before
  the test period begins, so no training label reaches into the test period.
"""
    )

# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
elif section == "Features":
    st.header("Features")
    trained = get_trained(stamp, feature_set, horizon, test_fraction)
    gain, perm = trained["gain"], trained["perm"]

    st.write(
        f"{len(gain)} features. Importance is measured two ways, because they answer different questions. "
        f"Test AUC of the model is {trained['auc']:.3f} (0.500 is a coin flip)."
    )
    st.markdown(
        "- **Gain importance** shows how much the trees relied on a feature while fitting the training period.\n"
        "- **Test importance** shows how much AUC drops on the held-out period when that feature is shuffled. "
        "Values near zero mean the model does not use the feature in a way that helps on unseen data."
    )

    table = pd.DataFrame({
        "Feature": gain.index,
        "Group": [describe(c)[0] for c in gain.index],
        "Gain importance (%)": gain.values,
        "Test importance (AUC drop)": perm.reindex(gain.index).values,
        "What it measures": [describe(c)[1] for c in gain.index],
    })

    left, right = st.columns(2)
    top_gain = table.nlargest(15, "Gain importance (%)")
    top_perm = table.nlargest(15, "Test importance (AUC drop)")
    with left:
        st.subheader("Largest gain importance")
        show_chart(
            alt.Chart(top_gain).mark_bar(color=BLUE).encode(
                x=alt.X("Gain importance (%):Q", title="Share of total gain (%)"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature", "Group", alt.Tooltip("Gain importance (%):Q", format=".2f")],
            ).properties(height=380)
        )
    with right:
        st.subheader("Largest test importance")
        show_chart(
            alt.Chart(top_perm).mark_bar(color=AMBER).encode(
                x=alt.X("Test importance (AUC drop):Q", title="AUC drop when shuffled"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature", "Group", alt.Tooltip("Test importance (AUC drop):Q", format=".4f")],
            ).properties(height=380)
        )

    st.subheader("All features")
    sort_by = st.selectbox("Sort by", ["Gain importance (%)", "Test importance (AUC drop)", "Group", "Feature"])
    ascending = sort_by in ("Group", "Feature")
    table = table.sort_values(sort_by, ascending=ascending)
    show_table(
        table,
        column_config={
            "Gain importance (%)": st.column_config.ProgressColumn(
                "Gain importance (%)", format="%.2f", min_value=0.0,
                max_value=float(max(table["Gain importance (%)"].max(), 1e-9)),
            ),
            "Test importance (AUC drop)": st.column_config.NumberColumn(
                "Test importance (AUC drop)", format="%.5f"),
            "What it measures": st.column_config.TextColumn("What it measures", width="large"),
        },
    )
    st.caption(
        "A feature with high gain importance and near-zero test importance was probably used to fit noise "
        "in the training period. Negative test importance means shuffling the feature slightly helped, "
        "which is also a sign the feature carries no stable signal."
    )

# --------------------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------------------
elif section == "Backtest":
    st.header("Backtest")
    trained = get_trained(stamp, feature_set, horizon, test_fraction)
    res = period_results(trained["test"], horizon, top_k, momentum_column(feature_set))
    if res.empty:
        st.warning("No test dates have enough stocks for this setting. Lower the number of stocks to pick.")
        st.stop()

    tr0, tr1 = trained["train_range"]
    te0, te1 = trained["test_range"]
    st.write(
        f"Trained on {tr0.date()} to {tr1.date()} ({trained['n_train']:,} rows). Tested on {te0.date()} to "
        f"{te1.date()}, which the model never saw. Every {horizon} trading days the top {top_k} stocks were "
        f"picked, giving {len(res)} non-overlapping periods."
    )

    rows = []
    for name, col in [("Model (top by score)", "model"), ("Momentum (top by 20-day return)", "momentum"),
                      ("All stocks (average)", "all")]:
        r = res[col]
        row = {"Strategy": name, "Average return per period": pct(r.mean())}
        if col != "all":
            ex = r - res["all"]
            exn = ex - cost
            row.update({
                "Excess over all stocks": pct(ex.mean()),
                "t-stat": f"{tstat(ex):+.2f}",
                "Excess after costs": pct(exn.mean()),
                "t-stat after costs": f"{tstat(exn):+.2f}",
                "Periods beating all stocks": f"{(ex > 0).mean():.0%}",
            })
        rows.append(row)
    show_table(pd.DataFrame(rows).fillna(""))

    ex = res["model"] - res["all"]
    t = tstat(ex)
    if abs(t) < 2:
        word = "beat" if ex.mean() >= 0 else "trailed"
        verdict = (f"The model's picks {word} the all-stock average by {pct(abs(ex.mean()), signed=False)} per "
                   f"period, but with t = {t:+.2f} that is within what luck alone can produce. "
                   "A common bar is a t-stat above 2.")
    elif t >= 2:
        verdict = (f"The model's picks beat the all-stock average by {pct(ex.mean())} per period (t = {t:+.2f}). "
                   "Check survivorship bias, liquidity and costs before reading this as a real edge.")
    else:
        verdict = (f"The model's picks trailed the all-stock average by {pct(abs(ex.mean()), signed=False)} "
                   f"per period (t = {t:+.2f}).")
    st.write(verdict)
    st.write(
        f"Precision of the model's picks: {res['precision'].mean():.1%} (50% is random). "
        f"Average rank correlation between score and actual return (IC): {res['ic'].mean():+.4f} "
        f"(0 means no ranking skill)."
    )

    st.subheader("Growth of 1 unit")
    growth = pd.DataFrame({
        "Model": (1 + res["model"] - cost).cumprod(),
        "Momentum": (1 + res["momentum"] - cost).cumprod(),
        "All stocks": (1 + res["all"]).cumprod(),
    }).reset_index().melt("date", var_name="Strategy", value_name="Value")
    show_chart(
        alt.Chart(growth).mark_line().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("Value:Q", title="Value of 1 unit", scale=alt.Scale(zero=False)),
            color=alt.Color("Strategy:N", scale=STRATEGY_COLORS, legend=alt.Legend(orient="top", title=None)),
        ).properties(height=320)
    )
    st.caption(f"Model and momentum lines subtract a {cost:.2%} cost every period. The all-stocks line is a "
               f"benchmark with no costs. Periods are {horizon} trading days long and do not overlap.")

    left, right = st.columns(2)
    with left:
        st.subheader("Return by score decile")
        dec = trained["deciles"].rename("excess").reset_index()
        show_chart(
            alt.Chart(dec).mark_bar(color=BLUE).encode(
                x=alt.X("decile:O", title="Score decile (10 = highest scored)"),
                y=alt.Y("excess:Q", title=f"Average {horizon}-day return vs. all stocks", axis=alt.Axis(format="%")),
                tooltip=["decile", alt.Tooltip("excess:Q", format=".3%")],
            ).properties(height=300)
        )
        st.caption("If the model ranked well, bars would rise from left to right. A flat or jagged "
                   "pattern means the scores do not separate winners from losers.")
    with right:
        st.subheader("Model minus all stocks, by period")
        per = (res["model"] - res["all"]).rename("excess").reset_index()
        show_chart(
            alt.Chart(per).mark_bar().encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("excess:Q", title="Excess return", axis=alt.Axis(format="%")),
                color=alt.condition(alt.datum.excess > 0, alt.value(BLUE), alt.value(RED)),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("excess:Q", format=".2%")],
            ).properties(height=300)
        )

    with st.expander("Limits of this test"):
        st.markdown(
            "- **Survivorship bias.** The stock list is today's index membership. Companies that fell out or "
            "were delisted are missing, which flatters every strategy, momentum most of all.\n"
            "- **Few periods.** Even a few hundred periods is a small sample. One lucky month can move the average.\n"
            "- **Costs are approximate.** A flat cost per period ignores spreads, price impact and taxes, which "
            "are larger for small or thinly traded stocks.\n"
            "- **Fills.** Picks are assumed to trade at the same close used to compute the features. In "
            "practice you see that close and trade afterwards."
        )

# --------------------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------------------
elif section == "Recommendations":
    st.header("Recommendations")
    latest_date, latest = get_latest(stamp, feature_set, horizon)
    st.write(
        f"Stocks ranked on {latest_date.date()} by the model's probability of beating the median stock over "
        f"the next {horizon} trading days. The model was fit on all labeled history. See Backtest for how "
        "well this kind of ranking has worked on data it had not seen."
    )

    show_cols = [c for c in ["return_20d", "ret_20d", "ret_60d", "rsi_14", "volatility_20d", "vol_20d",
                             "dist_52w_high"] if c in latest.columns]
    top = latest.nlargest(top_k, "score")[["ticker", "score"] + show_cols].rename(
        columns={"ticker": "Stock", "score": "Probability"})
    top.insert(0, "Rank", range(1, len(top) + 1))
    show_table(
        top,
        column_config={"Probability": st.column_config.NumberColumn("Probability", format="%.3f")},
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Top 20 scores")
        chart_df = latest.nlargest(20, "score")[["ticker", "score"]]
        bars = alt.Chart(chart_df).mark_bar(color=BLUE).encode(
            x=alt.X("score:Q", title="Probability of beating the median", scale=alt.Scale(domain=[0.4, 0.7])),
            y=alt.Y("ticker:N", sort="-x", title=None),
            tooltip=["ticker", alt.Tooltip("score:Q", format=".3f")],
        )
        rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color=GRAY, strokeDash=[4, 3]).encode(x="x:Q")
        show_chart((bars + rule).properties(height=420))
        st.caption("The dashed line is 0.5, the score of a coin flip.")
    with right:
        st.subheader("Spread of scores across all stocks")
        hist = alt.Chart(latest[["score"]]).mark_bar(color=GRAY).encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Probability of beating the median"),
            y=alt.Y("count()", title="Stocks"),
        ).properties(height=420)
        show_chart(hist)
        st.caption(f"Scores range from {latest['score'].min():.2f} to {latest['score'].max():.2f}. A narrow band "
                   "around 0.5 means the model sees little difference between most stocks.")

# --------------------------------------------------------------------------------------
# Stock explorer
# --------------------------------------------------------------------------------------
else:
    st.header("Stock explorer")
    panel, _, cols = get_panel(stamp, feature_set, horizon)
    tickers = sorted(prices["ticker"].unique())
    ticker = st.selectbox("Stock", tickers, index=tickers.index("TCS.NS") if "TCS.NS" in tickers else 0)
    window = st.radio("Window", ["6 months", "1 year", "3 years", "All"], index=1, horizontal=True)

    g = prices[prices["ticker"] == ticker]
    end = g.index.max()
    start = {"6 months": end - pd.DateOffset(months=6), "1 year": end - pd.DateOffset(years=1),
             "3 years": end - pd.DateOffset(years=3), "All": g.index.min()}[window]
    gw = g[g.index >= start]

    st.subheader("Closing price")
    show_chart(
        alt.Chart(gw["close"].reset_index()).mark_line(color=BLUE).encode(
            x=alt.X("Date:T", title=None), y=alt.Y("close:Q", title="Close", scale=alt.Scale(zero=False)),
        ).properties(height=260)
    )

    st.subheader("A feature over time, against the median stock")
    feat = st.selectbox("Feature", cols, index=cols.index(momentum_column(feature_set)))
    sub = panel.loc[panel.index >= start, [feat, "ticker"]]
    median = sub.groupby(level=0)[feat].median().rename("Median stock")
    own = sub[sub["ticker"] == ticker][feat].rename(ticker)
    lines = pd.concat([own, median], axis=1).reset_index().melt("Date", var_name="Series", value_name="Value")
    show_chart(
        alt.Chart(lines.dropna()).mark_line().encode(
            x=alt.X("Date:T", title=None), y=alt.Y("Value:Q", title=feat, scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(domain=[ticker, "Median stock"], range=[BLUE, GRAY]),
                            legend=alt.Legend(orient="top", title=None)),
        ).properties(height=260)
    )
    st.caption(describe(feat)[1])

    st.subheader("Latest values compared with other stocks")
    last_date = panel.index.max()
    latest = panel[panel.index == last_date]
    row = latest[latest["ticker"] == ticker]
    if row.empty:
        st.info("This stock has no feature values on the latest date, usually because it has too little history.")
    else:
        ranks = latest[cols].rank(pct=True)
        mine = ranks[latest["ticker"].values == ticker].iloc[0]
        values = row[cols].iloc[0]
        out = pd.DataFrame({
            "Feature": cols,
            "Group": [describe(c)[0] for c in cols],
            "Value": values.values.astype(float),
            "Percentile among stocks": mine.reindex(cols).values.astype(float),
            "What it measures": [describe(c)[1] for c in cols],
        })
        st.caption(f"As of {last_date.date()}. A percentile of 0.90 means 90% of stocks have a lower value.")
        show_table(
            out,
            column_config={
                "Value": st.column_config.NumberColumn("Value", format="%.4f"),
                "Percentile among stocks": st.column_config.ProgressColumn(
                    "Percentile among stocks", format="%.2f", min_value=0.0, max_value=1.0),
                "What it measures": st.column_config.TextColumn("What it measures", width="large"),
            },
        )
>>>>>>> 913fd7dbb9bb7f0997bcaaaca687cd76b868b2bc
