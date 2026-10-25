"""Richer features for the cross-sectional stock ranking model.

Works with close + volume only; if the data also has open/high/low columns it adds
ATR, gap and intraday features. Use it in recommend.py with:

    from features_v2 import build_panel_v2
    panel = build_panel_v2(long, args.horizon)
"""

from pathlib import Path

import numpy as np
import pandas as pd

USE_MACRO = False   # set True to add India VIX / Nifty features (needs src/macro.py and internet)

# --- Sector lookup -----------------------------------------------------------------------
# Fallback map for the original 48 stocks; replaced by data/nifty_500.csv if that file exists.
_SECTORS = {
    "bank_fin": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "BAJFINANCE",
                 "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "SHRIRAMFIN"],
    "it": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "PERSISTENT"],
    "energy": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "COALINDIA", "TATAPOWER", "ADANIGREEN"],
    "auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO"],
    "consumer": ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "TATACONSUM", "ASIANPAINT",
                 "TITAN", "TRENT"],
    "pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "APOLLOHOSP", "DIVISLAB"],
    "infra": ["LT", "ULTRACEMCO", "GRASIM", "ADANIPORTS", "BEL", "JSWSTEEL", "TATASTEEL"],
}
TICKER_TO_SECTOR = {f"{t}.NS": s for s, ts in _SECTORS.items() for t in ts}

_nifty_csv = Path(__file__).resolve().parent.parent / "data" / "nifty_500.csv"
if _nifty_csv.exists():
    _n = pd.read_csv(_nifty_csv)
    TICKER_TO_SECTOR = dict(zip(_n["Symbol"].str.strip() + ".NS", _n["Industry"]))


def _stock_features(g: pd.DataFrame) -> pd.DataFrame:
    """Per-stock time-series features. Every value uses data up to and including that day only."""
    close, vol = g["close"], g["volume"].replace(0, np.nan)
    ret = close.pct_change()
    f = pd.DataFrame(index=g.index)

    # Momentum at several horizons; "skip" versions drop the most recent days (they tend to reverse)
    f["ret_1d"] = ret
    f["ret_5d"] = close.pct_change(5)
    f["ret_20d"] = close.pct_change(20)
    f["ret_60d"] = close.pct_change(60)
    f["mom_120_5"] = close.shift(5) / close.shift(120) - 1
    f["mom_250_20"] = close.shift(20) / close.shift(250) - 1   # classic 12-1 momentum

    # Risk
    f["vol_20d"] = ret.rolling(20).std()
    f["vol_60d"] = ret.rolling(60).std()
    f["vol_ratio"] = f["vol_20d"] / f["vol_60d"]
    f["downside_vol_60d"] = ret.where(ret < 0).rolling(60, min_periods=20).std()
    f["skew_60d"] = ret.rolling(60).skew()
    f["max_dd_60d"] = close / close.rolling(60).max() - 1

    # Position in range
    f["dist_52w_high"] = close / close.rolling(250, min_periods=120).max() - 1
    f["dist_52w_low"] = close / close.rolling(250, min_periods=120).min() - 1
    f["sma_ratio_20d"] = close / close.rolling(20).mean() - 1
    f["sma_ratio_50d"] = close / close.rolling(50).mean() - 1
    f["sma_50_200"] = close.rolling(50).mean() / close.rolling(200, min_periods=150).mean() - 1

    # Volume / liquidity
    dollar_vol = close * vol
    f["volume_ratio_20d"] = vol / vol.rolling(20).mean()
    f["volume_trend"] = vol.rolling(5).mean() / vol.rolling(60).mean()
    f["log_dollar_vol"] = np.log(dollar_vol.rolling(20).mean())
    f["amihud_20d"] = (ret.abs() / dollar_vol).rolling(20).mean() * 1e9
    f["pv_corr_20d"] = ret.rolling(20).corr(vol.pct_change())
    f["ret_x_volratio"] = ret * f["volume_ratio_20d"]   # big move on heavy volume

    # OHLC features (only if the data has open/high/low)
    if {"open", "high", "low"}.issubset(g.columns):
        prev_close = close.shift(1)
        tr = pd.concat([g["high"] - g["low"],
                        (g["high"] - prev_close).abs(),
                        (g["low"] - prev_close).abs()], axis=1).max(axis=1)
        f["atr_14_pct"] = tr.rolling(14).mean() / close
        f["overnight_ret"] = g["open"] / prev_close - 1
        f["intraday_ret"] = close / g["open"] - 1
        f["overnight_20d"] = f["overnight_ret"].rolling(20).mean()
        f["intraday_20d"] = f["intraday_ret"].rolling(20).mean()
        f["range_pct"] = (g["high"] - g["low"]) / close
        f["close_loc"] = ((close - g["low"]) / (g["high"] - g["low"]).replace(0, np.nan)).rolling(5).mean()
    return f


def build_panel_v2(long: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """One row per (date, ticker): time-series, market-relative, cross-sectional-rank features,
    plus `fwd_return` (the label material; never use it as a feature)."""
    long = long.sort_index()
    close = long.pivot_table(index=long.index, columns="ticker", values="close")
    rets = close.pct_change()

    # Market (equal-weight universe) context shared by all stocks on a date
    mkt_ret = rets.mean(axis=1)
    market = pd.DataFrame(index=close.index)
    market["mkt_ret_5d"] = (1 + mkt_ret).rolling(5).apply(np.prod, raw=True) - 1
    market["mkt_ret_20d"] = (1 + mkt_ret).rolling(20).apply(np.prod, raw=True) - 1
    market["mkt_vol_20d"] = mkt_ret.rolling(20).std()
    market["mkt_breadth_50d"] = (close > close.rolling(50).mean()).mean(axis=1)
    market["xs_dispersion_20d"] = close.pct_change(20).std(axis=1)

    # Rolling beta and idiosyncratic volatility vs the equal-weight market
    beta = rets.rolling(60).cov(mkt_ret).div(mkt_ret.rolling(60).var(), axis=0)
    resid = rets.sub(beta.shift(1).mul(mkt_ret, axis=0))
    idio_vol = resid.rolling(60).std()

    parts = []
    for ticker, g in long.groupby("ticker"):
        g = g.sort_index()
        f = _stock_features(g)
        f["beta_60d"] = beta[ticker].reindex(f.index)
        f["idio_vol_60d"] = idio_vol[ticker].reindex(f.index)
        f["fwd_return"] = g["close"].shift(-horizon) / g["close"] - 1
        f["ticker"] = ticker
        parts.append(f)
    panel = pd.concat(parts)
    panel.index.name = "Date"
    panel = panel.join(market)

    if USE_MACRO:
        try:
            from macro import load_macro
        except ModuleNotFoundError:
            from src.macro import load_macro
        panel = panel.join(load_macro())

    panel["ret_20d_vs_mkt"] = panel["ret_20d"] - panel["mkt_ret_20d"]

    # Sector-relative momentum: how is the stock doing vs. its own sector peers?
    sector = panel["ticker"].map(TICKER_TO_SECTOR).fillna("other")
    sector_mean = panel.groupby([panel.index, sector])["ret_20d"].transform("mean")
    panel["ret_20d_vs_sector"] = panel["ret_20d"] - sector_mean
    panel["sector_ret_20d"] = sector_mean

    # Cross-sectional percentile ranks, per date (what a ranking model actually needs)
    rank_cols = ["ret_5d", "ret_20d", "ret_60d", "mom_120_5", "mom_250_20", "vol_60d",
                 "downside_vol_60d", "dist_52w_high", "volume_ratio_20d", "volume_trend",
                 "log_dollar_vol", "amihud_20d", "beta_60d", "idio_vol_60d", "sma_ratio_50d",
                 "max_dd_60d"]
    ranks = panel.groupby(level=0)[rank_cols].rank(pct=True).add_suffix("_rk")
    panel = pd.concat([panel, ranks], axis=1)

    return panel.replace([np.inf, -np.inf], np.nan).sort_index()