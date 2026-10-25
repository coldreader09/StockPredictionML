from pathlib import Path

import pandas as pd
import yfinance as yf
import time


def download_stock_data(ticker="AAPL"):
    df = yf.download(
        ticker,
        start="2011-10-24",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values("Price")

    df.columns = df.columns.str.lower()

    return df.sort_index()


def load_price_data(path="data/prices.csv", ticker="AAPL"):
    """Load prices from a CSV. If the file does not exist, download and save it first."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        download_stock_data(ticker).to_csv(path)

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df.columns = df.columns.str.lower()
    return df.sort_index()

def download_many(tickers, period="10y", pause=0.5, retries=2):
    """Download OHLCV one ticker at a time (slower but much less likely to be blocked)."""
    frames, failed = [], []
    for n, t in enumerate(tickers, 1):
        d = None
        for attempt in range(retries + 1):
            try:
                d = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=True)
                if not d.empty:
                    break
            except Exception:
                pass
            time.sleep(2 * (attempt + 1))
        if d is None or d.empty:
            failed.append(t)
            print(f"[{n}/{len(tickers)}] FAILED {t}")
            continue
        d.columns = d.columns.str.lower()
        d = d[["open", "high", "low", "close", "volume"]].copy()
        d["ticker"] = t
        frames.append(d)
        print(f"[{n}/{len(tickers)}] ok {t}")
        time.sleep(pause)

    if not frames:
        raise RuntimeError(
            "No data downloaded for any ticker. Update yfinance, switch network, "
            "or wait a while and retry."
        )
    if failed:
        print(f"\n{len(failed)} tickers failed: {failed}")

    long = pd.concat(frames)
    long.index = pd.to_datetime(long.index).tz_localize(None).normalize()
    long.index.name = "Date"
    return long.sort_index()

if __name__ == "__main__":
    data = load_price_data("data/prices.csv", "AAPL")
    print(data.head())