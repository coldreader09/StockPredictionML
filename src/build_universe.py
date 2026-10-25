import pandas as pd

from data_loader import download_many

syms = pd.read_csv("data/nifty_500.csv")
tickers = (syms["Symbol"].str.strip() + ".NS").tolist()


long = download_many(tickers, period="10y")

# keep stocks with at least ~3 years of history
counts = long.groupby("ticker").size()
long = long[long["ticker"].isin(counts[counts >= 750].index)]

long.to_csv("data/universe.csv")
print(long["ticker"].nunique(), "stocks,", long.index.min().date(), "to", long.index.max().date())