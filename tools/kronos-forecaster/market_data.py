"""Daily price history for the test universes, cached on disk so tests do not re-download every run."""
import os
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_cache")
START = "2000-01-01"          # older Yahoo candles often have no real open price, so candle shapes are unreliable

# the 30 tickers the forecaster was first built and judged on
UNIVERSE = {
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "fund": ["SPY", "QQQ", "IWM", "GLD", "TLT"],
    "stock": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM", "WMT", "KO", "JNJ", "PG", "HD",
              "DIS", "BA", "CAT", "NFLX", "AMD", "INTC", "GS", "V"],
}

INDEXES = ["^GSPC", "^DJI", "^IXIC", "^RUT", "^NYA", "^MID"]
SECTOR_FUNDS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "DIA", "MDY", "VTI"]

# Yahoo's continuous front-month futures. These are NOT roll-adjusted: when the front contract expires the
# series jumps to the next contract, so a move across a roll date is partly an artefact.
COMMODITIES = {
    "energy": ["CL=F", "BZ=F", "NG=F", "HO=F", "RB=F"],
    "metals": ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F"],
    "grains": ["ZC=F", "ZW=F", "ZS=F", "ZM=F", "ZL=F", "ZO=F", "ZR=F", "KE=F"],
    "softs": ["KC=F", "SB=F", "CC=F", "CT=F", "OJ=F"],
    "livestock": ["LE=F", "HE=F", "GF=F"],
}
COMMODITY_NAMES = {
    "CL=F": "Crude oil (WTI)", "BZ=F": "Crude oil (Brent)", "NG=F": "Natural gas", "HO=F": "Heating oil", "RB=F": "Gasoline",
    "GC=F": "Gold", "SI=F": "Silver", "HG=F": "Copper", "PL=F": "Platinum", "PA=F": "Palladium",
    "ZC=F": "Corn", "ZW=F": "Wheat", "ZS=F": "Soybeans", "ZM=F": "Soybean meal", "ZL=F": "Soybean oil", "ZO=F": "Oats",
    "ZR=F": "Rough rice", "KE=F": "KC wheat", "KC=F": "Coffee", "SB=F": "Sugar", "CC=F": "Cocoa", "CT=F": "Cotton",
    "OJ=F": "Orange juice", "LE=F": "Live cattle", "HE=F": "Lean hogs", "GF=F": "Feeder cattle",
}


def sp500():
    """Today's S&P 500 members. Companies that failed or were dropped are missing, which flatters the past."""
    path = os.path.join(CACHE, "_sp500_constituents.csv")
    d = pd.read_csv(path)
    return [s.replace(".", "-") for s in d["Symbol"].astype(str).tolist()]


def universe(name):
    """Tickers for a named test universe."""
    if name == "original":
        return [t for v in UNIVERSE.values() for t in v]
    if name == "stocks":
        first = set(UNIVERSE["stock"])
        return [t for t in sp500() if t not in first]          # the ~480 stocks the forecaster has never seen
    if name == "indexes":
        return INDEXES + SECTOR_FUNDS
    if name == "commodities":
        return [t for v in COMMODITIES.values() for t in v]
    raise ValueError(name)


def asset_class(ticker):
    t = ticker.upper()
    for k, v in UNIVERSE.items():
        if t in v:
            return k
    if t.endswith("=F"):
        return "commodity"
    if t.startswith("^") or t in SECTOR_FUNDS:
        return "fund"
    return "crypto" if t.endswith(("-USD", "-USDT", "-BTC", "-EUR")) else "stock"


def commodity_group(ticker):
    for k, v in COMMODITIES.items():
        if ticker in v:
            return k
    return None


def _path(ticker):
    return os.path.join(CACHE, f"{ticker.replace('^', '_idx_').replace('=', '_')}.csv")


def _clean(raw):
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    idx = pd.to_datetime(raw.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return pd.DataFrame({
        "timestamps": idx, "open": raw["Open"].values.astype(float), "high": raw["High"].values.astype(float),
        "low": raw["Low"].values.astype(float), "close": raw["Close"].values.astype(float),
        "volume": raw["Volume"].values.astype(float),
    }).dropna()


def fetch_many(tickers, max_age_hours=24, batch=40):
    """Download and cache any tickers that are missing or stale. Returns the ones that have data."""
    import yfinance as yf
    os.makedirs(CACHE, exist_ok=True)
    need = [t for t in tickers if not (os.path.exists(_path(t)) and time.time() - os.path.getmtime(_path(t)) < max_age_hours * 3600)]
    for i in range(0, len(need), batch):
        chunk = need[i:i + batch]
        try:
            raw = yf.download(chunk, period="max", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)
        except Exception as e:
            print(f"  batch failed: {str(e)[:80]}", flush=True)
            continue
        for t in chunk:
            try:
                one = raw[t] if len(chunk) > 1 else raw
                df = _clean(one.copy())
                if len(df) > 300:
                    df.to_csv(_path(t), index=False)
            except Exception:
                pass
        print(f"  downloaded {min(i + batch, len(need))}/{len(need)}", flush=True)
        time.sleep(1)
    return [t for t in tickers if os.path.exists(_path(t))]


def load(ticker, max_age_hours=12, need_volume=True):
    """Finished daily candles, oldest first: timestamps, open, high, low, close, volume. Prices are split-adjusted."""
    os.makedirs(CACHE, exist_ok=True)
    path = _path(ticker)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_hours * 3600:
        df = pd.read_csv(path, parse_dates=["timestamps"])
    else:
        import yfinance as yf
        raw = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
        if raw is None or len(raw) == 0:
            raise ValueError(f"No data for '{ticker}'.")
        df = _clean(raw)
        df.to_csv(path, index=False)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    df = df[df["timestamps"] < today]                       # drop a candle that is still forming
    ok = (df["timestamps"] >= START) & (df["open"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["high"] >= df["low"])
    if need_volume:
        ok &= df["volume"] > 0
    return df[ok].reset_index(drop=True)
