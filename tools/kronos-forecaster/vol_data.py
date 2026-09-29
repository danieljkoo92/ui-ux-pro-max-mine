"""Volatility indexes: the options market's own forecast of how much prices will swing.

Yahoo:   ^VIX (S&P 500, 30 days), ^VIX3M (S&P 500, 3 months), ^VIX9D (9 days), ^VXN (Nasdaq-100),
         ^RVX (Russell 2000), ^VXD (Dow), ^GVZ (gold), ^OVX (crude oil), ^MOVE (US Treasury bonds)
Deribit: DVOL for Bitcoin and Ether (30 days), free public feed, daily since March 2021.

Each reading is an annualised percentage. Daily move = reading / 100 / sqrt(252) (sqrt(365) for crypto).

Usage:  .venv/Scripts/python.exe vol_data.py
"""
import json
import os
import time
import urllib.request

import pandas as pd

import market_data as md

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = md.CACHE
YAHOO = ["^VIX", "^VIX3M", "^VIX9D", "^VXN", "^RVX", "^VXD", "^GVZ", "^OVX", "^MOVE"]

# which volatility index speaks for which market (direct match only)
DIRECT = {
    "^GSPC": "^VIX", "SPY": "^VIX", "VTI": "^VIX", "^NYA": "^VIX", "MDY": "^VIX", "^MID": "^VIX",
    "^DJI": "^VXD", "DIA": "^VXD",
    "QQQ": "^VXN", "^IXIC": "^VXN",
    "IWM": "^RVX", "^RUT": "^RVX",
    "GLD": "^GVZ", "GC=F": "^GVZ",
    "CL=F": "^OVX", "BZ=F": "^OVX",
    "TLT": "^MOVE",
    "BTC-USD": "DVOL-BTC", "ETH-USD": "DVOL-ETH",
}


def path(name):
    return os.path.join(CACHE, f"_vol_{name.replace('^', '').replace('-', '_')}.csv")


def fetch_deribit(currency):
    """Daily DVOL closes from Deribit's public feed, paging backwards from today."""
    end = int(time.time() * 1000)
    start_floor = int(pd.Timestamp("2021-03-01").timestamp() * 1000)
    rows = []
    while end > start_floor:
        url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data?"
               f"currency={currency}&start_timestamp={start_floor}&end_timestamp={end}&resolution=1D")
        with urllib.request.urlopen(url, timeout=30) as r:
            res = json.load(r)["result"]
        data = res.get("data", [])
        if not data:
            break
        rows += data
        cont = res.get("continuation")
        if not cont or cont >= end:
            break
        end = cont
        time.sleep(0.3)
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts")
    d["date"] = pd.to_datetime(d["ts"], unit="ms").dt.normalize()
    return d.sort_values("date")[["date", "close"]].reset_index(drop=True)


def load(name):
    d = pd.read_csv(path(name), parse_dates=["date"])
    return d.set_index("date")["close"]


def main():
    import yfinance as yf
    os.makedirs(CACHE, exist_ok=True)
    for t in YAHOO:
        raw = yf.download(t, period="max", interval="1d", auto_adjust=False, progress=False)
        if raw is None or len(raw) == 0:
            print(f"  {t}: no data")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        idx = pd.to_datetime(raw.index)
        idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
        d = pd.DataFrame({"date": idx.normalize(), "close": raw["Close"].values.astype(float)}).dropna()
        d = d[d["close"] > 0]
        d.to_csv(path(t), index=False)
        print(f"  {t}: {len(d)} days, {d['date'].iloc[0].date()} to {d['date'].iloc[-1].date()}, last {d['close'].iloc[-1]:.2f}")
    for cur in ("BTC", "ETH"):
        try:
            d = fetch_deribit(cur)
            d.to_csv(path(f"DVOL-{cur}"), index=False)
            print(f"  DVOL-{cur}: {len(d)} days, {d['date'].iloc[0].date()} to {d['date'].iloc[-1].date()}, last {d['close'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"  DVOL-{cur}: failed ({str(e)[:100]})")


if __name__ == "__main__":
    main()
