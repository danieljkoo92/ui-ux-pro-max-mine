"""Earnings report dates (Yahoo Finance, 2002 on, plus the next scheduled date), cached on disk."""
import os
import time

import numpy as np
import pandas as pd

import market_data as md

EDIR = os.path.join(md.CACHE, "_earnings")


def earnings_dates(t, max_age_days=7):
    """Report times in New York time, oldest first. None when Yahoo has nothing for the ticker."""
    os.makedirs(EDIR, exist_ok=True)
    p = os.path.join(EDIR, f"{t}.csv")
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < max_age_days * 86400:
        return pd.read_csv(p, parse_dates=["when"])
    import yfinance as yf
    try:
        e = yf.Ticker(t).get_earnings_dates(limit=100)
    except Exception:
        return None
    if e is None or not len(e):
        return None
    idx = pd.DatetimeIndex(e.index)
    local = idx.tz_convert("America/New_York").tz_localize(None) if idx.tz is not None else idx
    d = pd.DataFrame({"when": local}).drop_duplicates().sort_values("when")
    d.to_csv(p, index=False)
    return d


def reaction_index(dates, when):
    """Trading-day index where the market first reacts: same day if reported before 9:30, else the next day."""
    out = []
    dvals = pd.DatetimeIndex(dates).values
    for w in pd.DatetimeIndex(when):
        day = w.normalize()
        after_open = (w.hour, w.minute) >= (9, 30)
        j = np.searchsorted(dvals, np.datetime64(day), side="right" if after_open else "left")
        if j < len(dvals):
            out.append(j)
    return np.unique(np.array(out, dtype=int))
