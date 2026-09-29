"""Does each single ticker still hold close to 80%, or only the average?

Daniel forecasts one ticker at a time, so a version that holds 80% on average but 70% on some stocks and
90% on others is worse for him. This compares the hold rate per ticker, 2019 to today, for version 1,
version 2, and version 2 with each ticker's own past moves shaping its range ("v2 + own shape").

Usage:  .venv/Scripts/python.exe per_ticker_check.py [stocks|indexes|commodities|original]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import market_data as md
import range_model as rm
import range_v2

HERE = os.path.dirname(os.path.abspath(__file__))
U = sys.argv[1] if len(sys.argv) > 1 else "stocks"
chosen = json.load(open(os.path.join(HERE, "out", "range", "chosen.json")))
SPLIT = pd.Timestamp("2019-01-01")
ZQ = np.array([rm.Z[q] for q in rm.QS])


def v1(df, cls, h):
    m, vol = chosen[f"{cls}_{h}"].split("/")
    k = float(vol.split(" x")[1]) if " x" in vol else 1.0
    return rm.forecast(df["close"], h, vol_name=vol.split(" x")[0], method=m, k=k).values


def own_shape(df, Q, h):
    """Re-shape a range using this ticker's own past moves, measured in units of the range's own width."""
    c = df["close"]
    width = (Q[:, 4] - Q[:, 0]) / (2 * 1.2816)            # the range's implied swing over h days
    a = np.log(c.shift(-h) / c).values
    z = pd.Series(a / width)
    known = z.shift(h)                                       # a move is known only after it finishes
    own = np.column_stack([known.expanding(min_periods=500).quantile(q).values for q in rm.QS])
    shaped = width[:, None] * (ZQ[None, :] + own) / 2
    return np.where(np.isfinite(own).all(axis=1)[:, None], shaped, Q)


rows = []
for t in md.fetch_many(md.universe(U), max_age_hours=96):
    try:
        df = md.load(t, max_age_hours=96, need_volume=False)
    except Exception:
        continue
    if len(df) < 1500:
        continue
    cls = md.asset_class(t)
    ts = df["timestamps"]
    for h in (30, 90):
        a = np.log(df["close"].shift(-h) / df["close"]).values
        Q2 = range_v2.quantiles(df, t, h)[0].values
        cands = {"v1": v1(df, cls, h), "v2": Q2, "v2 + own shape": own_shape(df, Q2, h)}
        idx = np.arange(260, len(df) - h, 5)
        idx = idx[(ts.iloc[idx] >= SPLIT).values]
        for nm, Q in cands.items():
            ok = np.isfinite(Q[idx]).all(axis=1) & np.isfinite(a[idx])
            ii = idx[ok]
            if len(ii) < 30:
                continue
            inside = (a[ii] >= Q[ii, 0]) & (a[ii] <= Q[ii, 4])
            w = np.exp(Q[ii, 4]) - np.exp(Q[ii, 0])
            rows.append({"ticker": t, "h": h, "version": nm, "held80": inside.mean(), "width": 100 * w.mean(), "n": len(ii)})
d = pd.DataFrame(rows)
out = []
for (h, nm), g in d.groupby(["h", "version"]):
    out.append({"h": h, "version": nm, "tickers": len(g), "avg_held80": round(g["held80"].mean(), 3),
                "p10": round(g["held80"].quantile(0.1), 3), "median": round(g["held80"].median(), 3), "p90": round(g["held80"].quantile(0.9), 3),
                "share_within_75_85": round(((g["held80"] >= 0.75) & (g["held80"] <= 0.85)).mean(), 3),
                "avg_width_pct": round(g["width"].mean(), 2)})
o = pd.DataFrame(out)
o.to_csv(os.path.join(HERE, "out", "tighten", f"per_ticker_{U}.csv"), index=False)
d.to_csv(os.path.join(HERE, "out", "tighten", f"per_ticker_{U}_detail.csv"), index=False)
pd.set_option("display.width", 200)
print(o.to_string(index=False))
