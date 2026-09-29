"""Check the Bitcoin "1,064 days up / 364 days down" cycle chart against real daily prices.

Pulls every daily candle Yahoo has for BTC-USD, finds the real cycle highs and lows, measures the gaps,
and works out what would have to happen for the chart's projected low date to come true.

Usage:  .venv/Scripts/python.exe cycle_check.py [PROJECTED_LOW_DATE]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTED = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-10-05")

raw = yf.download("BTC-USD", period="max", interval="1d", auto_adjust=False, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
raw.index = pd.to_datetime(raw.index).tz_localize(None)
today = pd.Timestamp.utcnow().tz_localize(None).normalize()
d = raw[raw.index < today].dropna()          # finished days only
print(f"data: {d.index[0].date()} to {d.index[-1].date()}, {len(d)} days, last close {d['Close'].iloc[-1]:,.0f}")


def extreme(kind, start, end):
    w = d.loc[start:end]
    if kind == "high":
        i = w["High"].idxmax()
        return i, float(w.loc[i, "High"])
    i = w["Low"].idxmin()
    return i, float(w.loc[i, "Low"])


marks = [
    ("2015 low", "low", "2014-09-01", "2015-12-31"),
    ("2017 high", "high", "2017-01-01", "2018-06-30"),
    ("2018 low", "low", "2018-01-01", "2019-12-31"),
    ("2021 high", "high", "2021-01-01", "2022-06-30"),
    ("2022 low", "low", "2022-01-01", "2023-12-31"),
    ("2025 high", "high", "2024-06-01", "2026-06-30"),
]
pts = []
for name, kind, s, e in marks:
    when, px = extreme(kind, s, e)
    pts.append({"name": name, "kind": kind, "date": str(when.date()), "price": round(px, 2)})
ath_date = pd.Timestamp(pts[-1]["date"])
after = d.loc[ath_date:]
low_when = after["Low"].idxmin()
low_px = float(after.loc[low_when, "Low"])
pts.append({"name": "low since the 2025 high", "kind": "low", "date": str(low_when.date()), "price": round(low_px, 2)})

legs = []
for a, b in zip(pts[:-1], pts[1:]):
    days = (pd.Timestamp(b["date"]) - pd.Timestamp(a["date"])).days
    legs.append({"leg": f"{a['name']} -> {b['name']}", "days": days, "move_pct": round(100 * (b["price"] / a["price"] - 1), 1)})

last_close = float(d["Close"].iloc[-1])
last_day = d.index[-1]
days_left = (PROJECTED - last_day).days
need = low_px / last_close - 1                       # fall needed just to match the low already made
c = d["Close"]
r = (c.shift(-days_left) / c - 1).dropna()
hist_share = float((r <= need).mean())
hist_count = int((r <= need).sum())
worst = r.nsmallest(5)
vol20 = float(np.std(np.diff(np.log(c.values[-21:]))))
sigmas = float(np.log(1 + need) / (vol20 * np.sqrt(days_left)))

out = {
    "points": pts, "legs": legs,
    "projected_low_date": str(PROJECTED.date()), "last_day": str(last_day.date()), "last_close": round(last_close, 2),
    "days_until_projected_low": days_left,
    "days_from_2025_high_to_projected": (PROJECTED - ath_date).days,
    "days_from_2025_high_to_low_so_far": (low_when - ath_date).days,
    "rise_since_that_low_pct": round(100 * (last_close / low_px - 1), 1),
    "fall_needed_to_undercut_that_low_pct": round(100 * need, 1),
    "history_windows_checked": int(len(r)),
    "history_times_it_fell_that_much_in_that_many_days": hist_count,
    "history_share_pct": round(100 * hist_share, 3),
    "worst_falls_over_that_many_days": {str(k.date()): round(100 * v, 1) for k, v in worst.items()},
    "size_of_that_fall_in_normal_swings": round(sigmas, 1),
}
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
with open(os.path.join(HERE, "out", "cycle_check.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
