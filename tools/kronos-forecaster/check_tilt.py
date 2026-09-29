"""Does leaning the range after a stretched move make the forecast better?

On the stock market, a stock at a new 20-day low has done a little better than normal over the next 30 days,
and one at a new 20-day high a little worse. This checks whether shifting the whole range by that amount
(measured on the older years only) improves the range's score on the newer years. If it does not improve
the score, it does not go in the forecaster.

Usage:  .venv/Scripts/python.exe check_tilt.py
"""
import json
import os

import numpy as np
import pandas as pd

import market_data as md
import range_model as rm
from check_divergence import events, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
H = 30
chosen = json.load(open(os.path.join(HERE, "out", "range", "chosen.json")))
m_, v_ = chosen["stock_30"].split("/")
KW = dict(vol_name=v_, method=m_)

rows = []
for t in md.fetch_many(md.universe("stocks"), max_age_hours=96):
    try:
        df = md.load(t, max_age_hours=96)
    except Exception:
        continue
    if len(df) < 1000:
        continue
    c, ts = df["close"], df["timestamps"]
    fwd = np.log(c.shift(-H) / c)
    qf = rm.forecast(c, H, **KW)
    fade, elig, div = events(c, rsi(c, 14), 20)
    x = pd.DataFrame({"date": ts, "end": ts.shift(-H), "fwd": fwd, "fade": fade, "div": div})
    for q in rm.QS:
        x[q] = qf[q]
    x = x.dropna()
    x["split"] = np.where(x["end"] < rm.SPLIT, "tune", np.where(x["date"] >= rm.SPLIT, "holdout", "gap"))
    x = x[(x["split"] != "gap") & (x["fade"] != 0)]
    # what the stock normally does, measured inside each split, so the lean is only the extra
    for sp in ("tune", "holdout"):
        m = (np.where(ts.shift(-H) < rm.SPLIT, "tune", np.where(ts >= rm.SPLIT, "holdout", "gap")) == sp) & fwd.notna().values
        x.loc[x["split"] == sp, "normal"] = float(fwd[m].mean()) if m.sum() > 50 else np.nan
    x["ticker"] = t
    rows.append(x)
d = pd.concat(rows, ignore_index=True).dropna()


def loss(frame, shift):
    a = frame["fwd"].values
    tot = 0.0
    for q in rm.QS:
        tot += rm.pinball(a, frame[q].values + shift, q).mean()
    return tot / len(rm.QS)


out = {}
for name, col in (("every new 20-day extreme", "fade"), ("only with RSI divergence", "div")):
    tune = d[(d["split"] == "tune") & (d[col] != 0)]
    hold = d[(d["split"] == "holdout") & (d[col] != 0)]
    up_t, dn_t = tune[tune[col] > 0], tune[tune[col] < 0]
    lean_up = float((up_t["fwd"] - up_t["normal"]).mean())          # learned on the older years only
    lean_dn = float((dn_t["fwd"] - dn_t["normal"]).mean())
    shift = np.where(hold[col].values > 0, lean_up, lean_dn)
    base, leaned = loss(hold, 0.0), loss(hold, shift)
    half = loss(hold, shift / 2)
    mid = hold[0.50].values
    out[name] = {
        "days_tested": int(len(hold)), "lean_after_new_low_pct": round(100 * lean_up, 3), "lean_after_new_high_pct": round(100 * lean_dn, 3),
        "score_without_lean": round(base, 6), "score_with_lean": round(leaned, 6), "score_with_half_lean": round(half, 6),
        "improvement_pct": round(100 * (base - leaned) / base, 3),
        "ended_above_middle_without": round(float((hold["fwd"].values > mid).mean()), 4),
        "ended_above_middle_with": round(float((hold["fwd"].values > mid + shift).mean()), 4),
        "after_new_low_above_middle_without": round(float((hold["fwd"].values > mid)[hold[col].values > 0].mean()), 4),
        "after_new_low_above_middle_with": round(float((hold["fwd"].values > mid + shift)[hold[col].values > 0].mean()), 4),
        "after_new_high_above_middle_without": round(float((hold["fwd"].values > mid)[hold[col].values < 0].mean()), 4),
        "after_new_high_above_middle_with": round(float((hold["fwd"].values > mid + shift)[hold[col].values < 0].mean()), 4),
    }
print(json.dumps(out, indent=2))
with open(os.path.join(HERE, "out", "signals", "tilt_check.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
