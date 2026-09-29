"""Let each ticker's own track record nudge its stretch, leaning on its group when its history is thin.

The monthly re-stretch (rolling_calibration.py) uses one factor per kind of market, so a big trending stock
like Apple can sit well below 80% while a sleepy one sits above. Here each ticker also gets its own factor
from its own finished forecasts in the same trailing window, blended with the group factor:

    factor = w * own + (1 - w) * group,   w = n / (n + K)

n is how many of the ticker's own forecasts finished inside the window. K is picked on 2012 to 2018
(smallest average distance of single tickers from 80%, then narrowest), and judged once on 2019 to today.

Usage:  .venv/Scripts/python.exe per_ticker_calibration.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "tighten")
d = pd.read_pickle(os.path.join(OUT, "rolling_samples.pkl"))
roll = json.load(open(os.path.join(OUT, "rolling_factors.json")))
KS = (0, 25, 75, 200, 600)          # 0 = the group factor only (what runs today)
PICK = (pd.Timestamp("2012-01-01"), pd.Timestamp("2019-01-01"))
SPLIT = pd.Timestamp("2019-01-01")
t0 = time.time()

res, per_ticker_tables = [], {}
for (cls, h), g in d.groupby(["cls", "h"]):
    key = f"{cls}_{h}"
    if key not in roll:
        continue                                   # crypto: no monthly re-stretch
    W = roll[key]["window_years"]
    tab = roll[key]["monthly"]
    g = g.reset_index(drop=True)
    months = g["month"].astype(str).values
    grp_o = np.array([tab.get(m, (np.nan, np.nan))[0] for m in months])
    grp_i = np.array([tab.get(m, (np.nan, np.nan))[1] for m in months])
    # each ticker's own factor for every month, from its forecasts that had finished before the month
    own_o = np.full(len(g), np.nan)
    own_i = np.full(len(g), np.nan)
    own_n = np.zeros(len(g))
    for t, gi in g.groupby("ticker"):
        ends = gi["end"].values
        order = np.argsort(ends)
        e_s, d_s = ends[order], gi["date"].values[order]
        no, ni = gi["need_out"].values[order], gi["need_in"].values[order]
        cache = {}
        for pos, m in zip(gi.index.values, gi["month"].values):
            if m not in cache:
                start = m.start_time
                hi = np.searchsorted(e_s, np.datetime64(start), side="left")
                keep = d_s[:hi] >= np.datetime64(start - pd.DateOffset(years=W))
                n = int(keep.sum())
                cache[m] = (float(np.quantile(no[:hi][keep], 0.80)) if n >= 20 else np.nan,
                            float(np.quantile(ni[:hi][keep], 0.50)) if n >= 20 else np.nan, n)
            own_o[pos], own_i[pos], own_n[pos] = cache[m]
    R = g[[f"r{j}" for j in range(5)]].values
    a = g["a"].values
    med = R[:, 2:3]
    for K in KS:
        w = own_n / (own_n + K) if K > 0 else np.zeros(len(g))
        w = np.where(np.isfinite(own_o), w, 0.0)
        fo = w * np.nan_to_num(own_o) + (1 - w) * grp_o
        fi = w * np.nan_to_num(own_i) + (1 - w) * grp_i
        Q = R.copy()
        Q[:, [0, 4]] = med + fo[:, None] * (R[:, [0, 4]] - med)
        Q[:, [1, 3]] = med + fi[:, None] * (R[:, [1, 3]] - med)
        for lab, msk in (("2012-2018 (picking)", ((g["date"] >= PICK[0]) & (g["end"] < PICK[1])).values),
                         ("2019 on (judging)", (g["date"] >= SPLIT).values)):
            ok = msk & np.isfinite(Q).all(axis=1)
            inside = (a >= Q[:, 0]) & (a <= Q[:, 4])
            pt = pd.DataFrame({"t": g["ticker"].values[ok], "in": inside[ok]}).groupby("t")["in"].agg(["mean", "size"])
            pt = pt[pt["size"] >= 30]["mean"]
            res.append({"class": cls, "h": h, "K": K, "period": lab, "n": int(ok.sum()),
                        "held80": float(inside[ok].mean()),
                        "held50": float(((a >= Q[:, 1]) & (a <= Q[:, 3]))[ok].mean()),
                        "width80_pct": float(100 * (np.exp(Q[ok, 4]) - np.exp(Q[ok, 0])).mean()),
                        "tickers_within_75_85": float(((pt >= 0.75) & (pt <= 0.85)).mean()),
                        "avg_ticker_miss_from_80": float((pt - 0.80).abs().mean()),
                        "ticker_p10": float(pt.quantile(0.1)), "ticker_p90": float(pt.quantile(0.9))})
    print(f"  {cls} {h}, {time.time() - t0:.0f}s", flush=True)

r = pd.DataFrame(res)
pick = {}
for (cls, h), g in r[r["period"] == "2012-2018 (picking)"].groupby(["class", "h"]):
    best = g["avg_ticker_miss_from_80"].min()
    g = g[g["avg_ticker_miss_from_80"] <= best + 0.003].sort_values("width80_pct")
    pick[f"{cls}_{h}"] = int(g.iloc[0]["K"])
r.to_csv(os.path.join(OUT, "per_ticker_calibration.csv"), index=False)
with open(os.path.join(OUT, "per_ticker_K.json"), "w", encoding="utf-8") as f:
    json.dump(pick, f, indent=2)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(r.round(3).to_string(index=False))
print("picked K:", pick)
