"""Re-stretch the range every month using only the last few years, so it keeps holding 80% as markets change.

Version 2 was stretched once, on all years together. On stocks that left 2019 to today holding about 79%.
Here the stretch is recomputed at the start of every month from forecasts that had already finished,
over a trailing window (3, 5 or 8 years), pooled across every ticker of the same kind. The window is picked
on 2012 to 2018 and judged once on 2019 to today, per ticker, against version 1 and version 2.

Usage:  .venv/Scripts/python.exe rolling_calibration.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

import market_data as md
import range_model as rm
import range_v2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "tighten")
chosen = json.load(open(os.path.join(HERE, "out", "range", "chosen.json")))
WINDOWS = (3, 5, 8)
PICK = (pd.Timestamp("2012-01-01"), pd.Timestamp("2019-01-01"))
SPLIT = pd.Timestamp("2019-01-01")
t0 = time.time()


def v1(df, cls, h):
    m, vol = chosen[f"{cls}_{h}"].split("/")
    k = float(vol.split(" x")[1]) if " x" in vol else 1.0
    return rm.forecast(df["close"], h, vol_name=vol.split(" x")[0], method=m, k=k).values


rows = []
tickers = []
for u in ("original", "stocks", "indexes", "commodities"):
    for t in md.universe(u):
        if t not in tickers:
            tickers.append(t)
for k_, t in enumerate(tickers):
    try:
        df = md.load(t, max_age_hours=96, need_volume=False)
    except Exception:
        continue
    if len(df) < 1000:
        continue
    cls = md.asset_class(t)
    ts = df["timestamps"].values
    lc = np.log(df["close"].values)
    for h in (30, 90):
        raw = range_v2.quantiles(df, t, h, stage="raw")[0].values
        cal = range_v2.PARAMS[range_v2.key_for(cls, h)]["calibration_all_years"]
        med = raw[:, 2:3]
        Q2 = raw.copy()                                   # version 2 as first built: one stretch on all years
        Q2[:, [0, 4]] = med + cal["outer"] * (raw[:, [0, 4]] - med)
        Q2[:, [1, 3]] = med + cal["inner"] * (raw[:, [1, 3]] - med)
        Q1 = v1(df, cls, h)
        idx = np.arange(260, len(df) - h, 5)
        x = pd.DataFrame({"ticker": t, "cls": cls, "h": h, "date": ts[idx], "end": ts[idx + h], "a": lc[idx + h] - lc[idx]})
        for j in range(5):
            x[f"r{j}"] = raw[idx, j]
            x[f"w{j}"] = Q2[idx, j]
            x[f"o{j}"] = Q1[idx, j]
        rows.append(x)
    if (k_ + 1) % 100 == 0:
        print(f"  {k_ + 1}/{len(tickers)}, {time.time() - t0:.0f}s", flush=True)
d = pd.concat(rows, ignore_index=True)
d = d[np.isfinite(d["a"]) & d[[f"r{j}" for j in range(5)]].notna().all(axis=1) & d[[f"o{j}" for j in range(5)]].notna().all(axis=1)]
d = d.reset_index(drop=True)
print(f"{d['ticker'].nunique()} tickers, {len(d):,} forecasts, {time.time() - t0:.0f}s", flush=True)

R = d[[f"r{j}" for j in range(5)]].values
a = d["a"].values
med = R[:, 2]
need_out = np.where(a < med, (med - a) / np.maximum(med - R[:, 0], 1e-12), (a - med) / np.maximum(R[:, 4] - med, 1e-12))
need_in = np.where(a < med, (med - a) / np.maximum(med - R[:, 1], 1e-12), (a - med) / np.maximum(R[:, 3] - med, 1e-12))
d["need_out"], d["need_in"] = need_out, need_in
d["month"] = d["date"].dt.to_period("M")
d.to_pickle(os.path.join(OUT, "rolling_samples.pkl"))          # reused by per_ticker_calibration.py

factors = {}
for (cls, h), g in d.groupby(["cls", "h"]):
    months = pd.period_range(g["month"].min(), pd.Timestamp.today().to_period("M"), freq="M")
    ends = g["end"].values
    order = np.argsort(ends)
    ends_s, dates_s = ends[order], g["date"].values[order]
    no, ni = g["need_out"].values[order], g["need_in"].values[order]
    for W in WINDOWS:
        tab = {}
        for mth in months:
            start = mth.start_time
            hi = np.searchsorted(ends_s, np.datetime64(start), side="left")        # finished before the month starts
            lo_date = np.datetime64(start - pd.DateOffset(years=W))
            sel = slice(0, hi)
            keep = dates_s[sel] >= lo_date
            if keep.sum() < 300:
                continue
            tab[str(mth)] = (float(np.quantile(no[sel][keep], 0.80)), float(np.quantile(ni[sel][keep], 0.50)))
        factors[(cls, h, W)] = tab
print(f"factors built, {time.time() - t0:.0f}s", flush=True)


def apply_rolling(g, W):
    tab = factors[(g["cls"].iloc[0], g["h"].iloc[0], W)]
    fo = g["month"].astype(str).map(lambda m: tab.get(m, (np.nan, np.nan))[0]).values
    fi = g["month"].astype(str).map(lambda m: tab.get(m, (np.nan, np.nan))[1]).values
    Rg = g[[f"r{j}" for j in range(5)]].values
    md_ = Rg[:, 2:3]
    Q = Rg.copy()
    Q[:, [0, 4]] = md_ + fo[:, None] * (Rg[:, [0, 4]] - md_)
    Q[:, [1, 3]] = md_ + fi[:, None] * (Rg[:, [1, 3]] - md_)
    return Q


def per_ticker(g, Q, mask):
    out = []
    aa = g["a"].values
    for t, gi in g[mask].groupby("ticker"):
        ii = gi.index.values - g.index.values[0] if False else np.where(mask & (g["ticker"].values == t))[0]
        Qi = Q[ii]
        ok = np.isfinite(Qi).all(axis=1)
        if ok.sum() < 30:
            continue
        inside = (aa[ii][ok] >= Qi[ok, 0]) & (aa[ii][ok] <= Qi[ok, 4])
        out.append((inside.mean(), (np.exp(Qi[ok, 4]) - np.exp(Qi[ok, 0])).mean()))
    return np.array(out)


res = []
for (cls, h), g in d.groupby(["cls", "h"]):
    g = g.reset_index(drop=True)
    pick_mask = (g["date"] >= PICK[0]) & (g["end"] < PICK[1])
    hold_mask = g["date"] >= SPLIT
    cands = {"v1": g[[f"o{j}" for j in range(5)]].values, "v2 (stretched once, all years)": g[[f"w{j}" for j in range(5)]].values}
    for W in WINDOWS:
        cands[f"v2, re-stretched monthly on last {W} years"] = apply_rolling(g, W)
    for nm, Q in cands.items():
        for lab, msk in (("2012-2018 (picking)", pick_mask.values), ("2019 on (judging)", hold_mask.values)):
            ok = msk & np.isfinite(Q).all(axis=1)
            aa = g["a"].values
            pt = per_ticker(g, Q, ok)
            res.append({"class": cls, "h": h, "version": nm, "period": lab, "n": int(ok.sum()),
                        "held80": float(((aa[ok] >= Q[ok, 0]) & (aa[ok] <= Q[ok, 4])).mean()),
                        "held50": float(((aa[ok] >= Q[ok, 1]) & (aa[ok] <= Q[ok, 3])).mean()),
                        "width80_pct": float(100 * (np.exp(Q[ok, 4]) - np.exp(Q[ok, 0])).mean()),
                        "tickers_within_75_85": float(((pt[:, 0] >= 0.75) & (pt[:, 0] <= 0.85)).mean()) if len(pt) else np.nan,
                        "ticker_p10": float(np.quantile(pt[:, 0], 0.1)) if len(pt) else np.nan,
                        "ticker_p90": float(np.quantile(pt[:, 0], 0.9)) if len(pt) else np.nan})
    print(f"  judged {cls} {h}, {time.time() - t0:.0f}s", flush=True)
r = pd.DataFrame(res)
for (cls, h), g in r.groupby(["class", "h"]):
    for per in r["period"].unique():
        b = g[(g["version"] == "v1") & (g["period"] == per)]["width80_pct"]
        if len(b):
            m = (r["class"] == cls) & (r["h"] == h) & (r["period"] == per)
            r.loc[m, "narrower_than_v1"] = 1 - r.loc[m, "width80_pct"] / float(b.iloc[0])
# pick the window per kind of market on 2012-2018: closest to 80% held, then narrowest
pick = {}
for (cls, h), g in r[(r["period"] == "2012-2018 (picking)") & r["version"].str.contains("re-stretched")].groupby(["class", "h"]):
    g = g.assign(miss=(g["held80"] - 0.80).abs() + (g["held50"] - 0.50).abs())
    g = g[np.isfinite(g["miss"]) & (g["n"] >= 300)]
    if not len(g):
        continue                                     # not enough finished forecasts in 2012-2018 (crypto)
    g = g[g["miss"] <= g["miss"].min() + 0.01].sort_values("width80_pct")
    pick[f"{cls}_{h}"] = int(g.iloc[0]["version"].split("last ")[1].split(" ")[0])
r.to_csv(os.path.join(OUT, "rolling_calibration.csv"), index=False)
latest = {}
for (cls, h, W), tab in factors.items():
    if pick.get(f"{cls}_{h}") == W:
        latest[f"{cls}_{h}"] = {"window_years": W, "monthly": tab}
with open(os.path.join(OUT, "rolling_factors.json"), "w", encoding="utf-8") as f:
    json.dump(latest, f)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
print(r.round(3).to_string(index=False))
print("picked windows:", pick)
print(f"done in {time.time() - t0:.0f}s")
