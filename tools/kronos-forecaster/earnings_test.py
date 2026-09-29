"""Do earnings dates make stock ranges tighter?

A stock's swings are much bigger on the day after it reports earnings. Past-swing measures blend those days
in with ordinary days, so a window with no earnings gets a range that is too wide and a window with
earnings gets one that is too narrow. Two fixes are tested against the current (version 2) range:

  A  "by count"   stretch the version 2 range by one factor for windows with no earnings, one for one
                  report, one for two or more (each fitted on 2000 to 2018)
  B  "by stock"   rebuild the swing size from two parts: ordinary days (earnings days removed) plus each
                  stock's own typical earnings-day move for every report inside the window

Every candidate, including version 2, is stretched to hold exactly 80% and 50% on the older years, then
judged once on 2019 to today. Earnings dates come from Yahoo Finance (2002 on). The test uses the dates
that actually happened; companies announce them a few weeks ahead, and the quarterly rhythm is known further out.

Usage:  .venv/Scripts/python.exe earnings_test.py
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
os.makedirs(EDIR, exist_ok=True)
SPLIT = pd.Timestamp("2019-01-01")
QS = rm.QS
t0 = time.time()


from earnings_data import EDIR, earnings_dates, reaction_index  # noqa: E402


def calibrate_factors(Q, a, tune):
    med = Q[:, 2]
    f = {}
    for lo_j, hi_j, target in ((0, 4, 0.80), (1, 3, 0.50)):
        lo, hi = med - Q[:, lo_j], Q[:, hi_j] - med
        need = np.where(a < med, (med - a) / np.maximum(lo, 1e-12), (a - med) / np.maximum(hi, 1e-12))
        f[(lo_j, hi_j)] = float(np.quantile(need[tune], target))
    return f


def apply(Q, f):
    med = Q[:, 2:3]
    out = Q.copy()
    for (lo_j, hi_j), k in f.items():
        out[:, [lo_j, hi_j]] = med + k * (Q[:, [lo_j, hi_j]] - med)
    return out


def stats(Q, a, m):
    return {"n": int(m.sum()), "held80": float(((a[m] >= Q[m, 0]) & (a[m] <= Q[m, 4])).mean()),
            "held50": float(((a[m] >= Q[m, 1]) & (a[m] <= Q[m, 3])).mean()),
            "width80_pct": float((100 * (np.exp(Q[m, 4]) - np.exp(Q[m, 0]))).mean())}


tickers = md.UNIVERSE["stock"] + md.universe("stocks")
got = 0
for k, t in enumerate(tickers):
    if earnings_dates(t) is not None:
        got += 1
    if (k + 1) % 50 == 0:
        print(f"  earnings dates: {k + 1}/{len(tickers)} ({got} with data), {time.time() - t0:.0f}s", flush=True)
print(f"earnings dates for {got} of {len(tickers)} stocks", flush=True)

rows = []
params = range_v2.PARAMS
for k, t in enumerate(tickers):
    e = earnings_dates(t)
    if e is None:
        continue
    try:
        df = md.load(t, max_age_hours=96, need_volume=False)
    except Exception:
        continue
    if len(df) < 1000:
        continue
    dates = pd.DatetimeIndex(df["timestamps"])
    first_e = e["when"].min()
    react = reaction_index(dates, e["when"])
    if len(react) < 8:
        continue
    lc = np.log(df["close"].values)
    r1 = np.diff(lc, prepend=np.nan)
    is_e = np.zeros(len(df), dtype=bool)
    is_e[react] = True
    # ordinary-day swings: earnings reaction days blanked out
    r_ex = pd.Series(np.where(is_e, np.nan, r1))
    v60 = r_ex.rolling(60, min_periods=45).std()
    v250 = r_ex.rolling(250, min_periods=180).std()
    sig_ex = np.sqrt((v60 ** 2 + v250 ** 2) / 2).values
    # each stock's typical earnings-day move, from its last 8 reports known at the time
    e_sq = np.full(len(df), np.nan)
    past = []
    ptr = 0
    react_sorted = np.sort(react)
    for i in range(len(df)):
        while ptr < len(react_sorted) and react_sorted[ptr] <= i:
            j = react_sorted[ptr]
            if np.isfinite(r1[j]):
                past.append(r1[j] ** 2)
            ptr += 1
        if len(past) >= 2:
            e_sq[i] = np.mean(past[-8:])
    m = range_v2._regime(t, "stock", dates)
    cum = np.cumsum(is_e)
    for h in (30, 90):
        p = params[range_v2.key_for("stock", h)]
        # the unstretched version 2 range, so every candidate is calibrated the same way, on the older years only
        Qraw = range_v2.quantiles(df, t, h, stage="raw")[0].values
        idx = np.arange(260, len(df) - h, 5)
        idx = idx[dates[idx] >= first_e]                      # only days covered by the earnings history
        n_e = cum[idx + h] - cum[idx]
        a = lc[idx + h] - lc[idx]
        beta = p["fit"]["regime_power"]
        mm = np.where(np.isfinite(m[idx]), np.clip(m[idx], 0.3, 5.0) ** beta, 1.0)
        base_e = np.where(np.isfinite(e_sq[idx]), e_sq[idx], 4 * sig_ex[idx] ** 2)
        V = (sig_ex[idx] * mm) ** 2 * h + n_e * base_e
        x = pd.DataFrame({"ticker": t, "h": h, "date": dates[idx], "end": dates[idx + h], "a": a, "n_e": n_e, "sdB": np.sqrt(V)})
        for j, q in enumerate(QS):
            x[f"v{j}"] = Qraw[idx, j]
        rows.append(x)
    if (k + 1) % 50 == 0:
        print(f"  built {k + 1}/{len(tickers)}, {time.time() - t0:.0f}s", flush=True)

d = pd.concat(rows, ignore_index=True)
d = d[np.isfinite(d["a"]) & np.isfinite(d["sdB"]) & d[[f"v{j}" for j in range(5)]].notna().all(axis=1)]
d["split"] = np.where(d["end"] < SPLIT, "tune", np.where(d["date"] >= SPLIT, "holdout", "gap"))
d = d[d["split"] != "gap"].reset_index(drop=True)
print(f"{d['ticker'].nunique()} stocks, {len(d):,} forecasts, {time.time() - t0:.0f}s", flush=True)

res = []
saved_factors = {}
for h in (30, 90):
    x = d[d["h"] == h].reset_index(drop=True)
    a = x["a"].values
    tune, hold = x["split"].values == "tune", x["split"].values == "holdout"
    V2 = x[[f"v{j}" for j in range(5)]].values
    grp = np.minimum(x["n_e"].values, 2)
    cands = {}
    cands["version 2 (VIX-scaled)"] = apply(V2, calibrate_factors(V2, a, tune))
    # A: one stretch per earnings count, each fitted on the older years
    QA = V2.copy()
    for g in (0, 1, 2):
        sel = grp == g
        if (sel & tune).sum() > 500:
            QA[sel] = apply(V2[sel], calibrate_factors(V2[sel], a[sel], tune[sel]))
        else:
            QA[sel] = apply(V2[sel], calibrate_factors(V2, a, tune))
    cands["A: stretch by number of reports"] = QA
    # save A's settings as factors RELATIVE to the one-size stretch, so they can sit on top of any calibration
    f_all = calibrate_factors(V2, a, tune)
    rel = {}
    for g in (0, 1, 2):
        sel = grp == g
        if (sel & tune).sum() > 500:
            fg = calibrate_factors(V2[sel], a[sel], tune[sel])
            rel[str(g)] = {"outer": fg[(0, 4)] / f_all[(0, 4)], "inner": fg[(1, 3)] / f_all[(1, 3)], "tune_rows": int((sel & tune).sum())}
    saved_factors[str(h)] = rel
    # B: ordinary swings plus each stock's own earnings-day move
    z = a[tune] / x["sdB"].values[tune]
    shape = (np.array([rm.Z[q] for q in QS]) + np.quantile(z, QS)) / 2
    QB = x["sdB"].values[:, None] * shape[None, :]
    cands["B: ordinary swings plus each stock's earnings move"] = apply(QB, calibrate_factors(QB, a, tune))
    base_w = None
    for nm, Q in cands.items():
        for sp, msk in (("tune", tune), ("holdout", hold)):
            s = stats(Q, a, msk)
            rec = {"h": h, "candidate": nm, "split": sp, **s}
            for g, lab in ((0, "no report"), (1, "one report"), (2, "two or more")):
                mg = msk & (grp == g)
                if mg.sum() > 100:
                    sg = stats(Q, a, mg)
                    rec[f"{lab}: share"] = round(mg.sum() / msk.sum(), 3)
                    rec[f"{lab}: held80"] = round(sg["held80"], 3)
                    rec[f"{lab}: width80"] = round(sg["width80_pct"], 2)
            res.append(rec)
r = pd.DataFrame(res)
for h in (30, 90):
    b = r[(r["h"] == h) & (r["split"] == "holdout") & (r["candidate"].str.startswith("version 2"))]["width80_pct"].iloc[0]
    r.loc[r["h"] == h, "narrower_than_v2"] = 1 - r.loc[r["h"] == h, "width80_pct"] / b
r.to_csv(os.path.join(OUT, "earnings_test.csv"), index=False)
with open(os.path.join(OUT, "earnings_factors.json"), "w", encoding="utf-8") as f:
    json.dump(saved_factors, f, indent=2)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
print(r.round(3).to_string(index=False))
print(f"done in {time.time() - t0:.0f}s")
