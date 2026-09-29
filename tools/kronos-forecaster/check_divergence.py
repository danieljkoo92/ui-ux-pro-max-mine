"""Three divergence rules passed on the stock market. Are they real, and is the divergence doing the work?

A divergence rule fires when price makes a new 20-day high (or low) but an indicator does not confirm it,
and calls a turn. Every one of those signals sits on a 20-day extreme, so the plain question is:
does the indicator add anything, or is this just "stocks that stretch tend to snap back"?

Checks:
  1. CONTROL      fade every new 20-day extreme, no indicator at all
  2. CONFIRMED    the same kind of extreme where the indicator DID confirm (so no divergence)
  3. NEIGHBOURS   the rule with nearby settings (15, 25, 30 day extremes; RSI 9 and 21)
  4. BY PERIOD    the edge in each two-year stretch of the newer years
  5. IN DOLLARS   what price actually did afterwards, not just relative to the ticker's normal drift
  6. ELSEWHERE    the same rule on indexes and on commodity futures

Usage:  .venv/Scripts/python.exe check_divergence.py
"""
import json
import os

import numpy as np
import pandas as pd

import market_data as md

HERE = os.path.dirname(os.path.abspath(__file__))
H = 30
SPLIT = pd.Timestamp("2019-01-01")
rng = np.random.default_rng(11)


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def events(price, ind, look):
    """For each new `look`-day extreme that has an earlier, lesser extreme 5 to 60 days back:
    +1/-1 fade call, and whether the indicator diverged."""
    n = len(price)
    fade = np.zeros(n, dtype=np.int8)        # every new extreme
    elig = np.zeros(n, dtype=np.int8)        # extremes with an earlier one to compare against
    div = np.zeros(n, dtype=np.int8)         # of those, the ones where the indicator failed to confirm
    hi = (price >= price.rolling(look).max()).values
    lo = (price <= price.rolling(look).min()).values
    p, x = price.values, ind.values
    lh = ll = None
    for t in range(n):
        if hi[t]:
            fade[t] = -1
            if lh is not None and 5 <= t - lh <= 60 and p[t] > p[lh]:
                elig[t] = -1
                if x[t] < x[lh]:
                    div[t] = -1
            lh = t
        if lo[t]:
            fade[t] = 1
            if ll is not None and 5 <= t - ll <= 60 and p[t] < p[ll]:
                elig[t] = 1
                if x[t] > x[ll]:
                    div[t] = 1
            ll = t
    return fade, elig, div


def collect(universe):
    rows = []
    tickers = md.fetch_many(md.universe(universe), max_age_hours=96)
    for t in tickers:
        try:
            df = md.load(t, max_age_hours=96)
        except Exception:
            continue
        if len(df) < 1000:
            continue
        c = df["close"]
        ts = df["timestamps"]
        fwd = np.log(c.shift(-H) / c).values
        end = ts.shift(-H)
        split = np.where((end < SPLIT).values, 0, np.where((ts >= SPLIT).values, 1, -1))
        split[~np.isfinite(fwd)] = -1
        excess = np.full(len(df), np.nan)
        for sp in (0, 1):
            m = split == sp
            if m.sum() > 50:
                excess[m] = fwd[m] - fwd[m].mean()
        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        obv = (np.sign(c.diff()).fillna(0) * df["volume"]).cumsum()
        base = {"date": ts.values, "yr": ts.dt.year.values, "q": ts.dt.quarter.values, "split": split, "fwd": fwd, "excess": excess}
        sig = {}
        r14 = rsi(c, 14)
        f, e, d = events(c, r14, 20)
        sig["control_fade_every_extreme"] = f
        sig["rsi_divergence"] = d
        sig["rsi_confirmed_no_divergence"] = np.where(d == 0, e, 0).astype(np.int8)
        sig["macd_divergence"] = events(c, macd, 20)[2]
        sig["obv_divergence"] = events(c, obv, 20)[2]
        for look in (15, 25, 30):
            sig[f"rsi_divergence_look{look}"] = events(c, r14, look)[2]
        for n_ in (9, 21):
            sig[f"rsi{n_}_divergence"] = events(c, rsi(c, n_), 20)[2]
        x = pd.DataFrame({**base, **sig})
        x = x[(x["split"] >= 0) & np.isfinite(x["excess"])]
        keep = (x[list(sig)] != 0).any(axis=1)
        x = x[keep].copy()
        x["ticker"] = t
        # the ticker's normal 30-day move, for the "in dollars" comparison
        for sp in (0, 1):
            m = split == sp
            x.loc[x["split"] == sp, "normal"] = float(np.nanmean(fwd[m])) if m.sum() > 50 else np.nan
        rows.append(x)
    d = pd.concat(rows, ignore_index=True)
    return d, [c_ for c_ in d.columns if c_ not in ("date", "yr", "q", "split", "fwd", "excess", "ticker", "normal")]


def luck(signed, blk):
    g = pd.DataFrame({"s": signed, "b": blk}).groupby("b")["s"].agg(["sum", "count"])
    S, N = g["sum"].values, g["count"].values
    if len(S) < 6:
        return float("nan")
    idx = rng.integers(0, len(S), size=(4000, len(S)))
    return float(((S[idx].sum(axis=1) / N[idx].sum(axis=1)) <= 0).mean())


def table(d, names):
    out = []
    d = d.assign(blk=d["yr"] * 10 + d["q"])
    for nm in names:
        for sp, lab in ((0, "older"), (1, "newer")):
            x = d[(d[nm] != 0) & (d["split"] == sp)]
            if len(x) < 100:
                continue
            signed = (x[nm] * x["excess"]).values
            up, dn = x[x[nm] > 0], x[x[nm] < 0]
            out.append({
                "rule": nm, "years": lab, "fired": len(x), "edge_pct": round(100 * signed.mean(), 3), "luck_p": round(luck(signed, x["blk"].values), 4),
                "up_calls": len(up), "after_up_call_pct": round(100 * (np.exp(up["fwd"].mean()) - 1), 2) if len(up) else None,
                "down_calls": len(dn), "after_down_call_pct": round(100 * (np.exp(dn["fwd"].mean()) - 1), 2) if len(dn) else None,
                "normal_30d_pct": round(100 * (np.exp(x["normal"].mean()) - 1), 2),
                "up_call_price_rose": round(float((up["fwd"] > 0).mean()), 4) if len(up) else None,
                "down_call_price_fell": round(float((dn["fwd"] < 0).mean()), 4) if len(dn) else None,
            })
    return pd.DataFrame(out)


pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 30)
report = {}
for uni in ("stocks", "indexes", "commodities"):
    d, names = collect(uni)
    t = table(d, names)
    print(f"\n===== {uni}: {d['ticker'].nunique()} tickers, {H} days ahead =====")
    print(t.to_string(index=False))
    report[uni] = json.loads(t.to_json(orient="records"))
    if uni == "stocks":
        x = d[(d["rsi_divergence"] != 0) & (d["split"] == 1)].copy()
        x["signed"] = x["rsi_divergence"] * x["excess"]
        x["period"] = pd.cut(x["yr"], [2018, 2020, 2022, 2024, 2030], labels=["2019-20", "2021-22", "2023-24", "2025-26"])
        bp = x.groupby("period", observed=True)["signed"].agg(["size", "mean"])
        bp["mean"] = (100 * bp["mean"]).round(3)
        print("\nRSI divergence by period (newer years), edge in %"); print(bp.to_string())
        report["stocks_by_period"] = json.loads(bp.reset_index().to_json(orient="records"))
        # how many single stocks show a positive edge in the newer years
        per = x.groupby("ticker")["signed"].agg(["size", "mean"])
        per = per[per["size"] >= 20]
        print(f"\nSingle stocks with a positive edge (newer years): {(per['mean'] > 0).mean():.1%} of {len(per)}")
        report["share_of_stocks_positive"] = round(float((per["mean"] > 0).mean()), 4)
with open(os.path.join(HERE, "out", "signals", "divergence_check.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
