"""Test the range forecaster on a whole market it has never seen.

Two questions, kept apart:
  AS-IS   Take the settings already chosen (on the first 30 tickers) and apply them untouched.
          For stocks and indexes that is the "stock" and "fund" settings. For commodity futures,
          which the forecaster was never built for, the "fund" settings are borrowed.
  OWN     For commodities only: pick settings on the commodity futures' own older years (2000 to 2018),
          then read the result on 2019 to today.

Usage:  .venv/Scripts/python.exe range_universe.py stocks | indexes | commodities
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import market_data as md
import range_model as rm

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = sys.argv[1] if len(sys.argv) > 1 else "stocks"
chosen = json.load(open(os.path.join(HERE, "out", "range", "chosen.json")))
METHODS = ["normal", "own", "mix", "mix_own_center"]
VOLS = ["v20", "v60", "v250", "ewma", "blend3", "blend_60_250"]


def parse(name):
    m, vol = name.split("/")
    k = float(vol.split(" x")[1]) if " x" in vol else 1.0
    return dict(vol_name=vol.split(" x")[0], method=m, k=k)


def score(df, h, kw, label, t, group):
    c = df["close"]
    qf = rm.forecast(c, h, **kw)
    lc = np.log(c.values)
    ts = df["timestamps"]
    rows = []
    for i in range(260, len(df) - h, 10):
        q = qf.iloc[i]
        if q.isna().any():
            continue
        a = lc[i + h] - lc[i]
        path = lc[i + 1:i + h + 1] - lc[i]
        sp = "tune" if ts.iloc[i + h] < rm.SPLIT else ("holdout" if ts.iloc[i] >= rm.SPLIT else "gap")
        if sp == "gap":
            continue
        rows.append({
            "setting": label, "h": h, "ticker": t, "group": group, "split": sp,
            "inside80": q[0.10] <= a <= q[0.90], "inside50": q[0.25] <= a <= q[0.75], "above_mid": a > q[0.50],
            "below10": a < q[0.10], "above90": a > q[0.90], "width80": float(q[0.90] - q[0.10]),
            "loss": float(np.mean([rm.pinball(a, q[x], x) for x in rm.QS])),
            "touch_low": path.min() < q[0.10], "touch_high": path.max() > q[0.90],
        })
    return rows


tickers = md.fetch_many(md.universe(NAME), max_age_hours=72)
hist = {}
for t in tickers:
    try:
        d = md.load(t, max_age_hours=72, need_volume=False)
        if len(d) >= 1000:
            hist[t] = d
    except Exception:
        pass
print(f"{NAME}: {len(hist)} tickers", flush=True)

rows = []
for n, (t, df) in enumerate(hist.items()):
    cls = md.asset_class(t)
    borrow = "fund" if cls == "commodity" else cls
    group = md.commodity_group(t) or cls
    for h in (30, 90):
        rows += score(df, h, parse(chosen[f"{borrow}_{h}"]), "as-is", t, group)
        if NAME == "commodities":
            for m in METHODS:
                for v in (VOLS if m == "normal" else ["v60", "blend3", "blend_60_250"]):
                    rows += score(df, h, dict(vol_name=v, method=m), f"{m}/{v}", t, group)
    if (n + 1) % 50 == 0:
        print(f"  {n + 1}/{len(hist)}", flush=True)
d = pd.DataFrame(rows)

out_dir = os.path.join(HERE, "out", "range")
agg = lambda g: g.agg(n=("loss", "size"), inside80=("inside80", "mean"), inside50=("inside50", "mean"),
                      above_mid=("above_mid", "mean"), below10=("below10", "mean"), above90=("above90", "mean"),
                      width80=("width80", "mean"), loss=("loss", "mean"), touch_low=("touch_low", "mean"),
                      touch_high=("touch_high", "mean")).reset_index()
pd.set_option("display.width", 240); pd.set_option("display.max_rows", 300)
asis = d[d.setting == "as-is"]
a1 = agg(asis.groupby(["h", "split"]))
print("\nAS-IS, settings untouched"); print(a1.round(4).to_string(index=False))
a2 = agg(asis.groupby(["h", "split", "group"]))
print("\nAS-IS by group, holdout years"); print(a2[a2.split == "holdout"].round(4).to_string(index=False))
# how many individual tickers had a wide range that held between 70% and 90% of the time (holdout)
per = asis[asis.split == "holdout"].groupby(["h", "ticker"])["inside80"].agg(["mean", "size"]).reset_index()
per = per[per["size"] >= 30]
spread = per.groupby("h")["mean"].describe(percentiles=[0.1, 0.5, 0.9])
print("\nSpread across single tickers (holdout, share of the time the wide range held)"); print(spread.round(3).to_string())
report = {"universe": NAME, "tickers": len(hist), "as_is": json.loads(a1.round(4).to_json(orient="records")),
          "as_is_by_group_holdout": json.loads(a2[a2.split == "holdout"].round(4).to_json(orient="records")),
          "per_ticker_spread": json.loads(spread.round(4).reset_index().to_json(orient="records"))}

if NAME == "commodities":
    own = d[d.setting != "as-is"]
    o = agg(own.groupby(["h", "setting", "split"]))
    pick = {}
    for h in (30, 90):
        best = o[(o.h == h) & (o.split == "tune")].sort_values("loss").iloc[0]["setting"]
        pick[f"commodity_{h}"] = best
        print(f"\nOWN, {h} ahead: picked on the older years = {best}")
        print(o[(o.h == h) & (o.setting == best)].round(4).to_string(index=False))
        og = agg(own[(own.h == h) & (own.setting == best) & (own.split == "holdout")].groupby(["group"]))
        print(og.round(4).to_string(index=False))
        report[f"own_{h}"] = {"setting": best, "result": json.loads(o[(o.h == h) & (o.setting == best)].round(4).to_json(orient="records")),
                              "by_group_holdout": json.loads(og.round(4).to_json(orient="records"))}
        touch_t = own[(own.h == h) & (own.setting == best) & (own.split == "tune")]
        report[f"own_{h}"]["touch_vs_end_down"] = round(float(touch_t["touch_low"].mean() / touch_t["below10"].mean()), 3)
        report[f"own_{h}"]["touch_vs_end_up"] = round(float(touch_t["touch_high"].mean() / touch_t["above90"].mean()), 3)
    chosen.update(pick)
    with open(os.path.join(out_dir, "chosen.json"), "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
    byc = agg(own[own.setting.isin(pick.values())].groupby(["h", "setting", "split"]))
with open(os.path.join(out_dir, f"universe_{NAME}.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
