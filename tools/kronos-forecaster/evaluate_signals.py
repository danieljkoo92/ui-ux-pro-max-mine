"""Judge every chart signal on real daily prices. Keep only what beats luck, twice.

For each signal and each look-ahead (5, 30, 90 candles):
  1. Find every day the signal fired across the universe.
  2. Measure what price did afterwards, MINUS what that ticker normally does over the same number of days.
     (Stocks drift up. A signal only earns credit for what it adds on top of that drift.)
  3. Ask how likely a result that good is from luck alone, by reshuffling whole blocks of calendar time
     (days in the same month move together across tickers, so they are not independent tries).

A signal passes only if it clears the bar on the TUNE years and then again on the HOLDOUT years,
where the holdout bar is tightened for the number of signals that reached it.

Usage:  .venv/Scripts/python.exe evaluate_signals.py [module_name]      (default: ta_signals)
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

import market_data as md

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = sys.argv[1] if len(sys.argv) > 1 else "ta_signals"
HORIZONS = [5, 30, 90]
SPLIT = pd.Timestamp("2019-01-01")     # tune: 2000 to 2018, holdout: 2019 to today
MIN_FIRES = 200
BOOT = 4000
ALPHA = 0.05
rng = np.random.default_rng(7)

sigmod = importlib.import_module(MODULE)
names = list(sigmod.SIGNALS)


def block_key(dates, h):
    if h <= 5:
        return dates.dt.year * 100 + dates.dt.month
    if h <= 30:
        return dates.dt.year * 10 + (dates.dt.quarter)
    return dates.dt.year * 10 + (dates.dt.month > 6).astype(int)


def luck_p(signed, blocks):
    """Chance of a mean this far above zero if the signal added nothing (one-sided, block bootstrap)."""
    g = pd.DataFrame({"s": signed, "b": blocks}).groupby("b")["s"].agg(["sum", "count"])
    S, N = g["sum"].values, g["count"].values
    if len(S) < 8:
        return float("nan"), len(S)
    idx = rng.integers(0, len(S), size=(BOOT, len(S)))
    means = S[idx].sum(axis=1) / N[idx].sum(axis=1)
    return float((means <= 0).mean()), len(S)


frames = []
tickers = [t for v in md.UNIVERSE.values() for t in v]
for t in tickers:
    df = md.load(t)
    sig = sigmod.compute(df)
    c = df["close"]
    row = pd.DataFrame({"date": df["timestamps"], "ticker": t, "cls": md.asset_class(t)})
    for h in HORIZONS:
        row[f"fwd{h}"] = np.log(c.shift(-h) / c)
        row[f"end{h}"] = df["timestamps"].shift(-h)
    frames.append(pd.concat([row, sig[names].reset_index(drop=True)], axis=1))
    print(f"  {t}: {len(df)} days", flush=True)
data = pd.concat(frames, ignore_index=True)

results = []
for h in HORIZONS:
    d = data.dropna(subset=[f"fwd{h}"]).copy()
    d["split"] = np.where(d[f"end{h}"] < SPLIT, "tune", np.where(d["date"] >= SPLIT, "holdout", "gap"))
    d = d[d["split"] != "gap"]
    # what each ticker normally does over h days, inside each split
    grp = d.groupby(["ticker", "split"])[f"fwd{h}"]
    d["base"] = grp.transform("mean")
    d["p_up"] = grp.transform(lambda x: (x > 0).mean())
    d["excess"] = d[f"fwd{h}"] - d["base"]
    d["blk"] = block_key(d["date"], h)
    for name in names:
        f = d[d[name] != 0]
        rec = {"signal": name, "horizon": h}
        for split in ("tune", "holdout"):
            x = f[f["split"] == split]
            n = len(x)
            rec[f"{split}_n"] = int(n)
            if n < 30:
                rec[f"{split}_edge_pct"] = None
                rec[f"{split}_p"] = None
                continue
            signed = (x[name] * x["excess"]).values
            hit = ((x[name] * x[f"fwd{h}"]) > 0).mean()
            base_hit = np.where(x[name] > 0, x["p_up"], 1 - x["p_up"]).mean()
            p, nb = luck_p(signed, x["blk"].values)
            rec[f"{split}_edge_pct"] = round(100 * float(signed.mean()), 3)
            rec[f"{split}_hit"] = round(float(hit), 4)
            rec[f"{split}_base_hit"] = round(float(base_hit), 4)
            rec[f"{split}_p"] = None if np.isnan(p) else round(p, 4)
            rec[f"{split}_blocks"] = nb
            rec[f"{split}_up_calls"] = int((x[name] > 0).sum())
            rec[f"{split}_down_calls"] = int((x[name] < 0).sum())
            for side, lab in ((1, "up"), (-1, "down")):
                y = x[x[name] == side]
                rec[f"{split}_{lab}_edge_pct"] = round(100 * float((side * y["excess"]).mean()), 3) if len(y) >= 30 else None
        # crypto only, holdout, for context (small sample)
        xc = f[(f["split"] == "holdout") & (f["cls"] == "crypto")]
        rec["holdout_crypto_n"] = int(len(xc))
        rec["holdout_crypto_edge_pct"] = round(100 * float((xc[name] * xc["excess"]).mean()), 3) if len(xc) >= 30 else None
        results.append(rec)

res = pd.DataFrame(results)
res["tune_pass"] = (res["tune_n"] >= MIN_FIRES) & (res["tune_p"] < ALPHA) & (res["tune_edge_pct"] > 0)
m = int(res["tune_pass"].sum())
bar = ALPHA / max(m, 1)
res["holdout_bar"] = bar
res["final_pass"] = res["tune_pass"] & (res["holdout_n"] >= MIN_FIRES) & (res["holdout_p"] < bar) & (res["holdout_edge_pct"] > 0)

out_dir = os.path.join(HERE, "out", "signals")
os.makedirs(out_dir, exist_ok=True)
res.to_csv(os.path.join(out_dir, f"{MODULE}_report.csv"), index=False)
summary = {
    "module": MODULE, "signals_tested": len(names), "tests_run": int(len(res)),
    "universe": tickers, "days": int(len(data)), "split": str(SPLIT.date()),
    "passed_tune": m, "holdout_bar_p": bar,
    "passed_both": res[res["final_pass"]][["signal", "horizon", "tune_edge_pct", "holdout_edge_pct", "holdout_hit",
                                            "holdout_base_hit", "holdout_n", "holdout_p", "holdout_crypto_edge_pct",
                                            "holdout_up_edge_pct", "holdout_down_edge_pct"]].to_dict("records"),
}
with open(os.path.join(out_dir, f"{MODULE}_summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 30)
cols = ["signal", "horizon", "tune_n", "tune_edge_pct", "tune_p", "holdout_n", "holdout_edge_pct", "holdout_p",
        "holdout_hit", "holdout_base_hit", "tune_pass", "final_pass"]
print(res[cols].sort_values(["horizon", "tune_p"]).to_string(index=False))
print(json.dumps({k: v for k, v in summary.items() if k != "universe"}, indent=2))
