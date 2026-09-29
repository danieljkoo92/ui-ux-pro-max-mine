"""Why was the forecast off? Replay it at many past dates and score it against what happened.

One reality check is one coin flip. This runs the same forecast at N past dates for several tickers,
with the future hidden each time, and compares Kronos with the dumbest possible guess ("no change").

Usage:
  .venv/Scripts/python.exe diagnose.py --tag baseline
  .venv/Scripts/python.exe diagnose.py --tag short --horizon 10
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("KRONOS_DIR", os.path.join(HERE, "Kronos"))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import torch

import kronos_forecast as kf

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--tickers", default="BTC-USD,ETH-USD,SOL-USD,SPY,QQQ,AAPL,NVDA,TSLA")
ap.add_argument("--lookback", type=int, default=400)
ap.add_argument("--horizon", type=int, default=30)
ap.add_argument("--paths", type=int, default=20)
ap.add_argument("--origins", type=int, default=8)
ap.add_argument("--step", type=int, default=0, help="bars between test dates; 0 = same as horizon (no overlap)")
ap.add_argument("--skip", type=int, default=0, help="skip the most recent N test dates (keeps them as a holdout)")
ap.add_argument("--size", default="small")
ap.add_argument("--T", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--chunk", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
step = a.step or a.horizon

torch.manual_seed(a.seed)
np.random.seed(a.seed)
predictor = kf.load_predictor(a.size)
print(f"[{a.tag}] device {predictor.device}  lookback {a.lookback}  horizon {a.horizon}  paths {a.paths}  "
      f"origins {a.origins} (skip {a.skip})  size {a.size}  T {a.T}  top_p {a.top_p}", flush=True)

# 1) Build every test window
windows = []
for t in [x.strip().upper() for x in a.tickers.split(",") if x.strip()]:
    try:
        full = kf.fetch(t, lookback=10 ** 6)
    except Exception as e:
        print(f"  {t}: skipped ({str(e)[:80]})", flush=True)
        continue
    crypto = kf.is_crypto(t)
    for k in range(a.skip, a.skip + a.origins):
        end = len(full) - a.horizon - k * step
        if end - a.lookback < 0:
            break
        ctx = full.iloc[end - a.lookback:end].reset_index(drop=True)
        act = full.iloc[end:end + a.horizon].reset_index(drop=True)
        windows.append({"ticker": t, "k": k, "ctx": ctx, "act": act, "crypto": crypto})
print(f"  {len(windows)} test windows", flush=True)

# 2) Forecast them in batches (every window repeated `paths` times)
jobs = [(i, p) for i in range(len(windows)) for p in range(a.paths)]
closes = {i: [] for i in range(len(windows))}
t0 = time.time()
for s in range(0, len(jobs), a.chunk):
    batch = jobs[s:s + a.chunk]
    dfs, xts, yts = [], [], []
    for i, _ in batch:
        w = windows[i]
        dfs.append(w["ctx"][kf.X_COLS])
        xts.append(w["ctx"]["timestamps"])
        yts.append(w["act"]["timestamps"])
    preds = predictor.predict_batch(df_list=dfs, x_timestamp_list=xts, y_timestamp_list=yts, pred_len=a.horizon,
                                    T=a.T, top_p=a.top_p, sample_count=1, verbose=False)
    for (i, _), p in zip(batch, preds):
        closes[i].append(p["close"].values)
    done = s + len(batch)
    el = time.time() - t0
    print(f"  {done}/{len(jobs)} paths  {el:.0f}s elapsed, about {el / done * (len(jobs) - done):.0f}s left", flush=True)

# 3) Score
rows = []
for i, w in enumerate(windows):
    c = w["ctx"]["close"].values.astype(float)
    actual = w["act"]["close"].values.astype(float)
    P = np.stack(closes[i])                                  # (paths, horizon)
    last = float(c[-1])
    end = P[:, -1]
    med, p10, p90 = np.median(end), np.percentile(end, 10), np.percentile(end, 90)
    a_end = float(actual[-1])
    band_lo, band_hi = np.percentile(P, 10, axis=0), np.percentile(P, 90, axis=0)
    rets = np.diff(np.log(c))
    rows.append({
        "ticker": w["ticker"], "as_of": str(w["ctx"]["timestamps"].iloc[-1].date()), "k": w["k"],
        "model_change": med / last - 1, "actual_change": a_end / last - 1,
        "p10_change": p10 / last - 1, "p90_change": p90 / last - 1,
        "share_up": float((end > last).mean()),
        "end_inside_80": bool(p10 <= a_end <= p90),
        "days_inside_80": float(((actual >= band_lo) & (actual <= band_hi)).mean()),
        "z_last": float((last - c.mean()) / c.std()),
        "change_20d": float(last / c[-21] - 1), "change_60d": float(last / c[-61] - 1),
        # what a plain random walk with this ticker's own recent swings would give as an 80% range
        "rw_band": float(1.2816 * np.std(rets[-60:]) * np.sqrt(a.horizon)),
    })
df = pd.DataFrame(rows)
os.makedirs(os.path.join(HERE, "out", "diagnose"), exist_ok=True)
df.to_csv(os.path.join(HERE, "out", "diagnose", f"{a.tag}.csv"), index=False)
# every simulated path, so other ways of reading them can be scored later without rerunning the model
np.savez_compressed(
    os.path.join(HERE, "out", "diagnose", f"{a.tag}.npz"),
    paths=np.stack([np.stack(closes[i]) for i in range(len(windows))]),
    actual=np.stack([w["act"]["close"].values.astype(float) for w in windows]),
    last=np.array([float(w["ctx"]["close"].iloc[-1]) for w in windows]),
    daily_vol_60=np.array([float(np.std(np.diff(np.log(w["ctx"]["close"].values.astype(float)))[-60:])) for w in windows]),
)


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")


m, r = df["model_change"].values, df["actual_change"].values
rw_inside = (np.abs(np.log1p(r)) <= df["rw_band"].values)
s = {
    "tag": a.tag, "n": int(len(df)),
    "direction_hit": float((np.sign(m) == np.sign(r)).mean()),
    "always_up_hit": float((r > 0).mean()),
    "momentum_hit": float((np.sign(df["change_20d"].values) == np.sign(r)).mean()),
    "model_abs_err": float(np.mean(np.abs(m - r))),
    "no_change_abs_err": float(np.mean(np.abs(r))),
    "avg_model_call": float(m.mean()), "avg_actual": float(r.mean()),
    "bias": float((m - r).mean()),
    "share_calls_down": float((m < 0).mean()),
    "end_inside_80": float(df["end_inside_80"].mean()),
    "days_inside_80": float(df["days_inside_80"].mean()),
    "avg_band_width": float((df["p90_change"] - df["p10_change"]).mean()),
    "random_walk_band_inside_80": float(rw_inside.mean()),
    "corr_model_vs_actual": corr(m, r),
    "corr_model_vs_z_last": corr(m, df["z_last"].values),
    "corr_model_vs_change_60d": corr(m, df["change_60d"].values),
    "seconds": round(time.time() - t0),
}
with open(os.path.join(HERE, "out", "diagnose", f"{a.tag}.json"), "w", encoding="utf-8") as f:
    json.dump(s, f, indent=2)
print(json.dumps(s, indent=2))
