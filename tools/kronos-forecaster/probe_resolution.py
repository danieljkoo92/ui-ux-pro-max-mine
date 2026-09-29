"""How finely can Kronos even see a daily price chart?

Kronos turns every candle into a token (a code from a fixed dictionary) and back. This measures how far
the price moves in that round trip, with no forecasting at all, for several lookback lengths.
If the round trip alone shifts the price by more than a normal day's move, every forecast starts from
the wrong price before it has predicted anything.

Usage:  .venv/Scripts/python.exe probe_resolution.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("KRONOS_DIR", os.path.join(HERE, "Kronos"))
sys.path.insert(0, HERE)

import numpy as np
import torch

import kronos_forecast as kf

TICKERS = "BTC-USD,ETH-USD,SOL-USD,SPY,QQQ,AAPL,NVDA,TSLA".split(",")
LOOKBACKS = [30, 60, 120, 200, 400]
ORIGINS, STEP = 8, 30

predictor = kf.load_predictor("small")
tok, dev, clip = predictor.tokenizer, predictor.device, predictor.clip
history = {t: kf.fetch(t, lookback=10 ** 6) for t in TICKERS}

out = {}
for L in LOOKBACKS:
    last_err, avg_err, day_move = [], [], []
    for t, full in history.items():
        for k in range(ORIGINS):
            end = len(full) - k * STEP
            if end - L < 0:
                break
            ctx = full.iloc[end - L:end]
            x = ctx[kf.X_COLS].values.astype(np.float32)
            mean, std = x.mean(axis=0), x.std(axis=0)
            xn = np.clip((x - mean) / (std + 1e-5), -clip, clip)
            with torch.no_grad():
                xt = torch.from_numpy(xn[np.newaxis]).to(dev)
                rec = tok.decode(tok.encode(xt, half=True), half=True).cpu().numpy()[0]
            rec = rec * (std + 1e-5) + mean
            close, rclose = x[:, 3].astype(float), rec[:, 3].astype(float)
            last_err.append(abs(rclose[-1] / close[-1] - 1))
            avg_err.append(float(np.mean(np.abs(rclose[-20:] / close[-20:] - 1))))
            day_move.append(float(np.mean(np.abs(np.diff(close[-21:]) / close[-21:-1]))))
    out[L] = {
        "windows": len(last_err),
        "price_shift_on_last_day_pct": round(100 * float(np.mean(last_err)), 2),
        "price_shift_last_20_days_pct": round(100 * float(np.mean(avg_err)), 2),
        "normal_daily_move_pct": round(100 * float(np.mean(day_move)), 2),
        "shift_vs_daily_move": round(float(np.mean(last_err)) / float(np.mean(day_move)), 2),
    }
    print(L, out[L], flush=True)

os.makedirs(os.path.join(HERE, "out", "diagnose"), exist_ok=True)
with open(os.path.join(HERE, "out", "diagnose", "resolution.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
