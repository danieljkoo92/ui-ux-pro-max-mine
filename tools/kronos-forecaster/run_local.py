"""Run the notebook's Steps 2 to 5 on this computer instead of in Colab.

Steps 3 and 4 are the notebook's own code, unchanged. Step 5 builds the exact snapshot and questions
the notebook would send to Jev and writes them to out/jev_request.json, so the request can be sent
by whichever route to Jev is open. render_verdict.py turns Jev's reply into the notebook's verdict text.

Usage:  .venv/Scripts/python.exe run_local.py [TICKER] [DAYS_AHEAD] [PATHS]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("KRONOS_DIR", os.path.join(HERE, "Kronos"))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")  # no window; charts are saved as files
import matplotlib.pyplot as plt
import torch

import kronos_forecast as kf
import jev_judge as jj

# Step 2) Settings, same defaults as the notebook
TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "BTC-USD"
DAYS_AHEAD = int(sys.argv[2]) if len(sys.argv) > 2 else 30
PATHS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MODEL_SIZE = "small"
CHUNK = 20 if torch.cuda.is_available() else 5
JEV_PLAYGROUND_MODEL = "typesafe/jev-1.13"

OUT = os.path.join(HERE, "out", f"{TICKER}_{DAYS_AHEAD}d")  # one folder per ticker and horizon
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)  # same paths every run, so a rerun can be compared

print(f"Ticker {TICKER}  |  {DAYS_AHEAD} trading days ahead  |  {PATHS} paths  |  Kronos-{MODEL_SIZE}")

# Step 3) Run the forecast
print(f"Loading Kronos-{MODEL_SIZE} from Hugging Face (first time only)...")
predictor = kf.load_predictor(MODEL_SIZE)
df = kf.fetch(TICKER, lookback=400)
print(f"Got {len(df)} days of {TICKER}, last close {kf.fmt_money(df['close'].iloc[-1])} on {df['timestamps'].iloc[-1].date()}")
paths, future = kf.forecast_paths(predictor, df, DAYS_AHEAD, PATHS, crypto=kf.is_crypto(TICKER), chunk=CHUNK)
summary = kf.summarize(df, paths, future, TICKER)
forecast_text = kf.plain_english(summary)
print(forecast_text)
kf.plot_forecast(df, summary).savefig(os.path.join(OUT, "forecast.png"), dpi=150)
plt.close("all")

# Step 4) Reality check
bt_summary, actual, train, score = kf.backtest(predictor, df, DAYS_AHEAD, PATHS, crypto=kf.is_crypto(TICKER), chunk=CHUNK)
reality_text = kf.backtest_english(score, DAYS_AHEAD)
print(reality_text)
kf.plot_forecast(train, bt_summary, actual=actual,
                 title=f"{TICKER}: what Kronos would have said {DAYS_AHEAD} trading days ago (red dashed = reality)"
                 ).savefig(os.path.join(OUT, "reality_check.png"), dpi=150)
plt.close("all")

# Step 5) Build the Jev request the notebook would send
state = jj.build_state(df, summary, score)
questions = jj.build_questions(summary["horizon_days"])
with open(os.path.join(OUT, "jev_request.json"), "w", encoding="utf-8") as f:
    json.dump({"model": JEV_PLAYGROUND_MODEL, "state": state, "questions": questions}, f, indent=2)
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump({"ticker": TICKER, "horizon_days": int(summary["horizon_days"]),
               "forecast_text": forecast_text, "reality_text": reality_text}, f, indent=2)
print(f"Wrote jev_request.json, forecast.png and reality_check.png to {OUT}")
