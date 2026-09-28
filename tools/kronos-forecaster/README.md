# Kronos Forecaster + Jev

One notebook. Type a ticker, get a probabilistic price forecast from [Kronos](https://github.com/shiyu-coder/Kronos)
(the open-source candlestick foundation model, MIT) and a typed second opinion from
[Jev](https://docs.typesafe.ai) (TypeSafe AI's System One decision model) through Vercel AI Gateway.

## Open it

Google Colab, free tier, nothing to install:

1. Go to https://colab.research.google.com, choose the **GitHub** tab, paste `danieljkoo92/ui-ux-pro-max-mine`,
   pick `tools/kronos-forecaster/Kronos_Forecaster.ipynb`.
2. **Runtime → Run all.**
3. Step 5 asks once for a Vercel AI Gateway key (create one in the Vercel dashboard under AI Gateway → API keys).
   Store it in Colab Secrets as `AI_GATEWAY_API_KEY` to skip the prompt next time.

## What it does

| Step | Output |
|------|--------|
| 3 | Chart with median path, 50% and 80% bands, plain-English summary |
| 4 | Reality check: the last N days hidden from the model, forecast, then compared with what happened |
| 5 | Jev verdict: up / flat / down with probabilities, 0 to 3 trust score, price regime, agreement check |
| 6 | Watchlist scan: one table for several tickers |

## Files

- `Kronos_Forecaster.ipynb` — the notebook (self-contained; it writes the two helpers below at runtime)
- `kronos_forecast.py` — data fetch (Yahoo Finance via yfinance), batched Kronos sampling, summary, chart, backtest
- `jev_judge.py` — compact numeric snapshot, typed Jev questions, verdict rendering

## Honest limits

Kronos learned the shape of price moves from 12 billion candles. It knows nothing about news, earnings or rates.
Jev only sees the snapshot the notebook sends. Neither is trading advice. Run the reality check before trusting anything.
