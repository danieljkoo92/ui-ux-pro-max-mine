# Range Forecaster (and the Kronos notebook it replaced)

Where a price is likely to be 5 to 90 days from now, with a tested hit rate. No up-or-down call.

## Open it

Google Colab, free tier, nothing to install: open
[Range_Forecaster.ipynb in Colab](https://colab.research.google.com/github/danieljkoo92/ui-ux-pro-max-mine/blob/forecaster-v2/tools/kronos-forecaster/Range_Forecaster.ipynb),
then **Runtime → Run all**. Change the ticker in Step 2.

On a computer with Python: `pip install yfinance pandas numpy matplotlib`, then from this folder
`python vol_data.py` once, and `python forecast.py BTC-USD 90 --level 57747.77 100000`.

## What was tested (2019 to today, settings picked on 2000 to 2018)

| Market | Held the price (aims for 80%) | Narrower than the first version |
|---|---|---|
| 471 S&P 500 stocks | 79% | 7% to 8% |
| 18 indexes and sector funds | 80.5% | 6% to 11% |
| 26 commodity futures | 78% to 79% | 2.5% to 4% |
| Crypto | 80% to 82% | widened (the first version held 75% to 77%) |

- **Kronos**, the open-source candlestick model this folder started with, called direction at coin-flip accuracy on daily charts. Its "80% range" held only 9% to 22% of the time.
- **99 chart rules** from trading books and videos were tested for a direction call. None beat luck on both the older and newer years.
- **Weak spots:** sudden crashes (2008 held about 60%) and runaway trends (gold and Nvidia since 2019 held about 70% to 75%).

## Files

| File | What it does |
|---|---|
| `Range_Forecaster.ipynb` | The Colab notebook |
| `forecast.py` | One forecast: range, chances for price levels, chart |
| `history_view.py` | What it would have said in past years, with a year-by-year hit rate |
| `range_v2.py` | The range itself |
| `range_model.py`, `market_data.py`, `vol_data.py`, `earnings_data.py` | Swing measures, price data, volatility indexes, earnings dates |
| `out/range`, `out/tighten` | The fitted settings the forecast reads |
| `tighten.py`, `rolling_calibration.py`, `per_ticker_calibration.py`, `earnings_test.py` | Rebuild the settings, in that order |
| `evaluate_signals.py`, `evaluate_universe.py`, `ta_signals.py`, `ta_signals_book.py` | The chart-rule tests |
| `diagnose.py`, `run_local.py`, `kronos_forecast.py`, `jev_judge.py`, `Kronos_Forecaster.ipynb` | The original Kronos notebook and its replay test |
| `forecast_v1.py` | The first version of the range, kept for comparison |

Price data comes from Yahoo Finance and is downloaded when you run it. Nothing here is advice to buy or sell.
