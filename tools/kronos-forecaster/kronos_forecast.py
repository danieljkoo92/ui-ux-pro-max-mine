"""Kronos Forecaster helper: fetch a price chart for any ticker, run Kronos, return a probabilistic forecast."""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

KRONOS_DIR = os.environ.get("KRONOS_DIR", "Kronos")
if KRONOS_DIR not in sys.path:
    sys.path.insert(0, KRONOS_DIR)
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

MODEL_IDS = {"mini": "NeoQuasar/Kronos-mini", "small": "NeoQuasar/Kronos-small", "base": "NeoQuasar/Kronos-base"}
TOKENIZER_IDS = {"mini": "NeoQuasar/Kronos-Tokenizer-2k", "small": "NeoQuasar/Kronos-Tokenizer-base", "base": "NeoQuasar/Kronos-Tokenizer-base"}
MAX_CONTEXT = {"mini": 2048, "small": 512, "base": 512}
X_COLS = ["open", "high", "low", "close", "volume", "amount"]


def load_predictor(size="small"):
    tok = KronosTokenizer.from_pretrained(TOKENIZER_IDS[size])
    mdl = Kronos.from_pretrained(MODEL_IDS[size])
    return KronosPredictor(mdl, tok, max_context=MAX_CONTEXT[size])


def is_crypto(ticker):
    return ticker.upper().endswith(("-USD", "-USDT", "-BTC", "-EUR"))


def fetch(ticker, lookback=400):
    """Daily candles from Yahoo Finance. Works for stocks (AAPL), ETFs (SPY), crypto (BTC-USD), FX (EURUSD=X)."""
    import yfinance as yf
    raw = yf.download(ticker, period="5y", interval="1d", auto_adjust=False, progress=False)
    if raw is None or len(raw) == 0:
        raise ValueError(f"No data for '{ticker}'. Check the symbol (examples: AAPL, NVDA, BTC-USD, ETH-USD, SPY).")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    idx = pd.to_datetime(raw.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = pd.DataFrame({
        "timestamps": idx,
        "open": raw["Open"].values.astype(float), "high": raw["High"].values.astype(float),
        "low": raw["Low"].values.astype(float), "close": raw["Close"].values.astype(float),
        "volume": raw["Volume"].values.astype(float),
    }).dropna().reset_index(drop=True)
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    # Drop today's candle if it is still forming (partial volume would skew the model's view).
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    if df["timestamps"].iloc[-1].normalize() >= today:
        df = df.iloc[:-1]
        print(f"  (skipped today's unfinished candle; forecasting from {df['timestamps'].iloc[-1].date()})")
    if len(df) < 60:
        raise ValueError(f"Only {len(df)} days of data for '{ticker}'. Need at least 60.")
    return df.tail(lookback).reset_index(drop=True)


def future_dates(last_ts, n, crypto):
    start = last_ts + pd.Timedelta(days=1)
    idx = pd.date_range(start, periods=n, freq="D") if crypto else pd.bdate_range(start, periods=n)
    return pd.Series(idx)


def forecast_paths(predictor, df, days_ahead=30, n_paths=20, temperature=1.0, top_p=0.9, crypto=False, chunk=5):
    """Run Kronos n_paths times in small batches (memory-safe). Returns (paths[n, days, 6], future_timestamps)."""
    x_ts = df["timestamps"]
    y_ts = future_dates(x_ts.iloc[-1], days_ahead, crypto)
    out = []
    done = 0
    while done < n_paths:
        b = min(chunk, n_paths - done)
        preds = predictor.predict_batch(
            df_list=[df[X_COLS]] * b, x_timestamp_list=[x_ts] * b, y_timestamp_list=[y_ts] * b,
            pred_len=days_ahead, T=temperature, top_p=top_p, sample_count=1, verbose=False)
        out.extend(p[X_COLS].values for p in preds)
        done += b
        print(f"  simulated {done}/{n_paths} paths", end="\r")
    print()
    return np.stack(out), y_ts


def summarize(df, paths, y_ts, ticker):
    closes = paths[:, :, 3]                      # close column -> (n_paths, days)
    last = float(df["close"].iloc[-1])
    end = closes[:, -1]
    med_path = np.median(closes, axis=0)
    s = {
        "ticker": ticker, "last_close": last, "horizon_days": closes.shape[1], "paths": closes.shape[0],
        "median_end": float(np.median(end)),
        "p10_end": float(np.percentile(end, 10)), "p90_end": float(np.percentile(end, 90)),
        "prob_up": float((end > last).mean()),
        "median_path": med_path,
        "p10": np.percentile(closes, 10, axis=0), "p90": np.percentile(closes, 90, axis=0),
        "p25": np.percentile(closes, 25, axis=0), "p75": np.percentile(closes, 75, axis=0),
        "dates": y_ts,
    }
    s["median_change"] = s["median_end"] / last - 1
    s["p10_change"] = s["p10_end"] / last - 1
    s["p90_change"] = s["p90_end"] / last - 1
    return s


def fmt_money(v):
    if v >= 1e5:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:,.2f}"
    return f"${v:,.4f}"


def plain_english(s):
    d = s["horizon_days"]
    return "\n".join([
        f"{s['ticker']}  |  last close {fmt_money(s['last_close'])}  |  {d} trading days ahead  |  {s['paths']} simulated paths",
        "",
        f"Middle-of-the-road outcome: {fmt_money(s['median_end'])} ({s['median_change']:+.1%})",
        f"80% of paths finish between {fmt_money(s['p10_end'])} ({s['p10_change']:+.1%}) and {fmt_money(s['p90_end'])} ({s['p90_change']:+.1%})",
        f"Share of paths that end higher than today: {s['prob_up']:.0%}",
        "",
        "Read this like a weather forecast, not a promise. A 60% 'higher' means 40% of the time it isn't.",
    ])


def plot_forecast(df, s, history_days=120, actual=None, title=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    hist = df.tail(history_days)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(hist["timestamps"], hist["close"], color="#1f2937", lw=1.6, label="Actual (history)")
    x = pd.to_datetime(s["dates"])
    x0 = pd.concat([pd.Series([hist["timestamps"].iloc[-1]]), pd.Series(x)]).values
    l0 = s["last_close"]
    ax.fill_between(x0, np.r_[l0, s["p10"]], np.r_[l0, s["p90"]], color="#3b82f6", alpha=0.18, label="80% range")
    ax.fill_between(x0, np.r_[l0, s["p25"]], np.r_[l0, s["p75"]], color="#3b82f6", alpha=0.32, label="50% range")
    ax.plot(x0, np.r_[l0, s["median_path"]], color="#1d4ed8", lw=2.0, label="Median forecast")
    if actual is not None:
        ax.plot(actual["timestamps"], actual["close"], color="#dc2626", lw=1.8, ls="--", label="What actually happened")
    ax.axvline(hist["timestamps"].iloc[-1], color="#9ca3af", lw=1, ls=":")
    ax.set_title(title or f"{s['ticker']}: Kronos forecast, next {s['horizon_days']} trading days", fontsize=13, loc="left")
    ax.set_ylabel("Close price"); ax.grid(alpha=0.25); ax.legend(loc="upper left", frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


def backtest(predictor, df, days_ahead=30, n_paths=20, crypto=False, temperature=1.0, top_p=0.9, chunk=5):
    """Hide the last `days_ahead` bars, forecast them, and score against what really happened."""
    train = df.iloc[:-days_ahead].reset_index(drop=True)
    actual = df.iloc[-days_ahead:].reset_index(drop=True)
    paths, _ = forecast_paths(predictor, train, days_ahead, n_paths, temperature, top_p, crypto, chunk)
    s = summarize(train, paths, actual["timestamps"], "Reality check")
    a_end = float(actual["close"].iloc[-1]); last = s["last_close"]
    inside = (actual["close"].values >= s["p10"]) & (actual["close"].values <= s["p90"])
    res = {
        "direction_right": bool((a_end > last) == (s["median_end"] > last)),
        "actual_change": a_end / last - 1, "median_change": s["median_change"],
        "end_inside_80": bool(s["p10_end"] <= a_end <= s["p90_end"]),
        "days_inside_80": float(inside.mean()),
        "mape_median": float(np.mean(np.abs(s["median_path"] - actual["close"].values) / actual["close"].values)),
    }
    return s, actual, train, res


def backtest_english(res, days):
    return "\n".join([
        f"Reality check: the last {days} trading days were hidden from the model, then forecast.",
        f"  Direction call: {'RIGHT' if res['direction_right'] else 'WRONG'}  (model said {res['median_change']:+.1%}, market did {res['actual_change']:+.1%})",
        f"  Final price landed inside the 80% range: {'YES' if res['end_inside_80'] else 'NO'}",
        f"  Days the real price stayed inside the 80% range: {res['days_inside_80']:.0%}",
        f"  Average miss of the median line: {res['mape_median']:.1%}",
        "",
        "One reality check is one coin flip. Run it on a few tickers before trusting the pattern.",
    ])
