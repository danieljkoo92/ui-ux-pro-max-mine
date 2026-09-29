"""The fixed forecaster: an honest range, tested, with no direction call it cannot back up.

What changed from the Kronos notebook and why:
  - Kronos on daily candles called direction at coin-flip accuracy, missed by about twice the "no change"
    guess, and its "80% range" held the real price 9% to 22% of the time (out/diagnose).
  - 99 chart rules from two videos and the Murphy book were tested on 30 tickers of daily history.
    None beat luck on both the tune years and the holdout years (out/signals).
  - A range built from how much each ticker really moves held the price 80% of the time when it said 80%,
    on years it was never tuned on (out/range).
So the forecast is the range. It says how far, not which way.

Usage:  .venv/Scripts/python.exe forecast.py TICKER [DAYS_AHEAD] [--level PRICE ...]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import market_data as md
import range_model as rm

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
TICKER = (args[0] if args else "BTC-USD").upper()
H = int(args[1]) if len(args) > 1 else 90
LEVELS = []
if "--level" in sys.argv:
    i = sys.argv.index("--level")
    for a in sys.argv[i + 1:]:
        if a.startswith("--"):
            break
        LEVELS.append(float(a))

chosen = json.load(open(os.path.join(HERE, "out", "range", "chosen.json")))
touch = json.load(open(os.path.join(HERE, "out", "range", "touch.json")))
by_cls = pd.read_csv(os.path.join(HERE, "out", "range", "range_report_by_class.csv"))
METHODS = {"mix": "mix", "mix_own_center": "mix_own_center", "normal": "normal", "own": "own"}


def method_for(h):
    key = "30" if h <= 45 else "90"
    name = chosen[f"{md.asset_class(TICKER)}_{key}"]
    m, vol = name.split("/")
    k = float(vol.split(" x")[1]) if " x" in vol else 1.0
    return key, name, dict(vol_name=vol.split(" x")[0], method=METHODS[m], k=k)


def money(v):
    if v >= 1e5:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:,.2f}"
    return f"${v:,.4f}"


df = md.load(TICKER, max_age_hours=1, need_volume=False)
c = df["close"]
last = float(c.iloc[-1])
last_day = df["timestamps"].iloc[-1]
cls = md.asset_class(TICKER)
crypto = cls == "crypto"

key, variant, kw = method_for(H)
q_now = rm.forecast(c, H, **kw).iloc[-1]
if q_now.isna().any():
    raise SystemExit(f"Not enough history for {TICKER}: {len(df)} days.")
px = {q: last * float(np.exp(q_now[q])) for q in rm.QS}
p_up = rm.prob_above(q_now, 0.0)

tk = touch["all_" + key + "_tune"]          # measured across every ticker: the per-class samples are too small
if cls == "commodity":
    # commodity futures were judged separately (range_universe.py): settings picked on their own older years
    cu = json.load(open(os.path.join(HERE, "out", "range", "universe_commodities.json")))[f"own_{key}"]
    r_ = [x for x in cu["result"] if x["split"] == "holdout"][0]
    held80, held50, n_tests = r_["inside80"], r_["inside50"], int(r_["n"])
    tk = {"touch_vs_end_down": cu["touch_vs_end_down"], "touch_vs_end_up": cu["touch_vs_end_up"]}
else:
    rec = by_cls[(by_cls.h == int(key)) & (by_cls.variant == variant) & (by_cls.split == "holdout") & (by_cls.cls == cls)]
    held80 = float(rec["inside80"].iloc[0]) if len(rec) else float("nan")
    held50 = float(rec["inside50"].iloc[0]) if len(rec) else float("nan")
    n_tests = int(rec["n"].iloc[0]) if len(rec) else 0

# the fan: the same forecast at shorter look-aheads, for the chart
steps = sorted(set([1, 5, 10, 20, 30, 45, 60, 90, H]))
steps = [s for s in steps if s <= H]
fan = {}
for s in steps:
    _, _, kws = method_for(s)
    qs = rm.forecast(c, s, **kws).iloc[-1]
    if qs.isna().any():
        qs = rm.forecast(c, s, vol_name=kws["vol_name"], method="normal").iloc[-1]
    fan[s] = {q: last * float(np.exp(qs[q])) for q in rm.QS}
grid = np.arange(0, H + 1)
curve = {q: np.interp(grid, [0] + steps, [last] + [fan[s][q] for s in steps]) for q in rm.QS}
start = last_day + pd.Timedelta(days=1)
dates = pd.date_range(start, periods=H, freq="D") if crypto else pd.bdate_range(start, periods=H)
dates = pd.DatetimeIndex([last_day]).append(dates)

unit = "days" if crypto else "trading days"
label = md.COMMODITY_NAMES.get(TICKER, TICKER)
label = f"{label} ({TICKER})" if label != TICKER else TICKER
kind = {"commodity": "commodity futures", "fund": "fund and index", "stock": "stock", "crypto": "crypto"}[cls]
lines = [
    f"{label}  |  last close {money(last)} on {last_day.date()}  |  {H} {unit} ahead (to about {dates[-1].date()})",
    "",
    f"Wide range (aims to hold 80% of the time, has held {held80:.0%}):  {money(px[0.10])} to {money(px[0.90])}   ({px[0.10] / last - 1:+.1%} to {px[0.90] / last - 1:+.1%})",
    f"Tight range (aims to hold 50% of the time, has held {held50:.0%}): {money(px[0.25])} to {money(px[0.75])}   ({px[0.25] / last - 1:+.1%} to {px[0.75] / last - 1:+.1%})",
    f"Middle of the range: {money(px[0.50])}   ({px[0.50] / last - 1:+.1%})",
    f"Chance it ends higher than today: {p_up:.0%}",
]
level_out = []
for lv in LEVELS:
    x = float(np.log(lv / last))
    if lv < last:
        p_end = 1 - rm.prob_above(q_now, x)
        p_touch = min(1.0, p_end * tk["touch_vs_end_down"])
        lines.append(f"Chance it ends below {money(lv)} ({lv / last - 1:+.1%}): {p_end:.0%}.  Chance it touches it at any point: about {p_touch:.0%}")
    else:
        p_end = rm.prob_above(q_now, x)
        p_touch = min(1.0, p_end * tk["touch_vs_end_up"])
        lines.append(f"Chance it ends above {money(lv)} ({lv / last - 1:+.1%}): {p_end:.0%}.  Chance it touches it at any point: about {p_touch:.0%}")
    level_out.append({"level": lv, "chance_ends_beyond": round(p_end, 4), "chance_touches": round(p_touch, 4)})
lines += [
    "",
    f"\"Has held\" is measured on {n_tests:,} past {kind} forecasts from 2019 to today, years the range was never tuned on.",
    "Direction: no call. 99 chart rules were tested for one and none beat luck twice.",
    "This is a measured range, not advice to buy or sell.",
]
if cls == "commodity":
    lines.insert(-1, "Futures note: this is the front contract's price. When that contract expires the series jumps to the next one.")
text = "\n".join(lines)
print(text)

out_dir = os.path.join(HERE, "out", f"{TICKER.replace(chr(94), chr(95)).replace(chr(61), chr(95))}_{H}d_range")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "forecast.txt"), "w", encoding="utf-8") as f:
    f.write(text + "\n")
with open(os.path.join(out_dir, "forecast.json"), "w", encoding="utf-8") as f:
    json.dump({"ticker": TICKER, "as_of": str(last_day.date()), "horizon": H, "unit": unit, "last_close": last,
               "prices": {str(q): round(v, 4) for q, v in px.items()}, "chance_ends_higher": round(p_up, 4),
               "levels": level_out, "method": variant,
               "track_record": {"class": cls, "held_80": held80, "held_50": held50, "forecasts": n_tests}}, f, indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

hist = df.tail(250)
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.plot(hist["timestamps"], hist["close"], color="#1f2937", lw=1.6, label="Actual price")
ax.fill_between(dates, curve[0.10], curve[0.90], color="#3b82f6", alpha=0.18, label=f"Wide range (has held {held80:.0%} of the time)")
ax.fill_between(dates, curve[0.25], curve[0.75], color="#3b82f6", alpha=0.32, label=f"Tight range (has held {held50:.0%} of the time)")
ax.plot(dates, curve[0.50], color="#1d4ed8", lw=1.6, ls="--", label="Middle of the range")
ax.axvline(last_day, color="#9ca3af", lw=1, ls=":")
for lv in LEVELS:
    ax.axhline(lv, color="#dc2626", lw=1, ls=":")
    ax.annotate(money(lv), xy=(hist["timestamps"].iloc[0], lv), xytext=(4, 4), textcoords="offset points", color="#dc2626", fontsize=9)
for q, lab in ((0.90, "top of likely range"), (0.10, "bottom of likely range")):
    ax.annotate(f"{money(px[q])}", xy=(dates[-1], px[q]), xytext=(6, -3), textcoords="offset points", fontsize=9, color="#1d4ed8")
ax.set_title(f"{label}: where price is likely to be over the next {H} {unit}", fontsize=13, loc="left")
ax.set_ylabel("Price")
ax.grid(alpha=0.25)
ax.legend(loc="upper left", frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.margins(x=0.06)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "forecast.png"), dpi=150)
print(f"\nSaved to {out_dir}")
