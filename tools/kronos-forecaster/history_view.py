"""What the forecaster would have said in past years, drawn next to what actually happened.

For one ticker: every DAYS_AHEAD days, draw the range the forecaster would have given that day (using only
the prices known then), and mark whether the real price ended inside it. Below it, a bar for each year:
how often the wide range held, checked every 10 days.

Usage:  .venv/Scripts/python.exe history_view.py BTC-USD [DAYS_AHEAD]
        .venv/Scripts/python.exe history_view.py --years [DAYS_AHEAD]     (year-by-year table for every market tested)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import market_data as md
import range_model as rm
import range_v2

VERSION = 1 if "--v1" in sys.argv else 2          # 2 = the tightened range (range_v2.py), 1 = the original

HERE = os.path.dirname(os.path.abspath(__file__))
chosen = json.load(open(os.path.join(HERE, "out", "range", "chosen.json")))
SPLIT_YEAR = 2019

# chart colours (reference palette: blue ramp for the range, status colours for held / broke)
BLUE_WIDE, BLUE_TIGHT, BLUE_MID = "#cde2fb", "#86b6ef", "#256abf"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"


def kw_for(cls, h):
    key = f"{'fund' if cls == 'commodity' and f'commodity_{h}' not in chosen else cls}_{30 if h <= 45 else 90}"
    m, vol = chosen[key].split("/")
    k = float(vol.split(" x")[1]) if " x" in vol else 1.0
    return dict(vol_name=vol.split(" x")[0], method=m, k=k)


def quantiles(df, h, cls, ticker):
    if VERSION == 2:
        return range_v2.quantiles(df, ticker, h)[0]
    return rm.forecast(df["close"], h, **kw_for(cls, h))


def held_series(df, h, cls, ticker):
    """For every day: did the real price h rows later end inside the wide range (and the tight one)?"""
    c = df["close"]
    q = quantiles(df, h, cls, ticker)
    a = np.log(c.shift(-h) / c)
    ok = q.notna().all(axis=1) & a.notna()
    wide = (a >= q[0.10]) & (a <= q[0.90])
    tight = (a >= q[0.25]) & (a <= q[0.75])
    miss = np.where(a < q[0.10], a - q[0.10], np.where(a > q[0.90], a - q[0.90], 0.0))
    width = np.exp(q[0.90]) - np.exp(q[0.10])
    return pd.DataFrame({"date": df["timestamps"], "ok": ok, "wide": wide, "tight": tight, "a": a, "miss": miss, "width": width})


def one_ticker(ticker, h):
    df = md.load(ticker, max_age_hours=24, need_volume=False)
    cls = md.asset_class(ticker)
    c = df["close"].values
    ts = df["timestamps"]
    name = md.COMMODITY_NAMES.get(ticker, {"BTC-USD": "Bitcoin", "^GSPC": "S&P 500 index", "ETH-USD": "Ether"}.get(ticker, ticker))
    unit = "days" if cls == "crypto" else "trading days"

    steps = sorted(set([1, 5, 10, 20, 30, 45, 60, h]))
    steps = [s for s in steps if s <= h]
    qs = {s: quantiles(df, s, cls, ticker) for s in steps}
    hs = held_series(df, h, cls, ticker)

    first = int(np.argmax(hs["ok"].values))
    starts = list(range(first, len(df) - h, h))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(13, 8.6), gridspec_kw={"height_ratios": [3.2, 1.3], "hspace": 0.32})

    split_date = pd.Timestamp(f"{SPLIT_YEAR}-01-01")
    lo_date = ts.iloc[max(first - 30, 0)]
    ax.axvspan(lo_date, split_date, color="#f1f0ec", zorder=0, lw=0)
    ax.plot(ts, c, color=INK, lw=1.1, zorder=3)
    held_n = broke_n = 0
    for i in starts:
        d = [ts.iloc[i]] + [ts.iloc[i + s] for s in steps]
        p0 = c[i]
        lo80 = [p0] + [p0 * np.exp(qs[s][0.10].iloc[i]) for s in steps]
        hi80 = [p0] + [p0 * np.exp(qs[s][0.90].iloc[i]) for s in steps]
        lo50 = [p0] + [p0 * np.exp(qs[s][0.25].iloc[i]) for s in steps]
        hi50 = [p0] + [p0 * np.exp(qs[s][0.75].iloc[i]) for s in steps]
        ax.fill_between(d, lo80, hi80, color=BLUE_WIDE, alpha=0.85, lw=0, zorder=1)
        ax.fill_between(d, lo50, hi50, color=BLUE_TIGHT, alpha=0.75, lw=0, zorder=2)
        ok = bool(hs["wide"].iloc[i])
        held_n += ok; broke_n += (not ok)
        ax.plot(ts.iloc[i + h], c[i + h], marker="o" if ok else "X", ms=7 if ok else 9, color=GOOD if ok else CRITICAL,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5, ls="none")
    ax.set_yscale("log")
    ax.set_ylabel("Price (log scale)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2 if (ts.iloc[-1] - lo_date).days > 3650 else 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(lo_date, ts.iloc[-1] + pd.Timedelta(days=20))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ytop = ax.get_ylim()[1]
    if lo_date < split_date:
        ax.text(lo_date + (split_date - lo_date) / 2, ytop, "settings were picked on these years", ha="center", va="top",
                color=INK2, fontsize=9)
    ax.text(split_date + (ts.iloc[-1] - split_date) / 2, ytop, "never seen by the settings", ha="center", va="top", color=INK2, fontsize=9)
    total = held_n + broke_n
    ax.set_title(f"{name}: the range the forecaster{' (tightened version)' if VERSION == 2 else ''} would have drawn every {h} {unit}, and where price really ended",
                 loc="left", fontsize=13, color=INK)
    leg = [Line2D([0], [0], color=INK, lw=1.2, label="Actual price"),
           Patch(color=BLUE_WIDE, label="Wide range (aims to hold 80% of the time)"),
           Patch(color=BLUE_TIGHT, label="Tight range (aims to hold 50%)"),
           Line2D([0], [0], marker="o", color=GOOD, ls="none", ms=7, label=f"Ended inside the wide range ({held_n} of {total})"),
           Line2D([0], [0], marker="X", color=CRITICAL, ls="none", ms=9, label=f"Ended outside it ({broke_n} of {total})")]
    ax.legend(handles=leg, loc="upper left", frameon=False, fontsize=9, bbox_to_anchor=(0, 0.95))

    # year-by-year hold rate, checked every 10 days
    y = hs[hs["ok"]].iloc[::10].copy()
    y["year"] = y["date"].dt.year
    yr = y.groupby("year").agg(n=("wide", "size"), held=("wide", "mean"), tight=("tight", "mean")).reset_index()
    yr = yr[yr["n"] >= 12]          # skip part-years with too few checks to mean anything
    colors = [BLUE_MID if v >= SPLIT_YEAR else "#9ec5f4" for v in yr["year"]]
    bx.bar(yr["year"], 100 * yr["held"], color=colors, width=0.72, zorder=2)
    bx.axhline(80, color=INK2, lw=1, ls=(0, (4, 3)), zorder=3)
    bx.set_ylim(0, 105)
    bx.set_xlim(yr["year"].min() - 0.6, yr["year"].max() + 0.6)
    bx.set_ylabel("Wide range held (%)")
    bx.set_xticks(yr["year"])
    bx.set_xticklabels([str(v)[2:] if len(yr) > 16 else str(v) for v in yr["year"]])
    bx.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    worst = yr.loc[yr["held"].idxmin()]
    bx.annotate(f"{int(worst['year'])}: {worst['held']:.0%}", xy=(worst["year"], 100 * worst["held"]), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
    bx.set_title("Each year: how often the wide range held, checked every 10 days (dashed line = the 80% it aims for; light bars = years the settings were picked on" + ("; the new version's final stretch used all years)" if VERSION == 2 else ")"),
                 loc="left", fontsize=10, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.06)

    out_dir = os.path.join(HERE, "out", "history")
    os.makedirs(out_dir, exist_ok=True)
    safe = ticker.replace("^", "_").replace("=", "_") + ("" if VERSION == 2 else "_v1")
    png = os.path.join(out_dir, f"{safe}_{h}d_history.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)

    fans = pd.DataFrame([{"from": ts.iloc[i].date(), "to": ts.iloc[i + h].date(), "start_price": c[i], "end_price": c[i + h],
                          "held": bool(hs["wide"].iloc[i]),
                          "low": c[i] * np.exp(qs[h][0.10].iloc[i]), "high": c[i] * np.exp(qs[h][0.90].iloc[i])} for i in starts])
    fans.to_csv(os.path.join(out_dir, f"{safe}_{h}d_forecasts.csv"), index=False)
    yr.to_csv(os.path.join(out_dir, f"{safe}_{h}d_by_year.csv"), index=False)
    all_ = hs[hs["ok"]].iloc[::10]
    summary = {"ticker": ticker, "name": name, "horizon": h, "fans_drawn": total, "fans_held": int(held_n),
               "held_every_10_days": round(float(all_["wide"].mean()), 4), "tight_every_10_days": round(float(all_["tight"].mean()), 4),
               "held_2019_on": round(float(all_[all_["date"].dt.year >= SPLIT_YEAR]["wide"].mean()), 4),
               "avg_width_pct": round(float(100 * all_["width"].mean()), 2),
               "avg_width_pct_2019_on": round(float(100 * all_[all_["date"].dt.year >= SPLIT_YEAR]["width"].mean()), 2),
               "version": VERSION,
               "worst_year": int(worst["year"]), "worst_year_held": round(float(worst["held"]), 4),
               "broke_low": int((all_["miss"] < 0).sum()), "broke_high": int((all_["miss"] > 0).sum()), "checks": int(len(all_)),
               "chart": png}
    print(json.dumps(summary, indent=2, default=str))
    print(fans.assign(start_price=fans["start_price"].round(2), end_price=fans["end_price"].round(2), low=fans["low"].round(2),
                      high=fans["high"].round(2)).tail(16).to_string(index=False))
    return summary


def years_table(h):
    """How often the wide range held, year by year, on every market tested."""
    groups = {"S&P 500 stocks (471)": md.universe("stocks"), "Commodity futures (26)": md.universe("commodities"),
              "Indexes and sector funds (18)": md.universe("indexes"), "Crypto (3)": md.UNIVERSE["crypto"]}
    rows = []
    for g, tickers in groups.items():
        for t in tickers:
            try:
                df = md.load(t, max_age_hours=96, need_volume=False)
            except Exception:
                continue
            if len(df) < 1000:
                continue
            hs = held_series(df, h, md.asset_class(t), t)
            x = hs[hs["ok"]].iloc[::10]
            rows.append(pd.DataFrame({"group": g, "year": x["date"].dt.year.values, "wide": x["wide"].values, "tight": x["tight"].values}))
    d = pd.concat(rows, ignore_index=True)
    t = d.pivot_table(index="year", columns="group", values="wide", aggfunc="mean")
    n = d.pivot_table(index="year", columns="group", values="wide", aggfunc="size")
    t = (100 * t).round(0)
    t[n < 20] = np.nan
    t["All markets"] = (100 * d.groupby("year")["wide"].mean()).round(0)
    out_dir = os.path.join(HERE, "out", "history")
    os.makedirs(out_dir, exist_ok=True)
    t.to_csv(os.path.join(out_dir, f"all_markets_{h}d_by_year.csv"))
    pd.set_option("display.width", 200)
    print(t.to_string())
    return t


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--v1"]
    if args and args[0] == "--years":
        years_table(int(args[1]) if len(args) > 1 else 90)
    else:
        one_ticker((args[0] if args else "BTC-USD").upper(), int(args[1]) if len(args) > 1 else 90)
