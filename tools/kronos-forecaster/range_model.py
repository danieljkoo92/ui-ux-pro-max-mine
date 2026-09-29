"""The honest range: where price is likely to be N candles from now, built from how much it really moves.

No direction call. It answers one question: given how much this ticker has been swinging, what band holds
the price 80% of the time (and 50% of the time) N days out? Run as a script, it tests several ways of
building that band on real history and reports how often each band actually held the price.

Usage:  .venv/Scripts/python.exe range_model.py
"""
import json
import os

import numpy as np
import pandas as pd

import market_data as md

HERE = os.path.dirname(os.path.abspath(__file__))
QS = [0.10, 0.25, 0.50, 0.75, 0.90]
Z = {0.10: -1.2816, 0.25: -0.6745, 0.50: 0.0, 0.75: 0.6745, 0.90: 1.2816}
SPLIT = pd.Timestamp("2019-01-01")
MIN_HISTORY = 500        # past samples needed before a ticker's own history is trusted


def daily_vol(close):
    """Several readings of how much the price moves per day (as a fraction), all using past data only."""
    r = np.log(close / close.shift(1))
    v20, v60, v250 = r.rolling(20).std(), r.rolling(60).std(), r.rolling(250).std()
    ewma = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean())
    return pd.DataFrame({
        "v20": v20, "v60": v60, "v250": v250, "ewma": ewma,
        "blend3": np.sqrt((v20 ** 2 + v60 ** 2 + v250 ** 2) / 3),
        "blend_60_250": np.sqrt((v60 ** 2 + v250 ** 2) / 2),
    })


def own_history_quantiles(close, vol, h):
    """For each day: quantiles of this ticker's own past h-day moves, measured in units of its swing size at the time.

    Only moves that had already finished by that day are used, so nothing from the future leaks in.
    """
    z = np.log(close.shift(-h) / close) / (vol * np.sqrt(h))
    known = z.shift(h)                                  # the move that started h days ago has just finished
    out = {}
    for q in QS:
        out[q] = known.expanding(min_periods=MIN_HISTORY).quantile(q)
    return pd.DataFrame(out)


def forecast(close, h, vol_name="blend3", method="mix", k=1.0):
    """Quantiles of the h-day log return, for every day. method: normal | own | mix (half and half)."""
    vol = daily_vol(close)[vol_name]
    sig = vol * np.sqrt(h) * k
    normal = pd.DataFrame({q: Z[q] * sig for q in QS})
    if method == "normal":
        return normal
    own = own_history_quantiles(close, vol, h).mul(sig, axis=0)
    own = own.where(own.notna(), normal)               # too little history: fall back to the plain band
    if method == "own":
        return own
    mix = (normal + own) / 2
    if method == "mix_own_center":                     # width from the mix, middle from the ticker's own history
        return mix.sub(mix[0.50], axis=0).add(own[0.50], axis=0)
    return mix


def prob_above(quantiles, x):
    """Chance the h-day log return ends above x, reading between the five forecast quantiles."""
    from math import erf, sqrt
    zs = np.array([Z[q] for q in QS])
    rs = np.array([float(quantiles[q]) for q in QS])
    if x <= rs[0]:
        z = zs[0] + (x - rs[0]) * (zs[1] - zs[0]) / (rs[1] - rs[0])
    elif x >= rs[-1]:
        z = zs[-1] + (x - rs[-1]) * (zs[-1] - zs[-2]) / (rs[-1] - rs[-2])
    else:
        z = float(np.interp(x, rs, zs))
    return 1 - 0.5 * (1 + erf(z / sqrt(2)))


def pinball(actual, qhat, q):
    d = actual - qhat
    return np.where(d >= 0, q * d, (q - 1) * d)


def evaluate():
    variants = []
    for vol_name in ["v20", "v60", "v250", "ewma", "blend3", "blend_60_250"]:
        variants.append((f"normal/{vol_name}", dict(vol_name=vol_name, method="normal")))
    for vol_name in ["blend3", "blend_60_250", "v60"]:
        variants.append((f"own/{vol_name}", dict(vol_name=vol_name, method="own")))
        variants.append((f"mix/{vol_name}", dict(vol_name=vol_name, method="mix")))
        variants.append((f"mix_own_center/{vol_name}", dict(vol_name=vol_name, method="mix_own_center")))
    for k in (1.1, 1.2):
        variants.append((f"normal/blend3 x{k}", dict(vol_name="blend3", method="normal", k=k)))

    tickers = [t for v in md.UNIVERSE.values() for t in v]
    hist = {t: md.load(t) for t in tickers}
    rows = []
    for h in (30, 90):
        for name, kw in variants:
            for t in tickers:
                df = hist[t]
                c = df["close"]
                qf = forecast(c, h, **kw)
                actual = np.log(c.shift(-h) / c)
                end = df["timestamps"].shift(-h)
                grid = np.arange(260, len(df) - h, 10)
                x = pd.DataFrame({"date": df["timestamps"].iloc[grid].values, "end": end.iloc[grid].values,
                                  "actual": actual.iloc[grid].values})
                for q in QS:
                    x[q] = qf[q].iloc[grid].values
                x = x.dropna()
                x["split"] = np.where(x["end"] < SPLIT, "tune", np.where(x["date"] >= SPLIT, "holdout", "gap"))
                x = x[x["split"] != "gap"]
                x["inside80"] = (x["actual"] >= x[0.10]) & (x["actual"] <= x[0.90])
                x["inside50"] = (x["actual"] >= x[0.25]) & (x["actual"] <= x[0.75])
                x["above_mid"] = x["actual"] > x[0.50]
                x["below10"] = x["actual"] < x[0.10]
                x["above90"] = x["actual"] > x[0.90]
                x["loss"] = sum(pinball(x["actual"].values, x[q].values, q) for q in QS) / len(QS)
                x["width80"] = x[0.90] - x[0.10]
                x["variant"], x["h"], x["ticker"], x["cls"] = name, h, t, md.asset_class(t)
                rows.append(x[["variant", "h", "ticker", "cls", "split", "inside80", "inside50", "above_mid", "below10",
                               "above90", "loss", "width80"]])
    allr = pd.concat(rows, ignore_index=True)
    agg = allr.groupby(["h", "variant", "split"]).agg(
        n=("loss", "size"), inside80=("inside80", "mean"), inside50=("inside50", "mean"), above_mid=("above_mid", "mean"),
        below10=("below10", "mean"), above90=("above90", "mean"), loss=("loss", "mean"), width80=("width80", "mean")).reset_index()
    by_cls = allr.groupby(["h", "variant", "split", "cls"]).agg(
        n=("loss", "size"), inside80=("inside80", "mean"), inside50=("inside50", "mean"), above_mid=("above_mid", "mean"),
        loss=("loss", "mean")).reset_index()
    out_dir = os.path.join(HERE, "out", "range")
    os.makedirs(out_dir, exist_ok=True)
    agg.to_csv(os.path.join(out_dir, "range_report.csv"), index=False)
    by_cls.to_csv(os.path.join(out_dir, "range_report_by_class.csv"), index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    for h in (30, 90):
        print(f"\n=== {h} candles ahead, TUNE years (pick the variant here) ===")
        print(agg[(agg.h == h) & (agg.split == "tune")].sort_values("loss").round(4).to_string(index=False))
    # One choice per kind of ticker and look-ahead, made on the tune years only, by the standard score for
    # range forecasts (pinball loss: it rewards ranges that are both right and tight). Holdout is only read.
    best = {}
    for cls in sorted(by_cls["cls"].unique()):
        for h in (30, 90):
            a = by_cls[(by_cls.h == h) & (by_cls.split == "tune") & (by_cls.cls == cls) & (by_cls.n >= 100)]
            if not len(a):
                a = agg[(agg.h == h) & (agg.split == "tune")]
            v = a.sort_values("loss").iloc[0]["variant"]
            best[f"{cls}_{h}"] = v
            print(f"\n=== {cls}, {h} candles ahead: chosen on tune = {v}; its HOLDOUT result ===")
            print(by_cls[(by_cls.h == h) & (by_cls.variant == v) & (by_cls.cls == cls)].round(4).to_string(index=False))
    with open(os.path.join(out_dir, "chosen.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)


if __name__ == "__main__":
    evaluate()
