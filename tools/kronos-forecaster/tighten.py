"""Can the range be made narrower and still hold 80%?

Every candidate below is fitted on the older years only (forecasts that finished before 2019) and then
judged once on 2019 to today. A candidate wins only if it scores better than the current forecaster on the
newer years AND its wide range still holds close to 80%.

The score is the pinball loss (average over the five range edges). It rewards a range for being right and
for being narrow at the same time, so a range cannot win by simply being wider or simply being narrower.

Candidates, in the order from the reading list:
  base        the forecaster exactly as it runs today
  recal       today's range, stretched or shrunk around its middle by one factor per market type
  close       swings measured from closing prices (today's measure) with a shape fitted per market type
  ohlc        swings measured from each day's open, high, low and close (Yang and Zhang, 2000)
  implied     the options market's forecast (VIX, VXN, VXD, GVZ, OVX, DVOL) blended with past swings
  regime      past swings scaled by how jumpy the options market expects the whole market to be
  adaptive    the best of the above, widened after recent misses and narrowed after hits (Gibbs and Candes)

Usage:  .venv/Scripts/python.exe tighten.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

import market_data as md
import range_model as rm
import vol_data as vd

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
EXPERIMENT = "--dry" in sys.argv          # extra candidates (3-month VIX, fitted blend) tested 2026-09-29: no gain
OUT = os.path.join(HERE, "out", "tighten_exp" if EXPERIMENT else "tighten")      # --dry: never touches live settings
os.makedirs(OUT, exist_ok=True)
QS = rm.QS
ZQ = np.array([rm.Z[q] for q in QS])
SPLIT = pd.Timestamp("2019-01-01")
DVOL_SPLIT = pd.Timestamp("2024-01-01")        # crypto options data only starts in 2021
HS = (30, 90)
STEP = 5
chosen = json.load(open(os.path.join(HERE, "out", "range", "chosen.json")))
t0 = time.time()


def base_kw(cls, h):
    key = f"{cls}_{30 if h <= 45 else 90}"
    m, vol = chosen[key].split("/")
    k = float(vol.split(" x")[1]) if " x" in vol else 1.0
    return dict(vol_name=vol.split(" x")[0], method=m, k=k)


def yang_zhang(df):
    lo, lh, ll, lc = (np.log(df[x]) for x in ("open", "high", "low", "close"))
    over = lo - lc.shift(1)
    oc = lc - lo
    u, d = lh - lo, ll - lo
    rs = u * (u - oc) + d * (d - oc)
    out = {}
    for n in (20, 60, 250):
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        var = over.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean()
        out[n] = np.sqrt(var.clip(lower=0))
    return pd.DataFrame({
        "v20": out[20], "v60": out[60], "v250": out[250],
        "blend3": np.sqrt((out[20] ** 2 + out[60] ** 2 + out[250] ** 2) / 3),
        "blend_60_250": np.sqrt((out[60] ** 2 + out[250] ** 2) / 2),
        "ewma": out[60],
    })


def align(series, dates, limit=5):
    s = series[~series.index.duplicated()].sort_index()
    return s.reindex(s.index.union(dates)).ffill(limit=limit).reindex(dates).values


# ---------- the market each volatility index belongs to, for the "regime" reading
REGIME = {"stock": ("^VIX", "^GSPC"), "fund": ("^VIX", "^GSPC")}
REGIME_COMMODITY = {"energy": ("^OVX", "CL=F"), "metals": ("^GVZ", "GC=F")}
vol_idx = {}
for name in vd.YAHOO + ["DVOL-BTC", "DVOL-ETH"]:
    try:
        vol_idx[name] = vd.load(name)
    except Exception:
        pass
ref_vol = {}
for ref in ("^GSPC", "CL=F", "GC=F"):
    d = md.load(ref, max_age_hours=96, need_volume=False)
    ref_vol[ref] = pd.Series(rm.daily_vol(d["close"])["blend_60_250"].values, index=pd.DatetimeIndex(d["timestamps"]))

tickers = []
for u in ("original", "stocks", "indexes", "commodities"):
    for t in md.universe(u):
        if t not in tickers:
            tickers.append(t)

rows = []
daily = {}
for n_, t in enumerate(tickers):
    try:
        df = md.load(t, max_age_hours=96, need_volume=False)
    except Exception:
        continue
    if len(df) < 1000:
        continue
    cls = md.asset_class(t)
    group = md.commodity_group(t) or cls
    dates = pd.DatetimeIndex(df["timestamps"])
    c = df["close"]
    ann = np.sqrt(365.0 if cls == "crypto" else 252.0)
    cc_all = rm.daily_vol(c)
    yz_all = yang_zhang(df)
    iv_name = vd.DIRECT.get(t)
    iv = align(vol_idx[iv_name], dates) / 100 / ann if iv_name in vol_idx and iv_name != "^MOVE" else np.full(len(df), np.nan)
    reg = None
    if cls in REGIME:
        reg = REGIME[cls]
    elif group in REGIME_COMMODITY:
        reg = REGIME_COMMODITY[group]
    if reg and reg[0] in vol_idx:
        m = align(vol_idx[reg[0]], dates) / 100 / np.sqrt(252.0) / align(ref_vol[reg[1]], dates)
    else:
        m = np.full(len(df), np.nan)
    if cls in REGIME and "^VIX3M" in vol_idx:
        m3 = align(vol_idx["^VIX3M"], dates) / 100 / np.sqrt(252.0) / align(ref_vol["^GSPC"], dates)
    else:
        m3 = np.full(len(df), np.nan)
    lc = np.log(c.values)
    daily[t] = {"cls": cls, "group": group, "dates": dates, "lc": lc, "iv": iv, "m": m,
                "cc": {}, "yz": {}}
    for h in HS:
        kw = base_kw(cls, h)
        q = rm.forecast(c, h, **kw)
        sig_cc = cc_all[kw["vol_name"]].values
        sig_yz = yz_all[kw["vol_name"]].values
        daily[t]["cc"][h], daily[t]["yz"][h] = sig_cc, sig_yz
        r = np.full(len(df), np.nan)
        r[:-h] = lc[h:] - lc[:-h]
        end = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
        end[:-h] = dates.values[h:]
        idx = np.arange(260, len(df) - h, STEP)
        x = pd.DataFrame({"ticker": t, "cls": cls, "group": group, "h": h, "i": idx, "date": dates.values[idx], "end": end[idx],
                          "r": r[idx], "cc": sig_cc[idx], "yz": sig_yz[idx], "iv": iv[idx], "m": m[idx],
                          "direct": iv_name is not None and iv_name != "^MOVE" and iv_name in vol_idx, "m3": m3[idx]})
        for nm_, src in (("cc", cc_all), ("yz", yz_all)):
            for w_ in ("v20", "v60", "v250"):
                x[f"{nm_}_{w_}"] = src[w_].values[idx]
        for j, qq in enumerate(QS):
            x[f"b{j}"] = q[qq].values[idx]
        rows.append(x)
    if (n_ + 1) % 50 == 0:
        print(f"  {n_ + 1}/{len(tickers)} tickers, {time.time() - t0:.0f}s", flush=True)

d = pd.concat(rows, ignore_index=True)
d = d[np.isfinite(d["r"]) & np.isfinite(d["cc"]) & (d["cc"] > 0) & d[[f"b{j}" for j in range(5)]].notna().all(axis=1)]
d["split"] = np.where(d["end"] < SPLIT, "tune", np.where(d["date"] >= SPLIT, "holdout", "gap"))
d = d[d["split"] != "gap"].reset_index(drop=True)
print(f"{len(daily)} tickers, {len(d):,} forecasts, {time.time() - t0:.0f}s", flush=True)


def pinball(a, Q):
    """Q: (n, 5) forecast edges in log terms. Returns per-row mean pinball loss."""
    diff = a[:, None] - Q
    tau = np.array(QS)[None, :]
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff).mean(axis=1)


def stats(a, Q):
    return {"n": int(len(a)), "held80": float(((a >= Q[:, 0]) & (a <= Q[:, 4])).mean()),
            "held50": float(((a >= Q[:, 1]) & (a <= Q[:, 3])).mean()),
            "width80_pct": float((100 * (np.exp(Q[:, 4]) - np.exp(Q[:, 0]))).mean()),
            "score": float(pinball(a, Q).mean())}


def shape_fit(z_tune):
    """Pick a range shape on the older years: bell curve, the market type's own past moves, or the mix."""
    emp = np.quantile(z_tune, QS)
    return {"normal": ZQ, "empirical": emp, "mix": (ZQ + emp) / 2}


def edges(sig, h, shape, k=1.0):
    return k * (sig * np.sqrt(h))[:, None] * shape[None, :]


def fit_scaled(x, sig_col_fn, h, extra=None):
    """For a volatility series: choose shape and scale on tune, return edges for every row and the choice."""
    sig = sig_col_fn(x)
    ok = np.isfinite(sig) & (sig > 0)
    tune = (x["split"].values == "tune") & ok
    a = x["r"].values
    z = a[tune] / (sig[tune] * np.sqrt(h))
    best = None
    for sname, shape in shape_fit(z).items():
        for k in np.round(np.arange(0.8, 1.21, 0.02), 2):
            Q = edges(sig[tune], h, shape, k)
            s = pinball(a[tune], Q).mean()
            if best is None or s < best[0]:
                best = (s, sname, k, shape)
    _, sname, k, shape = best
    Q = np.full((len(x), 5), np.nan)
    Q[ok] = edges(sig[ok], h, shape, k)
    return Q, {"shape": sname, "scale": float(k), "shape_z": [float(v) for v in shape]}


results, choices, best_static, allQ = [], {}, {}, {}
for cls in ["stock", "fund", "commodity", "crypto"]:
    for h in HS:
        x = d[(d["cls"] == cls) & (d["h"] == h)].reset_index(drop=True)
        if not len(x):
            continue
        a = x["r"].values
        B = x[[f"b{j}" for j in range(5)]].values
        cand = {"base": (B, {})}
        # recal: stretch today's range around its middle
        tune = x["split"].values == "tune"
        mid = B[:, 2:3]
        best = min(((pinball(a[tune], mid[tune] + k * (B[tune] - mid[tune])).mean(), k) for k in np.round(np.arange(0.8, 1.21, 0.01), 2)))
        cand["recal"] = (mid + best[1] * (B - mid), {"stretch": float(best[1])})
        cand["close"] = fit_scaled(x, lambda y: y["cc"].values, h)
        yz = x["yz"].values
        cand["ohlc"] = fit_scaled(x, lambda y: np.where(np.isfinite(y["yz"].values) & (y["yz"].values > 0), y["yz"].values, y["cc"].values), h)
        # the better past-swings measure on tune becomes the base for the implied and regime blends
        hist_name = min(("close", "ohlc"), key=lambda nm: pinball(a[tune & np.isfinite(cand[nm][0][:, 0])],
                                                                    cand[nm][0][tune & np.isfinite(cand[nm][0][:, 0])]).mean())
        hist = x["cc"].values if hist_name == "close" else np.where(np.isfinite(yz) & (yz > 0), yz, x["cc"].values)
        # implied: blend options-market volatility with past swings, only where a matching index exists
        iv = x["iv"].values
        if cls != "crypto" and np.isfinite(iv[tune]).sum() > 500:
            bestb = None
            for w in (0.25, 0.5, 0.75, 1.0):
                for av in (0.6, 0.7, 0.8, 0.9, 1.0):
                    sig = np.where(np.isfinite(iv), np.sqrt(w * (av * iv) ** 2 + (1 - w) * hist ** 2), hist)
                    Q, info = fit_scaled(x.assign(_s=sig), lambda y: y["_s"].values, h)
                    s = pinball(a[tune], Q[tune]).mean()
                    if bestb is None or s < bestb[0]:
                        bestb = (s, Q, {**info, "weight_on_implied": w, "implied_scale": av})
            cand["implied"] = (bestb[1], bestb[2])
        m = x["m"].values
        if np.isfinite(m[tune]).sum() > 500:
            bestr = None
            for beta in (0.25, 0.5, 0.75, 1.0):
                sig = np.where(np.isfinite(m), hist * np.clip(m, 0.3, 5.0) ** beta, hist)
                Q, info = fit_scaled(x.assign(_s=sig), lambda y: y["_s"].values, h)
                s = pinball(a[tune], Q[tune]).mean()
                if bestr is None or s < bestr[0]:
                    bestr = (s, Q, {**info, "regime_power": beta})
            cand["regime"] = (bestr[1], bestr[2])
            beta_r = bestr[2]["regime_power"]
            # regime read from the 3-month VIX instead of the 30-day one (90-day ranges only)
            m3 = x["m3"].values
            if EXPERIMENT and h >= 60 and np.isfinite(m3[tune]).sum() > 500:
                best3 = None
                for beta in (0.25, 0.5, 0.75, 1.0):
                    sig = np.where(np.isfinite(m3), hist * np.clip(m3, 0.3, 5.0) ** beta, np.where(np.isfinite(m), hist * np.clip(m, 0.3, 5.0) ** beta, hist))
                    Q, info = fit_scaled(x.assign(_s=sig), lambda y: y["_s"].values, h)
                    s_ = pinball(a[tune], Q[tune]).mean()
                    if best3 is None or s_ < best3[0]:
                        best3 = (s_, Q, {**info, "regime_power": beta})
                cand["regime_3m"] = (best3[1], best3[2])
            # HAR-style: fitted weights on short, medium and long swings, then the same VIX scaling
            fam = "yz" if hist_name == "ohlc" else "cc"
            parts = [x[f"{fam}_v20"].values, x[f"{fam}_v60"].values, x[f"{fam}_v250"].values]
            besth = None
            for w1 in ((0.0, 0.25, 0.5, 0.75, 1.0) if EXPERIMENT else ()):
                for w2 in (0.0, 0.25, 0.5, 0.75, 1.0):
                    w3 = 1.0 - w1 - w2
                    if w3 < -1e-9:
                        continue
                    hv = np.sqrt(w1 * parts[0] ** 2 + w2 * parts[1] ** 2 + w3 * parts[2] ** 2)
                    hv = np.where(np.isfinite(hv) & (hv > 0), hv, hist)
                    sig = np.where(np.isfinite(m), hv * np.clip(m, 0.3, 5.0) ** beta_r, hv)
                    Q, info = fit_scaled(x.assign(_s=sig), lambda y: y["_s"].values, h)
                    s_ = pinball(a[tune], Q[tune]).mean()
                    if besth is None or s_ < besth[0]:
                        besth = (s_, Q, {**info, "w20": w1, "w60": w2, "w250": round(w3, 2), "regime_power": beta_r})
            if besth is not None:
                cand["har_regime"] = (besth[1], besth[2])
        # judge
        for name, (Q, info) in cand.items():
            ok = np.isfinite(Q).all(axis=1)
            for sp in ("tune", "holdout"):
                msk = ok & (x["split"].values == sp)
                results.append({"class": cls, "h": h, "candidate": name, "split": sp, **stats(a[msk], Q[msk]), **{f"fit_{k}": v for k, v in info.items()}})
        allQ[(cls, h)] = (x[["ticker", "i", "date", "split", "r"]].copy(), {nm: Q for nm, (Q, _) in cand.items()})
        tune_scores = {nm: pinball(a[tune & np.isfinite(Q).all(axis=1)], Q[tune & np.isfinite(Q).all(axis=1)]).mean()
                       for nm, (Q, _) in cand.items() if nm != "base"}
        pick = min(tune_scores, key=tune_scores.get)
        best_static[(cls, h)] = pick
        choices[f"{cls}_{h}"] = {"picked_on_older_years": pick, "hist_measure": hist_name,
                                 **{nm: info for nm, (_, info) in cand.items()}}
        # keep the picked range for the adaptive step
        for j in range(5):
            d.loc[(d["cls"] == cls) & (d["h"] == h), f"p{j}"] = cand[pick][0][:, j]
        print(f"  fitted {cls} {h}: picked {pick}, {time.time() - t0:.0f}s", flush=True)

res = pd.DataFrame(results)

# ---------- adaptive widening (Gibbs and Candes): multiply the picked range by m, updated from quick 5-day checks
FEED = 5


def adaptive_multiplier(lc, sig, gamma, target=0.2):
    """m[t] uses only 5-day outcomes already known at t. A miss widens, a hit narrows, aiming for 20% misses."""
    n = len(lc)
    out = np.ones(n)
    if gamma == 0:
        return out
    hist = np.zeros(n)
    logm = 0.0
    lo, hi = np.log(0.5), np.log(3.0)
    band = 1.2816 * np.sqrt(FEED)
    for t in range(FEED, n):
        s = sig[t - FEED]
        if s > 0:                                   # NaN compares False, so missing days are skipped
            miss = abs(lc[t] - lc[t - FEED]) > band * s * np.exp(hist[t - FEED])
            logm = min(max(logm + gamma * ((1.0 if miss else 0.0) - target), lo), hi)
        hist[t] = logm
        out[t] = np.exp(logm)
    return out


GAMMAS = (0.0, 0.005, 0.01, 0.02, 0.04)
ad_rows = []
for cls in ["stock", "fund", "commodity", "crypto"]:
    for h in HS:
        x = d[(d["cls"] == cls) & (d["h"] == h)]
        if not len(x) or "p0" not in x:
            continue
        mult = {g: np.ones(len(x)) for g in GAMMAS}
        for t, grp in x.groupby("ticker"):
            dd = daily[t]
            sig = dd["cc"][h]
            pos = np.where(x["ticker"].values == t)[0]
            for g in GAMMAS:
                mm = adaptive_multiplier(dd["lc"], sig, g)
                mult[g][pos] = mm[grp["i"].values]
        P = x[[f"p{j}" for j in range(5)]].values
        mid = P[:, 2:3]
        a = x["r"].values
        tune = x["split"].values == "tune"
        ok = np.isfinite(P).all(axis=1)
        sc = {g: pinball(a[tune & ok], (mid + mult[g][:, None] * (P - mid))[tune & ok]).mean() for g in GAMMAS}
        g = min(sc, key=sc.get)
        Q = mid + mult[g][:, None] * (P - mid)
        for sp in ("tune", "holdout"):
            msk = ok & (x["split"].values == sp)
            ad_rows.append({"class": cls, "h": h, "candidate": "adaptive", "split": sp, **stats(a[msk], Q[msk]),
                            "fit_step_size": g, "fit_on_top_of": best_static[(cls, h)]})
        print(f"  adaptive {cls} {h}: step {g}, {time.time() - t0:.0f}s", flush=True)
res = pd.concat([res, pd.DataFrame(ad_rows)], ignore_index=True)

# ---------- crypto options data: its own split (2021-2023 to fit, 2024 on to judge)
cr = []
for h in HS:
    x = d[(d["cls"] == "crypto") & (d["h"] == h) & d["direct"] & np.isfinite(d["iv"])].reset_index(drop=True)
    if len(x) < 100:
        continue
    a = x["r"].values
    x["split"] = np.where(pd.to_datetime(x["end"]) < DVOL_SPLIT, "tune", np.where(pd.to_datetime(x["date"]) >= DVOL_SPLIT, "holdout", "gap"))
    x = x[x["split"] != "gap"].reset_index(drop=True)
    a = x["r"].values
    tune = x["split"].values == "tune"
    B = x[[f"b{j}" for j in range(5)]].values
    bestb = None
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        for av in (0.6, 0.7, 0.8, 0.9, 1.0):
            sig = np.sqrt(w * (av * x["iv"].values) ** 2 + (1 - w) * x["cc"].values ** 2)
            Q, info = fit_scaled(x.assign(_s=sig), lambda y: y["_s"].values, h)
            s = pinball(a[tune], Q[tune]).mean()
            if bestb is None or s < bestb[0]:
                bestb = (s, Q, {**info, "weight_on_implied": w, "implied_scale": av})
    for name, Q, info in (("base", B, {}), ("implied (DVOL)", bestb[1], bestb[2])):
        for sp in ("tune", "holdout"):
            msk = x["split"].values == sp
            cr.append({"class": "crypto (BTC, ETH, 2021 on)", "h": h, "candidate": name, "split": sp, **stats(a[msk], Q[msk]),
                       **{f"fit_{k}": v for k, v in info.items()}})
res = pd.concat([res, pd.DataFrame(cr)], ignore_index=True)

# ---------- direct-index markets only (S&P, Nasdaq, Dow, gold, oil), where "implied" uses the market's own index
dm = []
for h in HS:
    x = d[(d["h"] == h) & d["direct"] & (d["cls"] != "crypto")].reset_index(drop=True)
    for cls in x["cls"].unique():
        y = x[x["cls"] == cls]
        a = y["r"].values
        for nm, cols in (("base", [f"b{j}" for j in range(5)]), ("picked", [f"p{j}" for j in range(5)])):
            Q = y[cols].values
            for sp in ("tune", "holdout"):
                msk = (y["split"].values == sp) & np.isfinite(Q).all(axis=1)
                dm.append({"class": f"{cls}: markets with their own options index", "h": h, "candidate": nm, "split": sp, **stats(a[msk], Q[msk])})
res = pd.concat([res, pd.DataFrame(dm)], ignore_index=True)

res.to_csv(os.path.join(OUT, "tighten_results.csv"), index=False)
with open(os.path.join(OUT, "tighten_choices.json"), "w", encoding="utf-8") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in choices.items()}, f, indent=2, default=str)

# ---------- the verdict, holdout years only
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
h_ = res[res["split"] == "holdout"].copy()
out = []
for (cls, h), g in h_.groupby(["class", "h"], sort=False):
    b = g[g["candidate"] == "base"]
    if not len(b):
        continue
    b = b.iloc[0]
    for _, r in g.iterrows():
        out.append({"class": cls, "h": h, "candidate": r["candidate"], "held80": round(r["held80"], 3), "held50": round(r["held50"], 3),
                    "width80_pct": round(r["width80_pct"], 2), "narrower_than_today": round(1 - r["width80_pct"] / b["width80_pct"], 3),
                    "score_better_than_today": round(1 - r["score"] / b["score"], 4), "n": r["n"]})
v = pd.DataFrame(out)
v.to_csv(os.path.join(OUT, "tighten_verdict_holdout.csv"), index=False)
print(v.to_string(index=False))


# ---------- the fair comparison: force every candidate to hold exactly 80% (and 50%) on the older years,
# then see which one is narrowest on the newer years and how steady it is year to year.
def calibrate(Q, a, tune):
    """Stretch the outer and inner edges around the middle so the older years hold exactly 80% and 50%."""
    med = Q[:, 2]
    out = Q.copy()
    for lo_j, hi_j, target in ((0, 4, 0.80), (1, 3, 0.50)):
        lo, hi = med - Q[:, lo_j], Q[:, hi_j] - med
        need = np.where(a < med, (med - a) / np.maximum(lo, 1e-12), (a - med) / np.maximum(hi, 1e-12))
        k = float(np.quantile(need[tune], target))
        out[:, lo_j], out[:, hi_j] = med - k * lo, med + k * hi
    return out


def judge(Q, a, split, dates):
    rec = {}
    for sp in ("tune", "holdout"):
        m = (split == sp) & np.isfinite(Q).all(axis=1)
        rec[f"{sp}_held80"] = float(((a[m] >= Q[m, 0]) & (a[m] <= Q[m, 4])).mean())
        rec[f"{sp}_held50"] = float(((a[m] >= Q[m, 1]) & (a[m] <= Q[m, 3])).mean())
        rec[f"{sp}_width80_pct"] = float((100 * (np.exp(Q[m, 4]) - np.exp(Q[m, 0]))).mean())
    m = np.isfinite(Q).all(axis=1)
    yr = pd.to_datetime(dates[m]).year
    inside = pd.Series(((a[m] >= Q[m, 0]) & (a[m] <= Q[m, 4])).astype(float)).groupby(np.asarray(yr)).agg(["mean", "size"])
    inside = inside[inside["size"] >= 50]
    rec["worst_year"] = int(inside["mean"].idxmin())
    rec["worst_year_held80"] = float(inside["mean"].min())
    rec["held80_2008"] = float(inside["mean"].get(2008, np.nan))
    rec["held80_2020"] = float(inside["mean"].get(2020, np.nan))
    hold_years = inside[inside.index >= 2019]["mean"]
    rec["holdout_worst_year_held80"] = float(hold_years.min()) if len(hold_years) else np.nan
    rec["years_below_70pct"] = int((inside["mean"] < 0.70).sum())
    return rec


eq_rows = []
final_params = {}
for (cls, h), (x, Qs) in allQ.items():
    a, split, dates = x["r"].values, x["split"].values, x["date"].values
    tune = split == "tune"
    calQ = {}
    for nm, Q in Qs.items():
        ok = np.isfinite(Q).all(axis=1)
        if (tune & ok).sum() < 100:
            continue
        Qc = np.full_like(Q, np.nan)
        Qc[ok] = calibrate(Q[ok], a[ok], tune[ok])
        calQ[nm] = Qc
    # adaptive widening on top of the static candidate that is narrowest on the older years at equal hold rate
    tw = {nm: judge(Q, a, split, dates)["tune_width80_pct"] for nm, Q in calQ.items() if nm != "base"}
    top = min(tw, key=tw.get)
    P = Qs[top]
    mid = P[:, 2:3]
    best = None
    for g in GAMMAS[1:]:
        mult = np.ones(len(x))
        for t, grp in x.groupby("ticker"):
            dd = daily[t]
            mm = adaptive_multiplier(dd["lc"], dd["cc"][h], g)
            mult[grp.index.values] = mm[grp["i"].values]
        Qg = mid + mult[:, None] * (P - mid)
        ok = np.isfinite(Qg).all(axis=1)
        Qgc = np.full_like(Qg, np.nan)
        Qgc[ok] = calibrate(Qg[ok], a[ok], tune[ok])
        w = judge(Qgc, a, split, dates)["tune_width80_pct"]
        if best is None or w < best[0]:
            best = (w, g, Qgc)
    calQ[f"adaptive({top}, step {best[1]})"] = best[2]
    # final settings for the live forecaster: the picked static candidate, its edges re-stretched on ALL years
    # (older and newer) so the live range holds 80% and 50% across everything seen so far
    Qp = Qs[top]
    okp = np.isfinite(Qp).all(axis=1)
    medp = Qp[okp, 2]
    fac = {}
    for lo_j, hi_j, target, nm_ in ((0, 4, 0.80, "outer"), (1, 3, 0.50, "inner")):
        lo, hi = medp - Qp[okp, lo_j], Qp[okp, hi_j] - medp
        aa = a[okp]
        need = np.where(aa < medp, (medp - aa) / np.maximum(lo, 1e-12), (aa - medp) / np.maximum(hi, 1e-12))
        fac[nm_] = float(np.quantile(need, target))
    final_params[f"{cls}_{h}"] = {"candidate": top, "calibration_all_years": fac,
                                  "fit": {k_: v_ for k_, v_ in choices[f"{cls}_{h}"].get(top, {}).items()},
                                  "hist_measure": choices[f"{cls}_{h}"]["hist_measure"], "vol_name": base_kw(cls, h)["vol_name"],
                                  "evidence_holdout": judge(calQ[top], a, split, dates)}
    for nm, Q in calQ.items():
        eq_rows.append({"class": cls, "h": h, "candidate": nm, **judge(Q, a, split, dates)})
    base_as_is = judge(Qs["base"], a, split, dates)
    eq_rows.append({"class": cls, "h": h, "candidate": "base as it runs today (not recalibrated)", **base_as_is})
    print(f"  equal-hold {cls} {h} done, {time.time() - t0:.0f}s", flush=True)

eq = pd.DataFrame(eq_rows)
for (cls, h), g in eq.groupby(["class", "h"], sort=False):
    today = g[g["candidate"].str.startswith("base as it runs")]["holdout_width80_pct"].iloc[0]
    eq.loc[g.index, "narrower_than_today"] = 1 - g["holdout_width80_pct"] / today
eq.to_csv(os.path.join(OUT, "tighten_equal_hold.csv"), index=False)
cols = ["class", "h", "candidate", "tune_held80", "holdout_held80", "holdout_held50", "holdout_width80_pct", "narrower_than_today",
        "worst_year", "worst_year_held80", "held80_2008", "held80_2020", "holdout_worst_year_held80", "years_below_70pct"]
with open(os.path.join(OUT, "final_params.json"), "w", encoding="utf-8") as f:
    json.dump(final_params, f, indent=2, default=float)
print("\nEQUAL HOLD RATE (every candidate forced to hold 80% on the older years)")
print(eq[cols].round(3).to_string(index=False))
print(f"done in {time.time() - t0:.0f}s")
