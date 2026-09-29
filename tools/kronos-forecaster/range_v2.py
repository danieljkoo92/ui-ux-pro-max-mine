"""The tightened range (version 2), built from the winners in tighten.py.

What changed from version 1, by kind of market:
  stocks, funds, indexes  past swings scaled by how jumpy the options market expects the whole market to be
                          (the VIX against the S&P 500's own recent swings), with a shape fitted on 2000 to 2018
  commodity futures       version 1's range, stretched to hold 80% and 50% (nothing tested beat it)
  crypto                  swings from closing prices with a fitted shape; this WIDENS the range, because
                          version 1 held only about 75%
Every range is then re-stretched month by month so it holds 80% and 50% over the last few years
(rolling_calibration.py; window picked on 2012 to 2018: 3 years, or 8 for 30-day stocks). Crypto has too little
history for that and keeps one stretch fitted on all years.
Stock ranges 45 days or shorter are stretched once more by the number of earnings reports inside the window
(earnings_test.py): narrower with no report, wider with one.
The settings live in out/tighten/ and are rebuilt by running tighten.py, rolling_calibration.py and earnings_test.py.
"""
import json
import os

import numpy as np
import pandas as pd

import market_data as md
import range_model as rm
import vol_data as vd

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = json.load(open(os.path.join(HERE, "out", "tighten", "final_params.json")))
chosen_v1 = json.load(open(os.path.join(HERE, "out", "range", "chosen.json")))


def _json_or_empty(name):
    p = os.path.join(HERE, "out", "tighten", name)
    return json.load(open(p)) if os.path.exists(p) else {}


ROLLING = _json_or_empty("rolling_factors.json")
EARNINGS = _json_or_empty("earnings_factors.json")
TICKER_K = _json_or_empty("per_ticker_K.json")


def _own_factors(R, lc, dates, h, W):
    """This ticker's own stretch for each month, from its forecasts that finished inside the last W years.

    Returns per-day arrays: outer factor, inner factor, and how many own forecasts they rest on."""
    n = len(dates)
    idx = np.arange(260, n - h, 5)
    ok = np.isfinite(R[idx]).all(axis=1)
    idx = idx[ok]
    a = lc[idx + h] - lc[idx]
    med = R[idx, 2]
    need_o = np.where(a < med, (med - a) / np.maximum(med - R[idx, 0], 1e-12), (a - med) / np.maximum(R[idx, 4] - med, 1e-12))
    need_i = np.where(a < med, (med - a) / np.maximum(med - R[idx, 1], 1e-12), (a - med) / np.maximum(R[idx, 3] - med, 1e-12))
    ends = dates.values[idx + h]
    starts = dates.values[idx]
    order = np.argsort(ends)
    ends, starts, need_o, need_i = ends[order], starts[order], need_o[order], need_i[order]
    months = dates.to_period("M")
    out_o, out_i, out_n = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    for m in months.unique():
        start = m.start_time
        hi = np.searchsorted(ends, np.datetime64(start), side="left")
        keep = starts[:hi] >= np.datetime64(start - pd.DateOffset(years=W))
        k = int(keep.sum())
        sel = months == m
        out_n[sel] = k
        if k >= 20:
            out_o[sel] = np.quantile(need_o[:hi][keep], 0.80)
            out_i[sel] = np.quantile(need_i[:hi][keep], 0.50)
    return out_o, out_i, out_n
REG = {"stock": ("^VIX", "^GSPC"), "fund": ("^VIX", "^GSPC")}
REG_COMMODITY = {"energy": ("^OVX", "CL=F"), "metals": ("^GVZ", "GC=F")}


def key_for(cls, h):
    return f"{cls}_{30 if h <= 45 else 90}"


def _yang_zhang(df, name):
    lo, lh, ll, lc = (np.log(df[x]) for x in ("open", "high", "low", "close"))
    over, oc = lo - lc.shift(1), lc - lo
    u, d = lh - lo, ll - lo
    rs = u * (u - oc) + d * (d - oc)
    v = {}
    for n in (20, 60, 250):
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        v[n] = np.sqrt((over.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean()).clip(lower=0))
    return {"v20": v[20], "v60": v[60], "v250": v[250], "ewma": v[60],
            "blend3": np.sqrt((v[20] ** 2 + v[60] ** 2 + v[250] ** 2) / 3),
            "blend_60_250": np.sqrt((v[60] ** 2 + v[250] ** 2) / 2)}[name]


def _align(series, dates, limit=5):
    s = series[~series.index.duplicated()].sort_index()
    return s.reindex(s.index.union(dates)).ffill(limit=limit).reindex(dates).values


def _regime(ticker, cls, dates):
    reg = REG.get(cls) or REG_COMMODITY.get(md.commodity_group(ticker) or "")
    if not reg:
        return np.full(len(dates), np.nan)
    try:
        iv = vd.load(reg[0])
    except Exception:
        return np.full(len(dates), np.nan)
    ref = md.load(reg[1], max_age_hours=24, need_volume=False)
    ref_v = pd.Series(rm.daily_vol(ref["close"])["blend_60_250"].values, index=pd.DatetimeIndex(ref["timestamps"]))
    return _align(iv, dates) / 100 / np.sqrt(252.0) / _align(ref_v, dates)


def _stretch(Q, outer, inner):
    med = Q[:, 2:3]
    out = Q.copy()
    out[:, [0, 4]] = med + outer * (Q[:, [0, 4]] - med)
    out[:, [1, 3]] = med + inner * (Q[:, [1, 3]] - med)
    return out


def fan(df, ticker, steps, H):
    """Range edges at each look-ahead in steps, for the last day, blending the 30-day and 90-day settings
    smoothly so the fan widens evenly instead of jumping where one set of settings hands over to the other."""
    out = {}
    for s in steps:
        if H <= 45 or s <= 30:
            out[s] = quantiles(df, ticker, s, settings_h=30 if H > 45 else None)[0].iloc[-1]
            continue
        q30 = quantiles(df, ticker, s, settings_h=30)[0].iloc[-1]
        q90 = quantiles(df, ticker, s, settings_h=90)[0].iloc[-1]
        w = min(1.0, max(0.0, (s - 30) / max(H - 30, 1)))
        out[s] = (1 - w) * q30 + w * q90
    return out


def quantiles(df, ticker, h, stage="final", settings_h=None):
    """Five range edges (10%, 25%, 50%, 75%, 90%) of the h-day log return, for every day in df.

    stage="raw" stops before any stretching (what the calibration scripts fit on); "final" is the live range.
    Returns (DataFrame with columns rm.QS, dict describing the method)."""
    cls = md.asset_class(ticker)
    key = key_for(cls, settings_h or h)
    p = PARAMS[key]
    c = df["close"]
    dates = pd.DatetimeIndex(df["timestamps"])
    cand, fit = p["candidate"], p["fit"]
    if cand in ("base", "recal"):
        m, vol = chosen_v1[key].split("/")
        kk = float(vol.split(" x")[1]) if " x" in vol else 1.0
        q = rm.forecast(c, h, vol_name=vol.split(" x")[0], method=m, k=kk)
        Q = q[rm.QS].values
        if cand == "recal":
            Q = _stretch(Q, fit["stretch"], fit["stretch"])
    else:
        close_v = rm.daily_vol(c)[p["vol_name"]].values
        if cand == "ohlc" or (cand in ("regime", "implied") and p["hist_measure"] == "ohlc"):
            yz = _yang_zhang(df, p["vol_name"]).values
            hist = np.where(np.isfinite(yz) & (yz > 0), yz, close_v)
        else:
            hist = close_v
        sig = hist
        if cand == "regime":
            mm = _regime(ticker, cls, dates)
            sig = np.where(np.isfinite(mm), hist * np.clip(mm, 0.3, 5.0) ** fit["regime_power"], hist)
        elif cand == "implied":
            ivn = vd.DIRECT.get(ticker)
            ann = np.sqrt(365.0 if cls == "crypto" else 252.0)
            iv = _align(vd.load(ivn), dates) / 100 / ann if ivn and ivn != "^MOVE" else np.full(len(df), np.nan)
            w, av = fit["weight_on_implied"], fit["implied_scale"]
            sig = np.where(np.isfinite(iv), np.sqrt(w * (av * iv) ** 2 + (1 - w) * hist ** 2), hist)
        shape = np.array(fit["shape_z"])
        Q = fit["scale"] * (sig * np.sqrt(h))[:, None] * shape[None, :]
    if stage == "raw":
        return pd.DataFrame(Q, columns=rm.QS, index=df.index), {"method": cand, "key": key}
    cal = p["calibration_all_years"]
    extra = {}
    roll = ROLLING.get(key)
    if roll:
        tab = roll["monthly"]
        months = sorted(tab)
        cur = dates.to_period("M").astype(str)
        last_m = months[-1]
        fo = np.array([tab[m][0] if m in tab else (tab[last_m][0] if m > last_m else cal["outer"]) for m in cur])
        fi = np.array([tab[m][1] if m in tab else (tab[last_m][1] if m > last_m else cal["inner"]) for m in cur])
        K = TICKER_K.get(key)
        if K:
            # lean toward this ticker's own record, more so the more of it there is (per_ticker_calibration.py)
            oo, oi, on = _own_factors(Q, np.log(c.values), dates, h, roll["window_years"])
            w = np.where(np.isfinite(oo), on / (on + K), 0.0)
            fo = w * np.nan_to_num(oo) + (1 - w) * fo
            fi = w * np.nan_to_num(oi) + (1 - w) * fi
            extra["own_record_weight_now"] = float(w[-1])
        med = Q[:, 2:3]
        Q = Q.copy()
        Q[:, [0, 4]] = med + fo[:, None] * (Q[:, [0, 4]] - med)
        Q[:, [1, 3]] = med + fi[:, None] * (Q[:, [1, 3]] - med)
        extra["recalibrated_monthly_years"] = roll["window_years"]
    else:
        Q = _stretch(Q, cal["outer"], cal["inner"])
    if cls == "stock" and h <= 45 and str(30) in EARNINGS:
        n_e = earnings_in_window(ticker, dates, h)
        if n_e is not None:
            rel = EARNINGS["30"]
            fo = np.array([rel.get(str(min(k, 2)), rel.get("1", {"outer": 1.0}))["outer"] for k in n_e])
            fi = np.array([rel.get(str(min(k, 2)), rel.get("1", {"inner": 1.0}))["inner"] for k in n_e])
            med = Q[:, 2:3]
            Q[:, [0, 4]] = med + fo[:, None] * (Q[:, [0, 4]] - med)
            Q[:, [1, 3]] = med + fi[:, None] * (Q[:, [1, 3]] - med)
            extra["earnings_reports_in_window_now"] = int(n_e[-1])
    return pd.DataFrame(Q, columns=rm.QS, index=df.index), {"method": cand, "key": key, "evidence": p["evidence_holdout"], **extra}


def earnings_in_window(ticker, dates, h):
    """For every day: how many earnings reactions fall in the next h trading days (future dates included)."""
    import earnings_data as et
    e = et.earnings_dates(ticker)
    if e is None or not len(e):
        return None
    when = pd.DatetimeIndex(e["when"])
    # extend the trading calendar h business days past the last day, so a scheduled report counts
    future = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=h + 5)
    cal = dates.append(future)
    react = et.reaction_index(cal, when)
    marks = np.zeros(len(cal), dtype=int)
    marks[react] = 1
    cum = np.cumsum(marks)
    idx = np.arange(len(dates))
    return cum[np.minimum(idx + h, len(cal) - 1)] - cum[idx]
