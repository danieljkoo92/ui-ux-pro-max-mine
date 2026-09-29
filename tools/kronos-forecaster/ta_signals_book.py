"""Chart rules from John Murphy's "Technical Analysis of the Financial Markets", coded for daily candles.

Only rules the book states objectively enough to code are here. Where the book gives a number (3% penetration,
two closes, 4 weeks, 9-18 day averages) that number is used. Where it gives none, the choice is noted.

Every signal is a call made at the close of a day using only that day and earlier days:
+1 means "up from here", -1 means "down from here", 0 means no call.

Left out: anything needing open interest, Commitments of Traders, options data, sentiment surveys or
candles inside the day; and patterns the book describes without a rule a computer could follow
(saucers, spikes, internal trendlines, Gann angles, wedges, broadening tops).
"""
import numpy as np
import pandas as pd

K = 5            # swing point: highest high / lowest low of the 5 days either side, known 5 days later


def _atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _pivots(h, l, k=K):
    """Lists of (day the swing happened, price, day it became known)."""
    highs, lows = [], []
    n = len(h)
    for c in range(k, n - k):
        if h[c] == h[c - k:c + k + 1].max():
            highs.append((c, h[c], c + k))
        if l[c] == l[c - k:c + k + 1].min():
            lows.append((c, l[c], c + k))
    return highs, lows


def _divergence(price, ind, look=20, gap=(5, 60)):
    """Price makes a new `look`-day extreme beyond its last one while the indicator does not."""
    n = len(price)
    up = np.zeros(n, dtype=bool); dn = np.zeros(n, dtype=bool)
    hi = (price >= price.rolling(look).max()).values
    lo = (price <= price.rolling(look).min()).values
    p, x = price.values, ind.values
    lh = ll = None
    for t in range(n):
        if hi[t]:
            if lh is not None and gap[0] <= t - lh <= gap[1] and p[t] > p[lh] and x[t] < x[lh]:
                dn[t] = True
            lh = t
        if lo[t]:
            if ll is not None and gap[0] <= t - ll <= gap[1] and p[t] < p[ll] and x[t] > x[ll]:
                up[t] = True
            ll = t
    return pd.Series(up, index=price.index), pd.Series(dn, index=price.index)


def month_of(df):
    return df["timestamps"].dt.month


def _cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def compute(df):
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    n = len(df)
    hv, lv, cv = h.values, l.values, c.values
    atr = _atr(df)
    sma = {k: c.rolling(k).mean() for k in (4, 5, 9, 10, 18, 20, 50, 200)}
    vol20 = v.rolling(20).mean().shift(1)
    rel_vol = v / vol20
    up_trend = (c > sma[50]) & (sma[50] > sma[50].shift(10))
    dn_trend = (c < sma[50]) & (sma[50] < sma[50].shift(10))
    s = {}

    def call(name, up=None, down=None):
        x = pd.Series(0, index=df.index, dtype="int8")
        if up is not None:
            x[pd.Series(up, index=df.index).fillna(False).astype(bool)] = 1
        if down is not None:
            x[pd.Series(down, index=df.index).fillna(False).astype(bool)] = -1
        s[name] = x

    highs, lows = _pivots(hv, lv)

    # ---------- Chapter 4: trend
    # key reversal day: new high for the move, close below yesterday's close (and the mirror)
    new_hi20 = h > h.rolling(20).max().shift(1)
    new_lo20 = l < l.rolling(20).min().shift(1)
    top_rev = new_hi20 & (c < c.shift(1))
    bot_rev = new_lo20 & (c > c.shift(1))
    call("reversal_day", up=bot_rev, down=top_rev)
    outside = (h > h.shift(1)) & (l < l.shift(1))
    call("reversal_day_outside_heavy_volume", up=bot_rev & outside & (rel_vol > 1.5), down=top_rev & outside & (rel_vol > 1.5))

    # weekly reversal, read on the last trading day of each week
    wk = df["timestamps"].dt.to_period("W")
    last_of_week = wk != wk.shift(-1)
    g = df.groupby(wk)
    w_high, w_low = g["high"].transform("max"), g["low"].transform("min")
    wclose = c.where(last_of_week)
    prev_wclose = wclose.dropna().shift(1).reindex(df.index)
    w13_hi = h.rolling(65).max().shift(5)
    w13_lo = l.rolling(65).min().shift(5)
    call("weekly_reversal", up=last_of_week & (w_low < w13_lo) & (c > prev_wclose),
         down=last_of_week & (w_high > w13_hi) & (c < prev_wclose))

    # gaps (stocks and funds; crypto trades round the clock and rarely gaps)
    gap_up = l > h.shift(1)
    gap_dn = h < l.shift(1)
    hi50 = h.rolling(50).max().shift(1)
    lo50 = l.rolling(50).min().shift(1)
    call("breakaway_gap", up=gap_up & (c > hi50) & (rel_vol > 1.5), down=gap_dn & (c < lo50) & (rel_vol > 1.5))
    call("runaway_gap", up=gap_up & up_trend.shift(1) & ~(c > hi50), down=gap_dn & dn_trend.shift(1) & ~(c < lo50))
    ext_up = c.shift(1) > sma[50].shift(1) * 1.10          # book gives no number for "late in a move": 10% above the 50-day
    ext_dn = c.shift(1) < sma[50].shift(1) * 0.90
    ex_up = np.zeros(n, dtype=bool); ex_dn = np.zeros(n, dtype=bool)
    isl_top = np.zeros(n, dtype=bool); isl_bot = np.zeros(n, dtype=bool)
    gu, gd, eu, ed = gap_up.values, gap_dn.values, ext_up.values, ext_dn.values
    for t in range(1, n):
        if gu[t] and eu[t]:
            edge = hv[t - 1]
            for j in range(t + 1, min(t + 6, n)):          # "within a couple of days to a week"
                if cv[j] < edge:
                    ex_up[j] = True
                    break
        if gd[t] and ed[t]:
            edge = lv[t - 1]
            for j in range(t + 1, min(t + 6, n)):
                if cv[j] > edge:
                    ex_dn[j] = True
                    break
        if gu[t]:
            for j in range(t + 2, min(t + 11, n)):         # island: a few days to two weeks above the gap, then a gap down
                if gd[j] and hv[j] < lv[t:j].min():
                    isl_top[j] = True
                    break
        if gd[t]:
            for j in range(t + 2, min(t + 11, n)):
                if gu[j] and lv[j] > hv[t:j].max():
                    isl_bot[j] = True
                    break
    call("exhaustion_gap", up=ex_dn, down=ex_up)
    call("island_reversal", up=isl_bot, down=isl_top)

    # support/resistance role reversal: a swing level broken by 3% on a close, then retested from the other side
    flip_up = np.zeros(n, dtype=bool); flip_dn = np.zeros(n, dtype=bool)
    for (ci, lvl, known) in highs:
        brk = None
        for t in range(known, min(known + 120, n)):
            if brk is None:
                if cv[t] > lvl * 1.03:
                    brk = t
            elif t > brk + 1 and lv[t] <= lvl * 1.01 and cv[t] > lvl:
                flip_up[t] = True
                break
            elif brk is not None and cv[t] < lvl * 0.97:
                break
    for (ci, lvl, known) in lows:
        brk = None
        for t in range(known, min(known + 120, n)):
            if brk is None:
                if cv[t] < lvl * 0.97:
                    brk = t
            elif t > brk + 1 and hv[t] >= lvl * 0.99 and cv[t] < lvl:
                flip_dn[t] = True
                break
            elif brk is not None and cv[t] > lvl * 1.03:
                break
    call("broken_level_retest", up=flip_up, down=flip_dn)

    # trendline through the last two swing lows (second one higher); break = two closes beyond it (two-day rule)
    tl_dn = np.zeros(n, dtype=bool); tl_up = np.zeros(n, dtype=bool)
    tl3_dn = np.zeros(n, dtype=bool); tl3_up = np.zeros(n, dtype=bool)
    for piv, below, out2, out3 in ((lows, True, tl_dn, tl3_dn), (highs, False, tl_up, tl3_up)):
        for a, b in zip(piv[:-1], piv[1:]):
            (i0, p0, _), (i1, p1, k1) = a, b
            if i1 - i0 < 10 or (below and p1 <= p0) or ((not below) and p1 >= p0):
                continue
            slope = (p1 - p0) / (i1 - i0)
            done2 = done3 = False
            for t in range(k1 + 1, min(k1 + 120, n)):
                line = p1 + slope * (t - i1)
                if line <= 0:
                    break
                if below:
                    two = cv[t] < line and cv[t - 1] < p1 + slope * (t - 1 - i1)
                    three = cv[t] < line * 0.97
                else:
                    two = cv[t] > line and cv[t - 1] > p1 + slope * (t - 1 - i1)
                    three = cv[t] > line * 1.03
                if two and not done2:
                    out2[t] = True; done2 = True
                if three and not done3:
                    out3[t] = True; done3 = True
                if done2 and done3:
                    break
    call("trendline_break_two_closes", up=tl_up, down=tl_dn)
    call("trendline_break_3pct", up=tl3_up, down=tl3_dn)

    # retracements of the last leg: 33-50% pullback = buy zone, beyond 66% = reversal
    rz_up = np.zeros(n, dtype=bool); rz_dn = np.zeros(n, dtype=bool)
    rv_up = np.zeros(n, dtype=bool); rv_dn = np.zeros(n, dtype=bool)
    ev = sorted([(x[2], "H", x[0], x[1]) for x in highs] + [(x[2], "L", x[0], x[1]) for x in lows])
    last_h = last_l = None
    ptr = 0
    in_zone = False; past = False; leg = None
    for t in range(n):
        while ptr < len(ev) and ev[ptr][0] <= t:
            _, kind, ci, px = ev[ptr]
            if kind == "H":
                last_h = (ci, px)
            else:
                last_l = (ci, px)
            ptr += 1
            in_zone = past = False
        if last_h is None or last_l is None:
            continue
        up_leg = last_h[0] > last_l[0]
        size = last_h[1] - last_l[1]
        if size <= 0 or size / last_l[1] < 0.05:             # ignore legs under 5%
            continue
        r = (last_h[1] - cv[t]) / size if up_leg else (cv[t] - last_l[1]) / size
        if 0.33 <= r <= 0.50 and not in_zone:
            (rz_up if up_leg else rz_dn)[t] = True
            in_zone = True
        if r > 0.66 and not past:
            (rv_dn if up_leg else rv_up)[t] = True
            past = True
    call("retrace_33_50_buy_zone", up=rz_up, down=rz_dn)
    call("retrace_beyond_66", up=rv_up, down=rv_dn)

    # ---------- Chapters 5 and 6: patterns
    dbl_top = np.zeros(n, dtype=bool); dbl_bot = np.zeros(n, dtype=bool)
    tri_top = np.zeros(n, dtype=bool); tri_bot = np.zeros(n, dtype=bool)
    hs_top = np.zeros(n, dtype=bool); hs_bot = np.zeros(n, dtype=bool)

    def between(piv, i0, i1, want_min):
        x = [p for p in piv if i0 < p[0] < i1]
        if not x:
            return None
        return min(x, key=lambda z: z[1]) if want_min else max(x, key=lambda z: z[1])

    for same, other, top, d2, d3, dhs in ((highs, lows, True, dbl_top, tri_top, hs_top), (lows, highs, False, dbl_bot, tri_bot, hs_bot)):
        sgn = 1 if top else -1
        for j in range(1, len(same)):
            a, b = same[j - 1], same[j]
            gap_days = b[0] - a[0]
            # double: two peaks about equal (within 3%), one to six months apart
            if 20 <= gap_days <= 130 and abs(b[1] / a[1] - 1) <= 0.03:
                mid = between(other, a[0], b[0], want_min=top)
                if mid is not None:
                    for t in range(b[2], min(b[2] + 60, n)):
                        if sgn * (mid[1] - cv[t]) > 0 and sgn * (mid[1] - cv[t]) / mid[1] >= 0.01:
                            d2[t] = True
                            break
                        if sgn * (cv[t] - max(a[1], b[1]) if top else min(a[1], b[1]) - cv[t]) > 0:
                            break
            if j >= 2:
                z = same[j - 2]
                # triple: three peaks about equal
                if max(abs(a[1] / z[1] - 1), abs(b[1] / a[1] - 1), abs(b[1] / z[1] - 1)) <= 0.03 and 40 <= b[0] - z[0] <= 200:
                    m1 = between(other, z[0], a[0], want_min=top); m2 = between(other, a[0], b[0], want_min=top)
                    if m1 is not None and m2 is not None:
                        lvl = min(m1[1], m2[1]) if top else max(m1[1], m2[1])
                        for t in range(b[2], min(b[2] + 60, n)):
                            if sgn * (lvl - cv[t]) > 0:
                                d3[t] = True
                                break
                # head and shoulders: middle peak highest, shoulders within 5% of each other
                head_ok = sgn * (a[1] - z[1]) > 0 and sgn * (a[1] - b[1]) > 0
                if head_ok and abs(b[1] / z[1] - 1) <= 0.05 and 30 <= b[0] - z[0] <= 200:
                    m1 = between(other, z[0], a[0], want_min=top); m2 = between(other, a[0], b[0], want_min=top)
                    if m1 is not None and m2 is not None and m2[0] > m1[0]:
                        slope = (m2[1] - m1[1]) / (m2[0] - m1[0])
                        for t in range(b[2], min(b[2] + 60, n)):
                            neck = m2[1] + slope * (t - m2[0])
                            if neck > 0 and sgn * (neck - cv[t]) / neck >= 0.01:
                                dhs[t] = True
                                break
    call("double_top_bottom", up=dbl_bot, down=dbl_top)
    call("triple_top_bottom", up=tri_bot, down=tri_top)
    call("head_and_shoulders", up=hs_bot, down=hs_top)

    # rectangle / range breakout: one to three months inside a band under 15% wide, then a close outside it
    for win, nm in ((40, "rectangle_breakout_40d"),):
        hh = h.rolling(win).max().shift(1); ll = l.rolling(win).min().shift(1)
        tight = (hh / ll - 1) <= 0.15
        call(nm, up=tight & (c > hh), down=tight & (c < ll))
        call(nm + "_volume", up=tight & (c > hh) & (rel_vol > 1.5), down=tight & (c < ll) & (rel_vol > 1.5))

    # triangles from the last two swing highs and lows (flat = within 1.5%)
    asc = np.zeros(n, dtype=bool); desc = np.zeros(n, dtype=bool); sym_up = np.zeros(n, dtype=bool); sym_dn = np.zeros(n, dtype=bool)
    hk = [(x[2], x) for x in highs]; lk = [(x[2], x) for x in lows]
    ih = il = 0
    H = []; L = []
    fired = None
    for t in range(n):
        while ih < len(hk) and hk[ih][0] <= t:
            H.append(hk[ih][1]); ih += 1; fired = None
        while il < len(lk) and lk[il][0] <= t:
            L.append(lk[il][1]); il += 1; fired = None
        if len(H) < 2 or len(L) < 2 or fired is not None:
            continue
        h0, h1, l0, l1 = H[-2], H[-1], L[-2], L[-1]
        start = min(h0[0], l0[0])
        if not (20 <= t - start <= 65):
            continue
        flat_top = abs(h1[1] / h0[1] - 1) <= 0.015
        flat_bot = abs(l1[1] / l0[1] - 1) <= 0.015
        rising_lows = l1[1] > l0[1] * 1.015
        falling_highs = h1[1] < h0[1] * 0.985
        upper = h1[1] + (h1[1] - h0[1]) / max(h1[0] - h0[0], 1) * (t - h1[0])
        lower = l1[1] + (l1[1] - l0[1]) / max(l1[0] - l0[0], 1) * (t - l1[0])
        if flat_top and rising_lows and cv[t] > max(h0[1], h1[1]):
            asc[t] = True; fired = t
        elif flat_bot and falling_highs and cv[t] < min(l0[1], l1[1]):
            desc[t] = True; fired = t
        elif falling_highs and rising_lows and upper > lower:
            if cv[t] > upper:
                sym_up[t] = True; fired = t
            elif cv[t] < lower:
                sym_dn[t] = True; fired = t
    call("right_angle_triangle", up=asc, down=desc)
    call("symmetrical_triangle_break", up=sym_up, down=sym_dn)

    # flag or pennant: a sharp pole, one to three weeks of quiet drift against it on lighter volume, then the break
    fl_up = np.zeros(n, dtype=bool); fl_dn = np.zeros(n, dtype=bool)
    av = atr.values; vv = v.values
    for t in range(30, n):
        for m in (5, 8, 10, 15):
            e = t - m                       # last day of the pole
            s0 = e - 8
            if s0 < 0 or not np.isfinite(av[e]) or av[e] <= 0:
                continue
            move = cv[e] - cv[s0]
            if abs(move) < 4 * av[e]:       # book gives no number for "sharp": 4 average daily ranges in 8 days
                continue
            seg_h, seg_l = hv[e + 1:t].max() if t > e + 1 else hv[e], lv[e + 1:t].min() if t > e + 1 else lv[e]
            quiet = vv[e + 1:t].mean() < vv[s0 + 1:e + 1].mean() if t > e + 1 else False
            if not quiet:
                continue
            if move > 0 and seg_l > cv[e] - 0.5 * move and seg_h <= hv[e] * 1.01 and cv[t] > max(seg_h, hv[e]):
                fl_up[t] = True; break
            if move < 0 and seg_h < cv[e] - 0.5 * move and seg_l >= lv[e] * 0.99 and cv[t] < min(seg_l, lv[e]):
                fl_dn[t] = True; break
    call("flag_pennant", up=fl_up, down=fl_dn)

    # ---------- Chapter 7: volume
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    d_up, d_dn = _divergence(c, obv)
    call("obv_divergence", up=d_up, down=d_dn)
    call("obv_trend_confirms", up=up_trend & (obv > obv.rolling(50).mean()), down=dn_trend & (obv < obv.rolling(50).mean()))
    climax_dn = (l < l.rolling(60).min().shift(1)) & (rel_vol > 2.5) & (c > (h + l) / 2)
    blowoff = (h > h.rolling(60).max().shift(1)) & (rel_vol > 2.5) & (c < (h + l) / 2)
    call("selling_climax_blowoff", up=climax_dn, down=blowoff)

    # ---------- Chapter 9: moving averages
    call("close_cross_50day", up=_cross_up(c, sma[50]), down=_cross_dn(c, sma[50]))
    call("close_cross_200day", up=_cross_up(c, sma[200]), down=_cross_dn(c, sma[200]))
    call("double_crossover_5_20", up=_cross_up(sma[5], sma[20]), down=_cross_dn(sma[5], sma[20]))
    call("double_crossover_10_50", up=_cross_up(sma[10], sma[50]), down=_cross_dn(sma[10], sma[50]))
    call("double_crossover_10_50_state", up=sma[10] > sma[50], down=sma[10] < sma[50])
    trip_up = (sma[4] > sma[9]) & (sma[9] > sma[18])
    trip_dn = (sma[4] < sma[9]) & (sma[9] < sma[18])
    call("triple_crossover_4_9_18", up=trip_up & ~trip_up.shift(1, fill_value=False).astype(bool),
         down=trip_dn & ~trip_dn.shift(1, fill_value=False).astype(bool))
    call("triple_crossover_4_9_18_state", up=trip_up, down=trip_dn)
    sd = c.rolling(20).std()
    bb_u, bb_l = sma[20] + 2 * sd, sma[20] - 2 * sd
    call("bollinger_band_touch_fade", up=c < bb_l, down=c > bb_u)                       # band as a turning point
    call("bollinger_band_touch_follow", up=c > bb_u, down=c < bb_l)                     # band break as strength
    call("bollinger_midline_target", up=(c > sma[20]) & (c.shift(1) <= sma[20].shift(1)),
         down=(c < sma[20]) & (c.shift(1) >= sma[20].shift(1)))
    bw = (bb_u - bb_l) / sma[20]
    squeeze = bw <= bw.rolling(120).min().shift(1) * 1.05
    was_squeeze = squeeze.rolling(10).max().shift(1).fillna(0).astype(bool)
    call("squeeze_then_break", up=was_squeeze & (c > bb_u), down=was_squeeze & (c < bb_l))
    sma21 = c.rolling(21).mean()
    env_u, env_l = sma21 * 1.03, sma21 * 0.97           # the book's daily setting: 3% around a 21-day average
    call("envelope_3pct_stretch_fade", up=c < env_l, down=c > env_u)
    # four-week rule (20 trading days), and its longer cousins
    for win, nm in ((20, "four_week_rule"), (60, "twelve_week_breakout")):
        hh = h.rolling(win).max().shift(1); ll = l.rolling(win).min().shift(1)
        call(nm, up=(c > hh) & ~(c.shift(1) > hh.shift(1)), down=(c < ll) & ~(c.shift(1) < ll.shift(1)))
    pos = pd.Series(np.nan, index=df.index)
    hh = h.rolling(20).max().shift(1); ll = l.rolling(20).min().shift(1)
    pos[c > hh] = 1; pos[c < ll] = -1
    call("four_week_rule_state", up=pos.ffill() > 0, down=pos.ffill() < 0)

    # ---------- Chapter 10: oscillators
    mom = c - c.shift(10)
    call("momentum_10_zero_cross", up=_cross_up(mom, mom * 0), down=_cross_dn(mom, mom * 0))
    rsi = _rsi(c)
    call("rsi_30_70_cross_back", up=_cross_up(rsi, rsi * 0 + 30), down=_cross_dn(rsi, rsi * 0 + 70))
    call("rsi_with_trend", up=_cross_up(rsi, rsi * 0 + 30) & (c > sma[200]), down=_cross_dn(rsi, rsi * 0 + 70) & (c < sma[200]))
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    kfast = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    kslow = kfast.rolling(3).mean(); dslow = kslow.rolling(3).mean()
    call("stochastic_cross_in_extreme", up=_cross_up(kslow, dslow) & (dslow < 20), down=_cross_dn(kslow, dslow) & (dslow > 80))
    call("stochastic_with_trend", up=_cross_up(kslow, dslow) & (dslow < 20) & (c > sma[200]),
         down=_cross_dn(kslow, dslow) & (dslow > 80) & (c < sma[200]))
    tp = (h + l + c) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean())
    call("cci_100_follow", up=_cross_up(cci, cci * 0 + 100), down=_cross_dn(cci, cci * 0 - 100))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sigl = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sigl
    call("macd_signal_cross", up=_cross_up(macd, sigl), down=_cross_dn(macd, sigl))
    call("macd_cross_far_from_zero", up=_cross_up(macd, sigl) & (macd < 0), down=_cross_dn(macd, sigl) & (macd > 0))
    call("macd_histogram_turn", up=(hist > hist.shift(1)) & (hist.shift(1) <= hist.shift(2)) & (hist < 0),
         down=(hist < hist.shift(1)) & (hist.shift(1) >= hist.shift(2)) & (hist > 0))
    m_up, m_dn = _divergence(c, macd)
    call("macd_divergence", up=m_up, down=m_dn)
    # weekly trend as a filter on a daily signal (the book's "combine weeklies and dailies")
    wk_up = c.rolling(50).mean() > c.rolling(50).mean().shift(5)
    call("macd_cross_with_weekly_trend", up=_cross_up(macd, sigl) & wk_up & (c > sma[200]),
         down=_cross_dn(macd, sigl) & ~wk_up & (c < sma[200]))

    call("stochastic_d_leaves_extreme", up=_cross_up(dslow, dslow * 0 + 20), down=_cross_dn(dslow, dslow * 0 + 80))
    # RSI failure swing: a peak above 70, a pullback, a lower peak, then a close under the pullback low (and the mirror)
    fs_dn = np.zeros(n, dtype=bool); fs_up = np.zeros(n, dtype=bool)
    rv_ = rsi.values
    for t in range(30, n):
        w = rv_[t - 30:t]
        if not np.isfinite(w).all():
            continue
        ip = int(np.argmax(w))
        if w[ip] > 70 and ip < 24:
            rest = w[ip + 1:]
            it = int(np.argmin(rest)) + ip + 1
            if it < 29:
                p2 = w[it + 1:].max()
                if p2 < w[ip] and rv_[t] < w[it] and rv_[t - 1] >= w[it]:
                    fs_dn[t] = True
        ib = int(np.argmin(w))
        if w[ib] < 30 and ib < 24:
            rest = w[ib + 1:]
            ip2 = int(np.argmax(rest)) + ib + 1
            if ip2 < 29:
                t2 = w[ip2 + 1:].min()
                if t2 > w[ib] and rv_[t] > w[ip2] and rv_[t - 1] <= w[ip2]:
                    fs_up[t] = True
    call("rsi_failure_swing", up=fs_up, down=fs_dn)
    # weekly MACD as the gatekeeper for a daily MACD signal (weekly 12-26-9 is about 60-130-45 trading days)
    wmacd = c.ewm(span=60, adjust=False).mean() - c.ewm(span=130, adjust=False).mean()
    wsig = wmacd.ewm(span=45, adjust=False).mean()
    call("macd_cross_weekly_agrees", up=_cross_up(macd, sigl) & (wmacd > wsig), down=_cross_dn(macd, sigl) & (wmacd < wsig))
    call("weekly_macd_state", up=wmacd > wsig, down=wmacd < wsig)

    # ---------- Chapter 12: candles, plain and filtered by the stochastic (the book's "filtered candle patterns")
    import ta_signals as video
    vs = video.compute(df)
    body = (c - o).abs()
    avg_body = body.rolling(20).mean().shift(1)
    dn_ctx = (c.shift(1) < c.shift(6)) & (c.shift(1) < sma[10].shift(1))
    up_ctx = (c.shift(1) > c.shift(6)) & (c.shift(1) > sma[10].shift(1))
    long_white_1 = (c.shift(1) > o.shift(1)) & (body.shift(1) > avg_body.shift(1))
    long_black_1 = (c.shift(1) < o.shift(1)) & (body.shift(1) > avg_body.shift(1))
    mid1 = (o.shift(1) + c.shift(1)) / 2
    dark_cloud = up_ctx & long_white_1 & (o > h.shift(1)) & (c < mid1) & (c > o.shift(1))
    piercing = dn_ctx & long_black_1 & (o < l.shift(1)) & (c > mid1) & (c < o.shift(1))
    call("piercing_dark_cloud", up=piercing, down=dark_cloud)
    rev = ["hammer_at_bottom", "hanging_man_at_top", "shooting_star_at_top", "dragonfly_doji_at_bottom",
           "gravestone_doji_at_top", "doji_at_extreme", "engulfing", "morning_evening_star", "tweezer"]
    any_up = piercing.copy(); any_dn = dark_cloud.copy()
    for r_ in rev:
        any_up |= (vs[r_] > 0); any_dn |= (vs[r_] < 0)
    call("any_candle_reversal", up=any_up, down=any_dn)
    call("candle_reversal_filtered_by_stochastic", up=any_up & (dslow < 20), down=any_dn & (dslow > 80))
    call("candle_reversal_filtered_by_rsi", up=any_up & (rsi < 30), down=any_dn & (rsi > 70))

    # ---------- more from chapters 7, 8, 9, 14 and appendix C
    call("obv_new_high_before_price", up=(obv >= obv.rolling(60).max()) & (c < c.rolling(60).max() * 0.98),
         down=(obv <= obv.rolling(60).min()) & (c > c.rolling(60).min() * 1.02))
    rising200 = sma[200] > sma[200].shift(20)
    falling200 = sma[200] < sma[200].shift(20)
    call("pullback_to_200day", up=rising200 & (l <= sma[200] * 1.01) & (c > sma[200]),
         down=falling200 & (h >= sma[200] * 0.99) & (c < sma[200]))
    mo = df["timestamps"].dt.to_period("M")
    last_of_month = mo != mo.shift(-1)
    gm = df.groupby(mo)
    m_high, m_low = gm["high"].transform("max"), gm["low"].transform("min")
    mclose = c.where(last_of_month).dropna()
    prev_mclose = mclose.shift(1).reindex(df.index)
    prev_mhigh = m_high.where(last_of_month).dropna().shift(1).reindex(df.index)
    prev_mlow = m_low.where(last_of_month).dropna().shift(1).reindex(df.index)
    call("monthly_reversal", up=last_of_month & (m_low < prev_mlow) & (c > prev_mclose),
         down=last_of_month & (m_high > prev_mhigh) & (c < prev_mclose))
    # the 4% model from appendix C, on weekly closes
    wc = c.where(last_of_week).dropna()
    state = 0; ext = None; st = {}
    for d_, px in wc.items():
        if ext is None:
            ext = px
        if state <= 0:
            ext = min(ext, px)
            if px >= ext * 1.04:
                state = 1; ext = px
        else:
            ext = max(ext, px)
            if px <= ext * 0.96:
                state = -1; ext = px
        st[d_] = state
    four = pd.Series(st).reindex(df.index).ffill().fillna(0)
    call("four_percent_model", up=four > 0, down=four < 0)
    # left / right translation: did the high of the last low-to-low cycle come before or after its midpoint
    tr_up = np.zeros(n, dtype=bool); tr_dn = np.zeros(n, dtype=bool)
    for a_, b_ in zip(lows[:-1], lows[1:]):
        i0, i1, k1 = a_[0], b_[0], b_[2]
        if i1 - i0 < 10:
            continue
        ih_ = i0 + int(np.argmax(hv[i0:i1 + 1]))
        frac = (ih_ - i0) / (i1 - i0)
        nxt = n
        for c_ in lows:
            if c_[2] > k1:
                nxt = c_[2]; break
        (tr_up if frac > 0.5 else tr_dn)[k1:min(nxt, n)] = True
    call("cycle_translation", up=tr_up, down=tr_dn)
    call("strong_and_weak_months", up=month_of(df).isin([11, 12, 1]), down=month_of(df).isin([9]))

    # ---------- Chapter 15: directional movement
    upm = h.diff(); dnm = -l.diff()
    plus_dm = upm.where((upm > dnm) & (upm > 0), 0.0)
    minus_dm = dnm.where((dnm > upm) & (dnm > 0), 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    mdi = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    call("di_cross", up=_cross_up(pdi, mdi), down=_cross_dn(pdi, mdi))
    call("di_state_adx_rising", up=(pdi > mdi) & (adx > adx.shift(5)) & (adx > 20), down=(mdi > pdi) & (adx > adx.shift(5)) & (adx > 20))

    # ---------- Chapter 14: calendar claims
    ts = df["timestamps"]
    month = ts.dt.month
    call("best_six_months", up=month.isin([11, 12, 1, 2, 3, 4]), down=month.isin([5, 6, 7, 8, 9, 10]))
    yr_in_cycle = ts.dt.year % 4                       # US election years are divisible by 4
    call("presidential_cycle", up=yr_in_cycle.isin([3, 0]), down=yr_in_cycle.isin([1, 2]))
    jan = df[month == 1].groupby(ts.dt.year)["close"].agg(["first", "last"])
    jan_up = (jan["last"] > jan["first"])
    year_sign = ts.dt.year.map(jan_up)
    call("january_barometer", up=(month >= 2) & (year_sign == True), down=(month >= 2) & (year_sign == False))  # noqa: E712

    return pd.DataFrame(s)


SIGNALS = [
    "reversal_day", "reversal_day_outside_heavy_volume", "weekly_reversal", "breakaway_gap", "runaway_gap",
    "exhaustion_gap", "island_reversal", "broken_level_retest", "trendline_break_two_closes", "trendline_break_3pct",
    "retrace_33_50_buy_zone", "retrace_beyond_66", "double_top_bottom", "triple_top_bottom", "head_and_shoulders",
    "rectangle_breakout_40d", "rectangle_breakout_40d_volume", "right_angle_triangle", "symmetrical_triangle_break",
    "flag_pennant", "obv_divergence", "obv_trend_confirms", "selling_climax_blowoff", "close_cross_50day",
    "close_cross_200day", "double_crossover_5_20", "double_crossover_10_50", "double_crossover_10_50_state",
    "triple_crossover_4_9_18", "triple_crossover_4_9_18_state", "bollinger_band_touch_fade", "bollinger_band_touch_follow",
    "bollinger_midline_target", "squeeze_then_break", "envelope_3pct_stretch_fade", "four_week_rule",
    "twelve_week_breakout", "four_week_rule_state", "momentum_10_zero_cross", "rsi_30_70_cross_back", "rsi_with_trend",
    "stochastic_cross_in_extreme", "stochastic_with_trend", "cci_100_follow", "macd_signal_cross",
    "macd_cross_far_from_zero", "macd_histogram_turn", "macd_divergence", "macd_cross_with_weekly_trend", "di_cross",
    "di_state_adx_rising", "best_six_months", "presidential_cycle", "january_barometer",
    "stochastic_d_leaves_extreme", "rsi_failure_swing", "macd_cross_weekly_agrees", "weekly_macd_state",
    "piercing_dark_cloud", "any_candle_reversal", "candle_reversal_filtered_by_stochastic",
    "candle_reversal_filtered_by_rsi", "obv_new_high_before_price", "pullback_to_200day", "monthly_reversal",
    "four_percent_model", "cycle_translation", "strong_and_weak_months",
]
