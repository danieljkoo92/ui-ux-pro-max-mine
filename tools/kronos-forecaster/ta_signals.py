"""Chart-reading signals from the two videos, written so they can be tested on daily candles.

Video 1 (Ross Cameron, "How to Read Candlestick Shapes & Charts"): single, two and three candle shapes,
each read in context (a hammer only counts at the bottom of a drop, and so on).
Video 2 (The Secret Mindset, "Every Trading Strategy in Technical Analysis RANKED"): market structure,
moving averages, volume, support and resistance, volume by price, RSI.

Left out because daily candles cannot compute them: footprint charts, order flow, market profile,
VWAP and pivot points (all need trades or candles inside the day).
Left out because the video itself ranks them useless: moon phases, Gann angles, Elliott waves,
harmonic patterns, diagonal trend lines, parabolic SAR, Heikin-Ashi.

Every signal is a call made at the close of a day using only that day and earlier days:
+1 means "up from here", -1 means "down from here", 0 means no call.
"""
import numpy as np
import pandas as pd

SWING_K = 5          # a swing high is the highest high of the 5 days either side; known 5 days later
PROFILE_DAYS = 120   # window for volume by price
PROFILE_BINS = 50


def _atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _swings(high, low, k=SWING_K):
    """For every day: the last two swing highs and swing lows that were already known on that day."""
    n = len(high)
    sh1 = np.full(n, np.nan); sh0 = np.full(n, np.nan)
    sl1 = np.full(n, np.nan); sl0 = np.full(n, np.nan)
    lows_hist = [[] for _ in range(n)]
    highs_hist = [[] for _ in range(n)]
    h, l = high.values, low.values
    a1 = a0 = b1 = b0 = np.nan
    recent_lows, recent_highs = [], []
    for t in range(n):
        c = t - k                      # the candidate swing day that becomes known today
        if c - k >= 0:
            if h[c] == h[c - k:c + k + 1].max():
                a0, a1 = a1, h[c]
                recent_highs.append((c, h[c]))
            if l[c] == l[c - k:c + k + 1].min():
                b0, b1 = b1, l[c]
                recent_lows.append((c, l[c]))
        sh1[t], sh0[t], sl1[t], sl0[t] = a1, a0, b1, b0
        recent_lows = [x for x in recent_lows if t - x[0] <= 250]
        recent_highs = [x for x in recent_highs if t - x[0] <= 250]
        lows_hist[t] = [x[1] for x in recent_lows]
        highs_hist[t] = [x[1] for x in recent_highs]
    return sh1, sh0, sl1, sl0, lows_hist, highs_hist


def _volume_profile_poc(df, stride=5):
    """Price level where the most volume traded over the last PROFILE_DAYS (each day's volume spread over its range)."""
    n = len(df)
    poc = np.full(n, np.nan)
    h, l, v = df["high"].values, df["low"].values, df["volume"].values
    for t in range(PROFILE_DAYS, n, stride):
        hh, ll, vv = h[t - PROFILE_DAYS + 1:t + 1], l[t - PROFILE_DAYS + 1:t + 1], v[t - PROFILE_DAYS + 1:t + 1]
        lo, hi = ll.min(), hh.max()
        if not np.isfinite(lo) or hi <= lo:
            continue
        edges = np.linspace(lo, hi, PROFILE_BINS + 1)
        left = np.clip(np.maximum(edges[None, :-1], ll[:, None]), None, None)
        right = np.minimum(edges[None, 1:], hh[:, None])
        overlap = np.clip(right - left, 0, None)
        width = np.where(hh > ll, hh - ll, np.nan)[:, None]
        share = np.where(np.isfinite(width), overlap / width, 0.0)
        hist = (share * vv[:, None]).sum(axis=0)
        i = int(np.argmax(hist))
        poc[t] = (edges[i] + edges[i + 1]) / 2
    return pd.Series(poc, index=df.index).ffill(limit=stride - 1)


def compute(df):
    """df: open, high, low, close, volume (daily, oldest first). Returns a DataFrame of calls, one column per signal."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    body = (c - o).abs()
    rng = (h - l)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    green, red = c > o, c < o
    atr = _atr(df)
    avg_body = body.rolling(20).mean().shift(1)
    sma10, sma20, sma50, sma200 = (c.rolling(n).mean() for n in (10, 20, 50, 200))
    ret = c.pct_change()

    # context for a shape: what price was doing BEFORE the candle
    dn_ctx = (c.shift(1) < c.shift(6)) & (c.shift(1) < sma10.shift(1))
    up_ctx = (c.shift(1) > c.shift(6)) & (c.shift(1) > sma10.shift(1))

    doji = (body <= 0.1 * rng) & (rng > 0)
    gravestone = doji & (lower <= 0.1 * rng) & (upper >= 0.6 * rng)
    dragonfly = doji & (upper <= 0.1 * rng) & (lower >= 0.6 * rng)
    spinning = (body <= 0.3 * rng) & (upper >= 0.25 * rng) & (lower >= 0.25 * rng) & ~doji
    hammer_shape = (lower >= 2 * body) & (upper <= 0.15 * rng) & (body > 0)
    inv_hammer_shape = (upper >= 2 * body) & (lower <= 0.15 * rng) & (body > 0)
    long_body = (body > 1.5 * avg_body) & (body >= 0.7 * rng)
    small_body = body < 0.5 * avg_body

    s = {}

    def call(name, up=None, down=None):
        x = pd.Series(0, index=df.index, dtype="int8")
        if up is not None:
            x[up.fillna(False)] = 1
        if down is not None:
            x[down.fillna(False)] = -1
        s[name] = x

    # ---- Video 1: candle shapes
    call("long_body_candle", up=long_body & green, down=long_body & red)
    call("hammer_at_bottom", up=hammer_shape & dn_ctx)
    call("hanging_man_at_top", down=hammer_shape & up_ctx)
    call("shooting_star_at_top", down=inv_hammer_shape & up_ctx)
    call("inverted_hammer_at_bottom", down=inv_hammer_shape & dn_ctx)      # the video reads this one as weak
    call("dragonfly_doji_at_bottom", up=dragonfly & dn_ctx)
    call("gravestone_doji_at_top", down=gravestone & up_ctx)
    call("doji_at_extreme", up=doji & dn_ctx, down=doji & up_ctx)
    call("spinning_top_at_extreme", up=spinning & dn_ctx, down=spinning & up_ctx)
    eng_up = dn_ctx & red.shift(1) & green & (o <= c.shift(1) * 1.001) & (c >= o.shift(1)) & (body > body.shift(1))
    eng_dn = up_ctx & green.shift(1) & red & (o >= c.shift(1) * 0.999) & (c <= o.shift(1)) & (body > body.shift(1))
    call("engulfing", up=eng_up, down=eng_dn)
    first_red_long = red.shift(2) & (body.shift(2) > avg_body.shift(2))
    first_green_long = green.shift(2) & (body.shift(2) > avg_body.shift(2))
    star_small = small_body.shift(1)
    morning = (dn_ctx.shift(2) & first_red_long & star_small
               & (np.maximum(o, c).shift(1) <= c.shift(2)) & green & (c > (o.shift(2) + c.shift(2)) / 2))
    evening = (up_ctx.shift(2) & first_green_long & star_small
               & (np.minimum(o, c).shift(1) >= c.shift(2)) & red & (c < (o.shift(2) + c.shift(2)) / 2))
    call("morning_evening_star", up=morning, down=evening)
    strong_g = green & (body > 0.8 * avg_body) & ((h - c) <= 0.3 * rng)
    strong_r = red & (body > 0.8 * avg_body) & ((c - l) <= 0.3 * rng)
    soldiers = strong_g & strong_g.shift(1) & strong_g.shift(2) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    crows = strong_r & strong_r.shift(1) & strong_r.shift(2) & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    call("three_soldiers_or_crows", up=soldiers, down=crows)
    tw_bot = dn_ctx & ((l - l.shift(1)).abs() <= 0.1 * atr) & (lower >= body) & (lower.shift(1) >= body.shift(1)) & green
    tw_top = up_ctx & ((h - h.shift(1)).abs() <= 0.1 * atr) & (upper >= body) & (upper.shift(1) >= body.shift(1)) & red
    call("tweezer", up=tw_bot, down=tw_top)
    # rising three / bull flag: a long candle, 2 to 4 small candles pulling back but holding, then the first new high
    flag_up = pd.Series(False, index=df.index)
    flag_dn = pd.Series(False, index=df.index)
    for n in (2, 3, 4):
        pole_g = (long_body & green).shift(n + 1)
        pole_r = (long_body & red).shift(n + 1)
        hold_up = pd.Series(True, index=df.index)
        hold_dn = pd.Series(True, index=df.index)
        for j in range(1, n + 1):
            hold_up &= (l.shift(j) > l.shift(n + 1)) & (body.shift(j) < body.shift(n + 1)) & (c.shift(j) <= h.shift(n + 1))
            hold_dn &= (h.shift(j) < h.shift(n + 1)) & (body.shift(j) < body.shift(n + 1)) & (c.shift(j) >= l.shift(n + 1))
        pull_up = c.shift(1) < c.shift(n)  if n > 1 else True
        pull_dn = c.shift(1) > c.shift(n) if n > 1 else True
        flag_up |= pole_g & hold_up & pull_up & green & (h > h.shift(1)) & (c > h.shift(1))
        flag_dn |= pole_r & hold_dn & pull_dn & red & (l < l.shift(1)) & (c < l.shift(1))
    call("rising_falling_three", up=flag_up, down=flag_dn)
    grow_up = green & green.shift(1) & green.shift(2) & (body > body.shift(1)) & (body.shift(1) > body.shift(2))
    grow_dn = red & red.shift(1) & red.shift(2) & (body > body.shift(1)) & (body.shift(1) > body.shift(2))
    call("bodies_growing", up=grow_up, down=grow_dn)
    fade_up = green & green.shift(1) & green.shift(2) & (body < body.shift(1)) & (body.shift(1) < body.shift(2)) & up_ctx
    fade_dn = red & red.shift(1) & red.shift(2) & (body < body.shift(1)) & (body.shift(1) < body.shift(2)) & dn_ctx
    call("bodies_shrinking_exhaustion", up=fade_dn, down=fade_up)

    # ---- Video 2: market structure
    sh1, sh0, sl1, sl0, lows_hist, highs_hist = _swings(h, l)
    sh1, sh0, sl1, sl0 = (pd.Series(x, index=df.index) for x in (sh1, sh0, sl1, sl0))
    bull_struct = (sh1 > sh0) & (sl1 > sl0)
    bear_struct = (sh1 < sh0) & (sl1 < sl0)
    call("market_structure", up=bull_struct, down=bear_struct)
    call("break_of_structure", up=bull_struct & (c > sh1) & (c.shift(1) <= sh1),
         down=bear_struct & (c < sl1) & (c.shift(1) >= sl1))
    call("change_of_character", up=bear_struct & (c > sh1) & (c.shift(1) <= sh1),
         down=bull_struct & (c < sl1) & (c.shift(1) >= sl1))

    # ---- moving averages
    call("ma50_trend", up=(c > sma50) & (sma50 > sma50.shift(10)), down=(c < sma50) & (sma50 < sma50.shift(10)))
    call("ma200_trend", up=(c > sma200) & (sma200 > sma200.shift(20)), down=(c < sma200) & (sma200 < sma200.shift(20)))
    call("golden_death_cross_state", up=sma50 > sma200, down=sma50 < sma200)
    call("golden_death_cross_event", up=(sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1)),
         down=(sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1)))

    # ---- volume
    rel_vol = v / v.rolling(20).mean().shift(1)
    big_up = (c - c.shift(1)) > 1.5 * atr.shift(1)
    big_dn = (c.shift(1) - c) > 1.5 * atr.shift(1)
    call("big_move_big_volume", up=big_up & (rel_vol > 1.5), down=big_dn & (rel_vol > 1.5))
    call("big_move_thin_volume_trap", up=big_dn & (rel_vol < 0.8), down=big_up & (rel_vol < 0.8))

    # ---- support and resistance
    hi52 = h.rolling(250).max().shift(1)
    lo52 = l.rolling(250).min().shift(1)
    call("new_52_week_extreme", up=c > hi52, down=c < lo52)
    av, lv, hv, cv, ov = atr.values, l.values, h.values, c.values, o.values
    bounce = np.zeros(len(df), dtype=bool)
    reject = np.zeros(len(df), dtype=bool)
    for t in range(len(df)):
        a = av[t]
        if not np.isfinite(a) or a <= 0:
            continue
        lows = lows_hist[t]
        if len(lows) >= 2:
            near = [x for x in lows if abs(lv[t] - x) <= 0.5 * a]
            if near:
                lvl = float(np.mean(near))
                touches = sum(1 for x in lows if abs(x - lvl) <= a)
                bounce[t] = touches >= 2 and cv[t] > lvl and cv[t] > ov[t]
        highs = highs_hist[t]
        if len(highs) >= 2:
            near = [x for x in highs if abs(hv[t] - x) <= 0.5 * a]
            if near:
                lvl = float(np.mean(near))
                touches = sum(1 for x in highs if abs(x - lvl) <= a)
                reject[t] = touches >= 2 and cv[t] < lvl and cv[t] < ov[t]
    call("support_bounce_resistance_reject", up=pd.Series(bounce, index=df.index), down=pd.Series(reject, index=df.index))

    # ---- volume by price
    poc = _volume_profile_poc(df)
    call("price_vs_volume_node", up=c > poc + atr, down=c < poc - atr)             # which side of the busiest price
    call("pull_to_volume_node", up=c < poc - 3 * atr, down=c > poc + 3 * atr)      # the "magnet" reading

    # ---- RSI
    rsi = _rsi(c)
    call("rsi_strength", up=rsi > 70, down=rsi < 30)       # the video's reading: overbought confirms strength
    hi20 = c >= c.rolling(20).max()
    lo20 = c <= c.rolling(20).min()
    rv, cvv = rsi.values, c.values
    div_dn = np.zeros(len(df), dtype=bool)
    div_up = np.zeros(len(df), dtype=bool)
    last_hi = last_lo = None
    hi_v, lo_v = hi20.values, lo20.values
    for t in range(len(df)):
        if hi_v[t]:
            if last_hi is not None and 5 <= t - last_hi <= 60 and cvv[t] > cvv[last_hi] and rv[t] < rv[last_hi]:
                div_dn[t] = True
            last_hi = t
        if lo_v[t]:
            if last_lo is not None and 5 <= t - last_lo <= 60 and cvv[t] < cvv[last_lo] and rv[t] > rv[last_lo]:
                div_up[t] = True
            last_lo = t
    call("rsi_divergence", up=pd.Series(div_up, index=df.index), down=pd.Series(div_dn, index=df.index))

    out = pd.DataFrame(s)
    # extras the range model uses (not direction calls)
    bw = (4 * c.rolling(20).std()) / sma20
    out.attrs["squeeze"] = (bw <= bw.rolling(120).quantile(0.10))
    out.attrs["atr_pct"] = atr / c
    return out


SIGNALS = [
    "long_body_candle", "hammer_at_bottom", "hanging_man_at_top", "shooting_star_at_top", "inverted_hammer_at_bottom",
    "dragonfly_doji_at_bottom", "gravestone_doji_at_top", "doji_at_extreme", "spinning_top_at_extreme", "engulfing",
    "morning_evening_star", "three_soldiers_or_crows", "tweezer", "rising_falling_three", "bodies_growing",
    "bodies_shrinking_exhaustion", "market_structure", "break_of_structure", "change_of_character", "ma50_trend",
    "ma200_trend", "golden_death_cross_state", "golden_death_cross_event", "big_move_big_volume",
    "big_move_thin_volume_trap", "new_52_week_extreme", "support_bounce_resistance_reject", "price_vs_volume_node",
    "pull_to_volume_node", "rsi_strength", "rsi_divergence",
]
