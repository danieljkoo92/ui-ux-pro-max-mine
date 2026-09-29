"""Judge every chart rule on a whole market: the stock market, or commodity futures.

Same judge as evaluate_signals.py (credit only for what a rule adds on top of the ticker's normal drift,
luck measured by reshuffling blocks of calendar time, pass the older years and then the newer years),
rebuilt to add results up ticker by ticker so hundreds of tickers fit in memory.

Usage:  .venv/Scripts/python.exe evaluate_universe.py stocks
        .venv/Scripts/python.exe evaluate_universe.py commodities
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import market_data as md
import ta_signals as video
import ta_signals_book as book

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = sys.argv[1] if len(sys.argv) > 1 else "stocks"
HORIZONS = [5, 30, 90]
SPLIT = pd.Timestamp("2019-01-01")
MIN_FIRES = 200
MIN_DAYS = 1000
BOOT = 4000
ALPHA = 0.05
NB = 400
rng = np.random.default_rng(7)

names = list(video.SIGNALS) + list(book.SIGNALS)
source = {n: "videos" for n in video.SIGNALS}
source.update({n: "book" for n in book.SIGNALS})
ns, nh = len(names), len(HORIZONS)
S = np.zeros((ns, nh, 2 * NB)); N = np.zeros((ns, nh, 2 * NB))
HIT = np.zeros((ns, nh, 2)); BASE = np.zeros((ns, nh, 2))
SIDE_S = np.zeros((ns, nh, 2, 2)); SIDE_N = np.zeros((ns, nh, 2, 2))      # [signal, horizon, split, up/down]

tickers = md.fetch_many(md.universe(NAME), max_age_hours=72)
used, days = [], 0
t0 = time.time()
for k, t in enumerate(tickers):
    try:
        df = md.load(t, max_age_hours=72)
    except Exception:
        continue
    if len(df) < MIN_DAYS:
        continue
    sig = pd.concat([video.compute(df)[list(video.SIGNALS)], book.compute(df)[list(book.SIGNALS)]], axis=1)
    c = df["close"]
    ts = df["timestamps"]
    yr, mo = ts.dt.year.values - 2000, ts.dt.month.values
    used.append(t); days += len(df)
    for j, h in enumerate(HORIZONS):
        fwd = np.log(c.shift(-h) / c).values
        end = ts.shift(-h)
        split = np.where((end < SPLIT).values, 0, np.where((ts >= SPLIT).values, 1, -1))
        split[~np.isfinite(fwd)] = -1
        blk = yr * 12 + mo - 1 if h <= 5 else (yr * 4 + (mo - 1) // 3 if h <= 30 else yr * 2 + (mo > 6))
        excess = np.full(len(df), np.nan); p_up = np.full(len(df), np.nan)
        for sp in (0, 1):
            m = split == sp
            if m.sum() > 50:
                excess[m] = fwd[m] - fwd[m].mean()
                p_up[m] = (fwd[m] > 0).mean()
        ok = (split >= 0) & np.isfinite(excess)
        bins = split * NB + blk
        sv = sig.values
        for i in range(ns):
            s = sv[:, i]
            m = ok & (s != 0)
            if not m.any():
                continue
            sm, em, bm, fm, pm, spm = s[m].astype(float), excess[m], bins[m], fwd[m], p_up[m], split[m]
            signed = sm * em
            S[i, j] += np.bincount(bm, weights=signed, minlength=2 * NB)
            N[i, j] += np.bincount(bm, minlength=2 * NB)
            hit = (sm * fm > 0).astype(float)
            base = np.where(sm > 0, pm, 1 - pm)
            for sp in (0, 1):
                q = spm == sp
                HIT[i, j, sp] += hit[q].sum(); BASE[i, j, sp] += base[q].sum()
                for d, side in enumerate((1.0, -1.0)):
                    qq = q & (sm == side)
                    SIDE_S[i, j, sp, d] += signed[qq].sum(); SIDE_N[i, j, sp, d] += qq.sum()
    if (k + 1) % 25 == 0 or k + 1 == len(tickers):
        print(f"  {k + 1}/{len(tickers)} tickers, {time.time() - t0:.0f}s", flush=True)


def luck_p(s, n):
    keep = n > 0
    s, n = s[keep], n[keep]
    if len(s) < 8:
        return float("nan"), int(len(s))
    idx = rng.integers(0, len(s), size=(BOOT, len(s)))
    means = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float((means <= 0).mean()), int(len(s))


rows = []
for i, name in enumerate(names):
    for j, h in enumerate(HORIZONS):
        rec = {"signal": name, "source": source[name], "horizon": h}
        for sp, lab in ((0, "tune"), (1, "holdout")):
            s, n = S[i, j, sp * NB:(sp + 1) * NB], N[i, j, sp * NB:(sp + 1) * NB]
            tot = n.sum()
            rec[f"{lab}_n"] = int(tot)
            if tot < 30:
                rec[f"{lab}_edge_pct"] = np.nan; rec[f"{lab}_p"] = np.nan
                continue
            p, nb = luck_p(s, n)
            rec[f"{lab}_edge_pct"] = round(100 * s.sum() / tot, 3)
            rec[f"{lab}_p"] = p
            rec[f"{lab}_blocks"] = nb
            rec[f"{lab}_hit"] = round(HIT[i, j, sp] / tot, 4)
            rec[f"{lab}_base_hit"] = round(BASE[i, j, sp] / tot, 4)
            for d, side in enumerate(("up", "down")):
                cn = SIDE_N[i, j, sp, d]
                rec[f"{lab}_{side}_n"] = int(cn)
                rec[f"{lab}_{side}_edge_pct"] = round(100 * SIDE_S[i, j, sp, d] / cn, 3) if cn >= 30 else np.nan
        rows.append(rec)
res = pd.DataFrame(rows)


def luck_line(splits):
    """The luck line for the BEST of all rules (White's Reality Check, as in Aronson's book).

    Pretend no rule has any edge, reshuffle calendar blocks, and each time record how good the best-looking
    rule was. A rule clears the line only if it beats what the best of all the worthless rules scored.
    Scores are divided by their own wobble first, so a rule that rarely fires cannot win on noise alone.
    """
    out = {}
    for j, h in enumerate(HORIZONS):
        cols = np.concatenate([np.arange(sp * NB, (sp + 1) * NB) for sp in splits])
        Sj, Nj = S[:, j, cols], N[:, j, cols]
        live = Nj.sum(axis=0) > 0
        Sj, Nj = Sj[:, live], Nj[:, live]
        ok = np.where(Nj.sum(axis=1) >= MIN_FIRES)[0]
        nb = Sj.shape[1]
        if nb < 8 or len(ok) == 0:
            continue
        idx = rng.integers(0, nb, size=(BOOT, nb))
        obs = Sj[ok].sum(axis=1) / Nj[ok].sum(axis=1)
        null = np.full((len(ok), BOOT), np.nan)
        t_obs = np.full(len(ok), np.nan)
        for r, i in enumerate(ok):
            den = Nj[i][idx].sum(axis=1)
            mean = np.where(den > 0, Sj[i][idx].sum(axis=1) / np.maximum(den, 1), np.nan)
            se = np.nanstd(mean)
            if se > 0:
                t_obs[r] = obs[r] / se
                null[r] = (mean - obs[r]) / se
        best = np.nanmax(null, axis=0)
        for r, i in enumerate(ok):
            if np.isfinite(t_obs[r]):
                out[(names[i], h)] = min(1.0, len(HORIZONS) * float((best >= t_obs[r]).mean()))
    return out


ll_new, ll_all = luck_line([1]), luck_line([0, 1])
res["luck_line_p_newer"] = [ll_new.get((a, b), np.nan) for a, b in zip(res["signal"], res["horizon"])]
res["luck_line_p_all_years"] = [ll_all.get((a, b), np.nan) for a, b in zip(res["signal"], res["horizon"])]
res["tune_pass"] = (res["tune_n"] >= MIN_FIRES) & (res["tune_p"] < ALPHA) & (res["tune_edge_pct"] > 0)
m = int(res["tune_pass"].sum())
bar = ALPHA / max(m, 1)
res["final_pass"] = res["tune_pass"] & (res["holdout_n"] >= MIN_FIRES) & (res["holdout_p"] < bar) & (res["holdout_edge_pct"] > 0)
# big enough to matter after the cost of trading in and out
need = res["horizon"].map({5: 0.2, 30: 0.5, 90: 0.5})
res["big_enough"] = res["final_pass"] & (res["holdout_edge_pct"] >= need) & (res["tune_edge_pct"] >= need)

out_dir = os.path.join(HERE, "out", "signals")
os.makedirs(out_dir, exist_ok=True)
res.to_csv(os.path.join(out_dir, f"universe_{NAME}_report.csv"), index=False)
cols = ["signal", "source", "horizon", "tune_n", "tune_edge_pct", "tune_p", "holdout_n", "holdout_edge_pct", "holdout_p",
        "holdout_hit", "holdout_base_hit", "holdout_up_edge_pct", "holdout_down_edge_pct", "luck_line_p_newer",
        "luck_line_p_all_years", "final_pass", "big_enough"]
summary = {
    "universe": NAME, "tickers": len(used), "days": int(days), "rules": ns, "tests": int(len(res)),
    "expected_tune_passes_by_luck": round(ALPHA * len(res), 1), "passed_tune": m, "holdout_bar_p": bar,
    "passed_both": int(res["final_pass"].sum()), "passed_and_big_enough": int(res["big_enough"].sum()),
    "passers": json.loads(res[res["final_pass"]][cols].to_json(orient="records")),
    "beat_the_luck_line_all_years": json.loads(res[res["luck_line_p_all_years"] < ALPHA][cols].to_json(orient="records")),
    "beat_the_luck_line_newer_years": json.loads(res[res["luck_line_p_newer"] < ALPHA][cols].to_json(orient="records")),
}
with open(os.path.join(out_dir, f"universe_{NAME}_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
print("\nPASSED THE OLDER YEARS")
print(res[res["tune_pass"]][cols].sort_values(["horizon", "tune_p"]).to_string(index=False))
print()
print("BEAT THE LUCK LINE (best of all rules), all years")
print(res[res["luck_line_p_all_years"] < ALPHA][cols].sort_values("luck_line_p_all_years").to_string(index=False))
print()
print("BEAT THE LUCK LINE, newer years only")
print(res[res["luck_line_p_newer"] < ALPHA][cols].sort_values("luck_line_p_newer").to_string(index=False))
print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))
