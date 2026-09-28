"""Jev judgment layer for the Kronos Forecaster.

Hands the Kronos forecast plus simple price statistics to Jev, TypeSafe AI's System One decision model,
and gets back typed, calibrated answers. Jev judges; Kronos forecasts; code executes.

Access route: Vercel AI Gateway's TypeSafe-compatible API, exactly as documented by TypeSafe
(docs.typesafe.ai/sdk/python/usage, "Configuring the base URL", Vercel tab):
    base_url = "https://ai-gateway.vercel.sh/typesafe", model = "typesafe-ai/jev"
Request/response shapes follow docs.typesafe.ai/api (POST /v1/systemone).
"""
import os
import numpy as np
import pandas as pd
import requests

JEV_ENDPOINT = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"   # official Vercel AI Gateway route
JEV_MODEL = "typesafe-ai/jev"
FLAT_BAND_PCT = 2.0


def get_gateway_key():
    """Vercel AI Gateway key: environment variable, then Colab Secrets, then a hidden paste prompt."""
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key:
        try:
            from google.colab import userdata  # type: ignore
            key = userdata.get("AI_GATEWAY_API_KEY")
        except Exception:
            key = None
    if not key:
        import getpass
        key = getpass.getpass("Paste your Vercel AI Gateway key (hidden): ").strip()
    return key


def _pct(a, b):
    return float((a / b - 1.0) * 100.0)


def build_state(df, summary, score=None):
    """Compact, numbers-only snapshot. Jev sees exactly this and nothing else."""
    c = df["close"].astype(float).values
    last = float(c[-1])
    rets = np.diff(np.log(c[-21:]))
    vol_20 = float(np.std(rets) * np.sqrt(252) * 100) if len(rets) > 2 else None
    window = c[-60:]
    peak = np.maximum.accumulate(window)
    max_dd_60 = float(((window / peak) - 1.0).min() * 100)
    state = {
        "ticker": summary["ticker"],
        "as_of": str(pd.Timestamp(df["timestamps"].iloc[-1]).date()),
        "horizon_trading_days": int(summary["horizon_days"]),
        "flat_band_pct": FLAT_BAND_PCT,
        "price": {
            "last_close": round(last, 4),
            "change_5d_pct": round(_pct(last, c[-6]), 2) if len(c) > 6 else None,
            "change_20d_pct": round(_pct(last, c[-21]), 2) if len(c) > 21 else None,
            "change_60d_pct": round(_pct(last, c[-61]), 2) if len(c) > 61 else None,
            "vs_50d_average_pct": round(_pct(last, c[-50:].mean()), 2) if len(c) >= 50 else None,
            "vs_200d_average_pct": round(_pct(last, c[-200:].mean()), 2) if len(c) >= 200 else None,
            "volatility_20d_annualized_pct": round(vol_20, 1) if vol_20 is not None else None,
            "max_drawdown_60d_pct": round(max_dd_60, 2),
            "last_10_closes": [round(float(v), 4) for v in c[-10:]],
        },
        "kronos_forecast": {
            "model": f"Kronos candlestick foundation model, {summary['paths']} simulated paths",
            "median_change_pct": round(summary["median_change"] * 100, 2),
            "p10_change_pct": round(summary["p10_change"] * 100, 2),
            "p90_change_pct": round(summary["p90_change"] * 100, 2),
            "share_of_paths_ending_higher": round(summary["prob_up"], 2),
        },
    }
    if score is not None:
        state["kronos_reality_check_on_previous_period"] = {
            "direction_correct": bool(score["direction_right"]),
            "model_said_pct": round(score["median_change"] * 100, 2),
            "market_did_pct": round(score["actual_change"] * 100, 2),
            "final_price_inside_80pct_band": bool(score["end_inside_80"]),
            "share_of_days_inside_band": round(score["days_inside_80"], 2),
            "avg_miss_of_median_pct": round(score["mape_median"] * 100, 2),
        }
    return state


def build_questions(horizon):
    return {
        "direction": {
            "type": "choice",
            "instructions": (
                f"Over the next {horizon} trading days, which outcome is most likely for `ticker`? "
                "Weigh the `price` statistics, the `kronos_forecast`, and, if present, how Kronos did on the "
                "`kronos_reality_check_on_previous_period`."
            ),
            "criteria": {
                "up": f"Closes more than {FLAT_BAND_PCT}% above `price.last_close`",
                "flat": f"Closes within {FLAT_BAND_PCT}% of `price.last_close`",
                "down": f"Closes more than {FLAT_BAND_PCT}% below `price.last_close`",
            },
        },
        "trust": {
            "type": "score",
            "instructions": "How much weight should a cautious investor give this Kronos forecast?",
            "criteria": [
                "Ignore it: the reality check failed badly or the 80% range is too wide to act on",
                "Weak hint: directionally plausible, but the range is wide or the reality check was mixed",
                "Worth watching: consistent with recent price action and a decent reality check",
                "Strong signal: tight range, good reality check, and price action clearly agrees",
            ],
        },
        "regime": {
            "type": "choice",
            "instructions": "Which best describes the current price regime in `price`?",
            "criteria": {
                "trending_up": "Steady gains, above its averages, moderate volatility",
                "trending_down": "Steady losses, below its averages, moderate volatility",
                "sideways_choppy": "No clear direction, mean-reverting moves",
                "volatile_shock": "Unusually large recent moves or drawdown; volatility elevated",
            },
        },
        "agrees": {
            "type": "noul",
            "instructions": "Does the recent price action in `price` support the direction of `kronos_forecast.median_change_pct`?",
            "criteria": {"true": "Momentum and trend point the same way as the forecast",
                         "false": "They disagree or the forecast is essentially flat"},
        },
    }


def ask_jev(state, questions, api_key, timeout=60):
    payload = {"model": JEV_MODEL, "state": state, "questions": questions}
    r = requests.post(JEV_ENDPOINT, json=payload, timeout=timeout,
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    if r.status_code != 200:
        body = r.text[:400]
        if "customer_verification_required" in body:
            raise RuntimeError("Vercel needs a payment card on the team before AI Gateway works (it stays free: $5 of "
                               "monthly credit). Add a card at vercel.com -> Settings -> Billing, then run this cell again.")
        if r.status_code in (401, 403):
            raise RuntimeError(f"The AI Gateway key was rejected ({r.status_code}). Check the key and try again. Details: {body}")
        raise RuntimeError(f"Jev call failed with HTTP {r.status_code}: {body}")
    return r.json()


def jev_verdict(df, summary, score=None, api_key=None):
    api_key = api_key or get_gateway_key()
    state = build_state(df, summary, score)
    out = ask_jev(state, build_questions(summary["horizon_days"]), api_key)
    return out, state


def _bar(p):
    n = int(round(p * 20))
    return "#" * n + "." * (20 - n)


def verdict_english(out, summary):
    a = out["answers"]
    d, t, r, g = a["direction"], a["trust"], a["regime"], a["agrees"]
    probs = d["probabilities"]
    trust_level = t["legend"][str(int(round(t["score"])))].split(":")[0]
    lines = [
        f"Jev's verdict on {summary['ticker']}, next {summary['horizon_days']} trading days   (model: {out.get('model', '?')})",
        "",
        f"  Direction call: {d['choice'].upper()}   confidence {d['confidence']:.2f}",
    ]
    for k in ("up", "flat", "down"):
        lines.append(f"     {k:<5} {_bar(probs.get(k, 0))} {probs.get(k, 0):.0%}")
    lines += [
        f"  Trust in the Kronos forecast: {trust_level}  ({t['score']:.1f} on a 0-3 scale, confidence {t['confidence']:.2f})",
        f"  Price regime: {r['choice'].replace('_', ' ')}  ({r['probabilities'][r['choice']]:.0%})",
        f"  Price action agrees with Kronos: {g['noul']:.0%} yes",
        "",
        "  Jev judges, Kronos forecasts, you decide. Jev only sees the numbers in the snapshot: no news, no earnings, no macro.",
        f"  Cost of this call: about {out.get('usage', {}).get('input_tokens', 0)} tokens (fractions of a cent).",
    ]
    return "\n".join(lines)
