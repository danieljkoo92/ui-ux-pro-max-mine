"""How much more likely is price to TOUCH a level during the window than to END beyond it?

The range model says where price ends. People usually ask something else: "will it get down to X at any
point in the next 90 days?" This measures, on real history, how often price touched the forecast's 10% and
25% levels at any time, against how often it finished beyond them. The ratio is measured on the tune years
and checked on the holdout years.

Usage:  .venv/Scripts/python.exe touch_check.py
"""
import json
import os

import numpy as np
import pandas as pd

import market_data as md
import range_model as rm

HERE = os.path.dirname(os.path.abspath(__file__))
chosen = json.load(open(os.path.join(HERE, "out", "range", "chosen.json")))
KW = {"mix": dict(method="mix"), "mix_own_center": dict(method="mix_own_center"), "normal": dict(method="normal"), "own": dict(method="own")}


def method_for(h, cls):
    name = chosen[f"{cls}_{30 if h <= 45 else 90}"]
    m, vol = name.split("/")
    k = float(vol.split(" x")[1]) if " x" in vol else 1.0
    return dict(vol_name=vol.split(" x")[0], k=k, **KW[m])


def main():
    rows = []
    for t in [x for v in md.UNIVERSE.values() for x in v]:
        df = md.load(t)
        c = df["close"]
        lc = np.log(c.values)
        for h in (30, 90):
            qf = rm.forecast(c, h, **method_for(h, md.asset_class(t)))
            for i in range(260, len(df) - h, 10):
                q = qf.iloc[i]
                if q.isna().any():
                    continue
                path = lc[i + 1:i + h + 1] - lc[i]
                end = path[-1]
                rows.append({
                    "h": h, "cls": md.asset_class(t),
                    "split": "tune" if df["timestamps"].iloc[i + h] < rm.SPLIT else ("holdout" if df["timestamps"].iloc[i] >= rm.SPLIT else "gap"),
                    "end_below10": end < q[0.10], "touch_below10": path.min() < q[0.10],
                    "end_below25": end < q[0.25], "touch_below25": path.min() < q[0.25],
                    "end_above90": end > q[0.90], "touch_above90": path.max() > q[0.90],
                    "end_above75": end > q[0.75], "touch_above75": path.max() > q[0.75],
                })
    d = pd.DataFrame(rows)
    d = d[d["split"] != "gap"]
    out = {}
    for keys, name in ((["h", "split"], "all"), (["h", "split", "cls"], None)):
        g = d.groupby(keys).mean(numeric_only=True).reset_index()
        for _, r in g.iterrows():
            down = (r["touch_below10"] + r["touch_below25"]) / (r["end_below10"] + r["end_below25"])
            up = (r["touch_above90"] + r["touch_above75"]) / (r["end_above90"] + r["end_above75"])
            label = name or r["cls"]
            out[f"{label}_{int(r['h'])}_{r['split']}"] = {"touch_vs_end_down": round(float(down), 3), "touch_vs_end_up": round(float(up), 3)}
    with open(os.path.join(HERE, "out", "range", "touch.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
