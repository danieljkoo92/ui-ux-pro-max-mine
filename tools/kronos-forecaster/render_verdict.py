"""Turn Jev's reply (out/jev_response.json) into the notebook's Step 5 verdict text.

Accepts both reply shapes: the gateway's {answers, model, usage} and thejevai.com's
{code, message, data: {result: {answers, usage}}}.

Usage:  .venv/Scripts/python.exe render_verdict.py [RUN_FOLDER, e.g. BTC-USD_90d]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jev_judge as jj

OUT = os.path.join(HERE, "out", sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "out")
with open(os.path.join(OUT, "jev_response.json"), encoding="utf-8") as f:
    body = json.load(f)
with open(os.path.join(OUT, "jev_request.json"), encoding="utf-8") as f:
    request = json.load(f)
with open(os.path.join(OUT, "summary.json"), encoding="utf-8") as f:
    meta = json.load(f)

result = body.get("data", {}).get("result") if "data" in body else body
if not result or "answers" not in result:
    raise SystemExit(f"No answers in the reply: {json.dumps(body)[:300]}")
result.setdefault("model", request["model"])

text = jj.verdict_english(result, {"ticker": meta["ticker"], "horizon_days": meta["horizon_days"]})
print(text)
with open(os.path.join(OUT, "jev_verdict.txt"), "w", encoding="utf-8") as f:
    f.write(text + "\n")
