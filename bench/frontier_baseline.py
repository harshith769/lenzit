"""Frontier-model baseline: can Gemini itself tell AI-faked damage photos from real ones?

Same rows as Lenzit-Bench v1 (data/fb_bench/scores.csv), stripped variant (JPEG 75, no metadata), so Gemini sees
exactly what S2 sees. Two prompts: (A) evidence photo only; (B) evidence + seller reference + claim text.
Comparison is fair: Lenzit's S2 recall is read at Gemini's OWN false-alarm rate on real-damage photos.
Resumable (appends to data/fb_bench/frontier.csv). Needs GEMINI_API_KEY in the environment or .env.
Usage: pip install google-genai && python -m bench.frontier_baseline --n 120 [--model gemini-2.5-flash]
       python -m bench.frontier_baseline --report
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from bench.fb_bench import make_variant

SCORES = Path("data/fb_bench/scores.csv")
OUT = Path("data/fb_bench/frontier.csv")
PROMPT_A = ("You are a fraud analyst at an online marketplace. A buyer uploaded this photo as evidence of damage for a "
            "refund. Was this photo created or edited by an AI image generator (for example, damage added by AI), or "
            "is it a genuine camera photo? Answer ONLY with JSON: "
            '{"ai_probability": <0-100>, "verdict": "ai_edited" or "genuine", "reason": "<one sentence>"}')
PROMPT_B = ("You are a fraud analyst at an online marketplace. Image 1 is the buyer's damage photo for a refund; image 2 "
            "is the seller's reference photo of the item; the buyer's claim is: \"{claim}\". Was image 1 created or "
            "edited by an AI image generator, or is it a genuine camera photo? Answer ONLY with JSON: "
            '{"ai_probability": <0-100>, "verdict": "ai_edited" or "genuine", "reason": "<one sentence>"}')


def _key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("GEMINI_API_KEY not set (add it to .env)")


def _img(path, variant):
    im = Image.open(io.BytesIO(make_variant(path, variant))).convert("RGB")
    im.thumbnail((1024, 1024))
    return im


def _parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        return float(d.get("ai_probability", np.nan)), str(d.get("verdict", "")), str(d.get("reason", ""))[:200]
    except (ValueError, TypeError):
        return np.nan, "", (text or "")[:200]


def sample(n):
    df = pd.read_csv(SCORES)
    d = df[(df.pair == "same") & (df.variant == "stripped")]
    fakes = d[d.label == "fake"].sample(n // 2, random_state=0)
    dmg = d[d.label == "real_damaged"].sample(n // 4, random_state=0)
    und = d[d.label == "real_undamaged"].sample(n - len(fakes) - len(dmg), random_state=0)
    return pd.concat([fakes, dmg, und]).sample(frac=1, random_state=0)


def run(a):
    from google import genai
    client = genai.Client(api_key=_key())
    rows = sample(a.n)
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT).apply(lambda r: f"{r.row_id}|{r.prompt}", axis=1))
    new = not OUT.exists()
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["row_id", "prompt", "label", "generator", "ai_probability", "verdict", "reason", "s2_seen"])
        for i, r in enumerate(rows.itertuples(), 1):
            for p in ("A", "B"):
                if f"{r.row_id}|{p}" in done:
                    continue
                ev = _img(r.ev_path, r.variant)
                if p == "A":
                    contents = [PROMPT_A, ev]
                else:
                    ref = _img(r.ref_path, "original") if isinstance(r.ref_path, str) and r.ref_path else None
                    claim = r.claim_text if isinstance(r.claim_text, str) and r.claim_text else "Item arrived damaged"
                    contents = [PROMPT_B.replace("{claim}", claim), ev] + ([ref] if ref else [])
                for attempt in range(4):
                    try:
                        resp = client.models.generate_content(model=a.model, contents=contents)
                        prob, verdict, reason = _parse(resp.text)
                        break
                    except Exception as e:  # noqa: BLE001 -- rate limits: back off and retry
                        prob, verdict, reason = np.nan, "", f"error {type(e).__name__}"
                        time.sleep(15 * (attempt + 1))
                w.writerow([r.row_id, p, r.label, r.generator, prob, verdict, reason, r.s2_seen])
                fh.flush()
                time.sleep(a.sleep)
            print(f"  {i}/{len(rows)}", flush=True)
    report(a)


def report(a):
    f = pd.read_csv(OUT).dropna(subset=["ai_probability"])
    lines = [f"# Frontier baseline ({a.model}) vs Lenzit S2 — stripped images, same rows\n",
             "| Method | Fakes caught | False alarms on real damage | n (fake / real-damage) |", "|---|---|---|---|"]
    for p, name in (("A", "Gemini, photo only"), ("B", "Gemini, photo + reference + claim")):
        d = f[f.prompt == p]
        fake, dmg = d[d.label == "fake"], d[d.label == "real_damaged"]
        if fake.empty or dmg.empty:
            continue
        g_rec = np.mean(fake.verdict == "ai_edited")
        g_fa = np.mean(dmg.verdict == "ai_edited")
        thr = np.quantile(dmg.s2_seen, 1 - max(g_fa, 1 / len(dmg)))      # S2 at Gemini's own false-alarm rate
        s_rec = np.mean(fake.s2_seen > thr)
        s_fa = np.mean(dmg.s2_seen > thr)
        lines.append(f"| {name} | {g_rec:.0%} | {g_fa:.0%} | {len(fake)} / {len(dmg)} |")
        lines.append(f"| Lenzit S2 at the same false-alarm rate | {s_rec:.0%} | {s_fa:.0%} | {len(fake)} / {len(dmg)} |")
    Path("bench/results").mkdir(parents=True, exist_ok=True)
    Path("bench/results/frontier.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--sleep", type=float, default=4.5)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else run(a)
