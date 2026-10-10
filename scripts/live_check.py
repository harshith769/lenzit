"""End-to-end check of the LIVE deployment on photos it has never seen (not the 5 demo cases).

Sends real test images from Lenzit-Bench v1 (FraudBench products, metadata stripped like an attacker would) to the
public API exactly as the web page does, then compares each verdict with the ground truth. Also runs 3 abuse tests.
Stays under the 30-checks-per-hour rate limit (20 photos + 3 abuse tests).
Usage (repo root): python scripts/live_check.py [--url https://...modal.run] [--n 20]
"""
import argparse
import random
import time

import httpx
import pandas as pd

from bench.fb_bench import make_variant

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="https://raghavapuramharshith--lenzit-api-web.modal.run")
ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()
U = a.url.rstrip("/")

df = pd.read_csv("data/fb_bench/scores.csv")
demo = set(open("web/demo/cases.json").read().split('"'))          # never reuse the demo images
s = df[(df.pair == "same") & (df.variant == "stripped")]
s = s[~s.ev_path.apply(lambda p: any(x and x in p for x in demo if len(x) > 8))]
k = a.n // 2
pick = pd.concat([s[s.label == "fake"].sample(k, random_state=7),
                  s[s.label == "real_damaged"].sample(a.n // 4, random_state=7),
                  s[s.label == "real_undamaged"].sample(a.n - k - a.n // 4, random_state=7)]).sample(frac=1, random_state=7)

c = httpx.Client(timeout=240)
print("health:", c.get(U + "/healthz").json())
rows = []
for i, r in enumerate(pick.itertuples(), 1):
    ev = make_variant(r.ev_path, "stripped")
    files = {"evidence": ("evidence.jpg", ev, "image/jpeg"),
             "reference": ("reference.jpg", open(r.ref_path, "rb").read(), "image/jpeg")}
    t = time.time()
    resp = c.post(U + "/v1/claims", files=files, data={"claim_text": "Item arrived damaged"})
    dt = time.time() - t
    if resp.status_code != 200:
        print(f"{i:2d}. HTTP {resp.status_code}: {resp.text[:120]}"); continue
    j = resp.json()
    flagged = j["verdict"] != "likely_genuine"
    rows.append({"truth": r.label, "generator": r.generator, "verdict": j["verdict"], "flagged": flagged,
                 "s2": j["signals"]["global_synthetic"]["score"], "sec": round(dt, 1)})
    ok = flagged == (r.label == "fake")
    print(f"{i:2d}. truth={r.label:15s} -> {j['verdict']:20s} S2={rows[-1]['s2']:.2f}  {dt:4.1f}s  {'OK' if ok else 'MISS'}")

R = pd.DataFrame(rows)
print("\n=== live results on unseen photos (metadata stripped) ===")
print(pd.crosstab(R.truth, R.verdict))
fk = R[R.truth == "fake"]
print(f"fakes flagged (manipulated or needs verification): {fk.flagged.mean():.0%} of {len(fk)}")
print(f"fakes 'likely manipulated': {(fk.verdict == 'likely_manipulated').mean():.0%}")
for lab in ("real_damaged", "real_undamaged"):
    x = R[R.truth == lab]
    print(f"{lab}: passed as likely genuine {(~x.flagged).mean():.0%} of {len(x)}")
print(f"latency: median {R.sec.median():.1f}s, max {R.sec.max():.1f}s")

print("\n=== abuse tests ===")
r1 = c.post(U + "/v1/claims", files={"evidence": ("x.jpg", b"this is not an image" * 20, "image/jpeg")}, data={"claim_text": "x"})
print("text file renamed .jpg ->", r1.status_code, "(expect 422)")
r2 = c.post(U + "/v1/claims", files={"evidence": ("big.jpg", b"\xff\xd8\xff" + bytes(13_000_000), "image/jpeg")}, data={"claim_text": "x"})
print("13 MB file ->", r2.status_code, "(expect 413)")
r3 = c.post(U + "/v1/claims", files={"evidence": ("e.jpg", make_variant(pick.iloc[0].ev_path, "stripped"), "image/jpeg")}, data={"claim_text": "   "})
print("empty claim text ->", r3.status_code, "(expect 422)")
h = c.get(U + "/").headers
print("security headers:", {k: h.get(k) for k in ("x-frame-options", "x-content-type-options", "referrer-policy")},
      "CSP set:", "content-security-policy" in h)
