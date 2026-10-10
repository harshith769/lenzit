# Red-team results (Oct 10, 2026)

Rows marked Live or Local were actually run on Oct 10, 2026, and the results are exactly as observed, including misses. Rows marked **Not run** were not executed against the live deployment; for those, the source of evidence is stated instead.
- **Live** = the public deployment, https://raghavapuramharshith--lenzit-api-web.modal.run
- **Local** = the same code run with `pipeline.run.analyse` on a laptop

## Attack cases

| # | Attack (tier) | Input | Where | Result | Pass? |
|---|---|---|---|---|---|
| a | Chat-app AI edit sent as downloaded (T1) | ChatGPT-edited bottle photo, untouched download (C2PA generator "OpenAI Media Service API") | Live | **Likely manipulated**: provenance 0.95, rule "Content Credentials declare AI generation" (pixel detector S2 = 0.01) | ✅ |
| a2 | Same, second ChatGPT download | "Dent-Ready Sage Shampoo Bottle.png" | Local | **Likely manipulated** via the Content Credentials rule | ✅ |
| b | AI edit with credentials stripped (T2) | The same ChatGPT fake re-saved as pixels only (simulated screenshot) | Local | **Likely genuine**: S2 = 0.01, no credentials left | ❌ **Miss** (known limitation; shown on the live page) |
| b2 | AI fakes, metadata stripped (JPEG 75), 6 generators (T2) | 10 FraudBench fakes via the live API (`scripts/live_check.py`) | Live | 10/10 flagged (3 likely manipulated, 7 needs verification) | ✅ (see caveat below) |
| c | Wrong item vs the seller's photo (T0) | FraudBench real photo + a different product's photo | Local | **Needs verification**: "photo may not show the seller's item" | ✅ |
| c2 | Wrong item, benchmark (T0) | 60 wrong-item pairs, Lenzit-Bench v1 | Local | 83.3% flagged; 13.0% of 200 genuine same-item pairs also flagged | ✅ / trade-off |
| d | Honest buyer, real damage | Our own photo of a really dented bottle + our listing photo | Live | **Likely genuine**: S2 = 0.00 | ✅ |
| d2 | Honest buyers, live sample | 5 real-damage + 5 intact FraudBench photos via the live API | Live | 5/5 real damage passed; 3/5 intact passed (one S2 false alarm 0.74, one wrong-item flag) | ✅ / 2 false alarms |

## Platform abuse

| # | Attack | Where | Result | Pass? |
|---|---|---|---|---|
| e | Text file renamed to `.jpg` | Live | HTTP **422** upload_rejected | ✅ |
| f | 13 MB file | Live | HTTP **413** | ✅ |
| f2 | Empty claim text | Live | HTTP **422** | ✅ |
| g | Over 50 megapixels (decompression bomb) | — | **Not run** in this session. The upload guard enforces a 50 MP limit and Pillow's bomb guard; covered by the unit test `test_decompression_bomb_rejected_413` in `tests/test_s1_upload_guard.py` | ⚠️ not run live |
| h | Rate limit (more than 30 checks an hour from one IP) | — | **Not run** live. Implemented in `serving/api.py` (returns 429); not verified live | ⚠️ not run live |
| i | Security headers | Live | `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, Content-Security-Policy set on the page | ✅ |

## Caveat on b2 and d2

The deployed detector head was trained on all of FraudBench, so those photos are not unseen by it. The live run proves the system works end to end; it is **not** an accuracy measurement. Accuracy comes only from the out-of-fold benchmark: 66.5% of fakes caught at 5% false alarms on real damage with metadata stripped, and 56.3% for an unseen generator (`bench/results/fb_results.md`).

## What this tells us

- **Content Credentials are the strongest defence against chat-app edits** (case a), but only while the file is untouched. A screenshot defeats them (case b). That is the most important open gap, and the planned live re-capture link addresses it.
- **The pixel detector catches most full-image AI fakes** (b2), but not a ChatGPT edit of a real photo once its credentials are stripped (case b: S2 = 0.01).
- **Honest buyers with real damage passed in every live case** (d, d2). False alarms fall on intact photos that look different from the reference, and the response to those is a fresh-photo request, never a denial.
