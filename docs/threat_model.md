# Lenzit Threat Model

Lenzit checks a buyer's "damaged item" photo before a seller issues a refund. It returns **likely_genuine**, **needs_verification** or **likely_manipulated**. It never returns "fraud". The output is a prompt for a human to verify, not an accusation.

## Adversaries

| Adversary | Goal | Capability | Effort they'll spend |
|---|---|---|---|
| **Opportunistic buyer** | One free refund | Phone + a free AI editor (ChatGPT, Gemini, Photoshop generative fill) | Minutes. Uploads whatever the tool gives them. |
| **Serial abuser** | Repeated refunds across many sellers | Learns which tricks work; screenshots, crops, re-shares to strip metadata | Hours. Will iterate against our verdicts. |
| **Organised refund-fraud service** | Refunds sold as a service to many "customers" | Custom or fine-tuned generators, many accounts and IPs, adversarial testing against detectors | Days. Treats Lenzit as a target to defeat. |

## Attack tiers

| Tier | Attack | Lenzit stops it? | How / why not |
|---|---|---|---|
| **T0** | Photo of a *different* item (old order, stock image, someone else's product) | **Partly** | S4 compares against the seller's reference photo → *needs_verification*. Does not reverse-image-search the web. |
| **T1** | Untouched AI download with Content Credentials (e.g. ChatGPT "Download") | **Yes** | S1 reads the C2PA manifest; an AI `digitalSourceType` scores 0.95 → *likely_manipulated*. Verified on a real ChatGPT image whose pixel score was 0.01. |
| **T2** | Same AI image with metadata stripped (screenshot, WhatsApp/Telegram re-share) | **Partly** | S1 sees nothing. Only the pixel detector (S2) remains: catches **66.5%** at a 5% false-alarm rate on real damage photos, **56.3%** on a generator it never saw. One in three gets through. |
| **T3** | Real photo of the item taken *before* delivery, EXIF intact | **Partly** | S1 flags capture date < delivery date → *needs_verification*. Strip the EXIF and this check is blind. |
| **T4** | Small local AI edit (inpaint a dent onto a real photo of the right item), metadata stripped | **No (often)** | Most of the image is a genuine camera photo and it matches the reference. Small edited regions frequently pass S2. This is our biggest known gap. |
| **T5** | Item damaged for real, then photographed honestly | **No** | The photo is genuine. Image forensics cannot detect intent. Out of scope. |
| **T6** | Detector-aware attacker: fine-tuned generators, adversarial noise, many accounts/IPs to evade rate limits | **No** | Our detector is a fixed public endpoint, and rate limiting is per IP only. A determined service will out-iterate a static model. |

**Design rule:** absence of metadata is never treated as evidence. Most honest photos sent through chat apps have none. Metadata can only *raise* suspicion (score 0.95 / 0.8 / 0.6); missing metadata scores a neutral 0.3. Faking a C2PA "AI-generated" claim only hurts the attacker, so trusting it in that direction is safe.

## Platform security

| Control | Detail |
|---|---|
| **Magic-byte upload check** | File type is detected from its first bytes, not the file name or the client's Content-Type. JPEG, PNG, WebP and HEIC only. Anything else → **422**. |
| **Size limit** | 12 MB per upload → **413**. |
| **Pixel limit** | 50 MP, read from the image header *before* decoding → **413**. |
| **Decompression-bomb guard** | Pillow's pixel ceiling is set to our limit while decoding; bomb warnings are treated as errors → **413**. |
| **Rate limit** | 30 checks per hour per IP → **429**. |
| **No server-side URL fetching** | Users upload bytes; the server never downloads a URL a user supplies (no SSRF path). |
| **Secrets** | API keys live only in environment variables / Modal secrets, never in the repo or the client page. |
| **Fail closed, never crash** | Any parse error in S1 returns a neutral 0.3 with a reason. A malformed file can't take the pipeline down or force a "genuine" verdict. |

## What doesn't work yet (roadmap)

- **Re-capture link (S7):** the seller sends the buyer a one-time link to take a fresh photo in-browser. This closes T2–T4 for the opportunistic tier. Not built.
- **Per-account limits and abuse history:** needed against T6; today's limit is per IP only.
- **Localised edit detection:** a region-level detector for T4 inpainting.
- **Web reverse-image search** for T0 stock and re-used photos.
