# Lenzit
Verify the claim, not just the pixels. Lenzit checks refund-claim photos against the
seller's reference photo, flags AI-faked damage with a heatmap and reasons, and asks for a
live re-capture instead of accusing honest buyers.

Built for ForgeHacks Online 2026 (AI + Cybersecurity track), Oct 4-10, 2026.

## Status
Scaffold. Live demo, results table and video links will be added here.

## Layout
- `api/` FastAPI service
- `pipeline/` one module per signal + fusion
- `models/` weights manifest only (no weights in git)
- `bench/` labels, splits, evaluation
- `redteam/` attack generation scripts and logs
- `web/` Next.js frontend
- `docs/` architecture, threat model, licences

## Pre-trained models
Listed with licences in `docs/LICENSES.md`. TruFor is for non-profit use only; its weights
are downloaded by script, not redistributed.
