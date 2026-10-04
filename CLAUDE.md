# Lenzit (ForgeHacks 2026) - rules for every Claude Code session

Lenzit checks whether a refund-claim photo is genuine evidence, using the seller's reference photo.
Full plan: the Lenzit Source of Truth doc; per-person work briefs for Sriman and Hotrish.

## Ownership (edit ONLY your own paths; ask the owner for anything else)
- Harshith (ML, integration owner): pipeline/base.py, pipeline/run.py, pipeline/stubs.py,
  pipeline/canonical.py, pipeline/clip_features.py, pipeline/s2_global.py, pipeline/s4_reference.py,
  pipeline/s5_plausibility.py, pipeline/fusion.py, bench/*.py, docs/results.md, requirements/ml.txt, tests/ml/
- Sriman (Security & Data): pipeline/s1_provenance.py, pipeline/s7_recapture.py, security/, redteam/,
  bench/labels.csv, docs/threat_model.md, docs/security_checklist.md, docs/redteam_results.md,
  docs/LICENSES.md, requirements/security.txt, tests/security/
- Hotrish (Product): api/, web/, serving/, demo/, pipeline/s3_local.py, scripts/deploy_*,
  docs/api_contract.md, README.md, requirements/api.txt, tests/api/
- Shared (PR approved by Harshith): CLAUDE.md, .gitignore, requirements.txt, .env.example, pytest.ini,
  pipeline/__init__.py

## Contracts (frozen; never rename)
- Every signal returns pipeline.base.SignalResult(name, score 0=genuine..1=manipulated, reason, artifacts, latency_ms).
- Signal names: provenance, global_synthetic, local_edit, reference, plausibility, recapture.
- The API gets every verdict from pipeline.run.analyse(evidence, reference, claim, delivery_date) -> dict.
- Exact signatures and artifact keys: see the work briefs, section 3.

## Rules
- Never commit images, data/, model weights (TruFor is non-profit licensed), .env, keys or tokens.
- Branches: ml/<topic>, sec/<topic>, web/<topic>. One topic per PR, tests pass (pytest -q), squash merge.
- Pull with `git pull --rebase origin main` before starting work and before opening a PR.
- UI and docs never say "fraud"; use likely genuine / needs verification / likely manipulated.
- Report results exactly as measured.
