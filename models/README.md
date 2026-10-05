# models/

Only small heads trained by us live here. Large third-party weights (CLIP, DINOv2, TruFor) are
downloaded at setup or build time and are never committed.

| File | What | Trained on | Licence |
|---|---|---|---|
| `s2_head_v1.npz` | S2 linear head (768 weights + scaler) on frozen CLIP ViT-L/14 features | FraudBench, all 6 generators, fakes as generated + compression-matched | Inherits CC BY-NC-SA 4.0 from FraudBench: non-commercial; retrain without FraudBench before any paid use |

Reported S2 numbers come from `bench/evaluate.py` (leave-one-generator-out), not from these files.
