"""Lenzit HTTP API (FastAPI). Runs locally with uvicorn and on Modal via serving/modal_app.py.

POST /v1/claims   multipart: evidence (file, required), reference (file, optional), claim_text (1-500 chars),
                  delivery_date (YYYY-MM-DD, optional)  -> analyse() result + claim_id; heatmap as base64 PNG
GET  /v1/claims/{claim_id}   the stored result (in memory, newest 200)
GET  /healthz     status + model version + signal modes
GET  /            web/index.html if present, else a minimal built-in page
Local:  uvicorn serving.api:app --port 8000
"""
from __future__ import annotations

import base64
import datetime as dt
import os
import secrets
import threading
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

Image.MAX_IMAGE_PIXELS = 50_000_000            # decompression-bomb guard (raises above 2x this)
MAX_BYTES = 12_000_000
MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"RIFF")   # JPEG, PNG, WebP (RIFF....WEBP)
RATE_PER_HOUR = int(os.environ.get("LENZIT_RATE_PER_HOUR", "30"))
ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="Lenzit API", version="1.0")
_lock = threading.Lock()                       # analyse() is not thread-safe: one claim at a time per container
_claims: OrderedDict[str, dict] = OrderedDict()
_hits: dict[str, deque] = defaultdict(deque)


def _validate(data: bytes, field: str) -> bytes:
    try:  # Sriman's guard when merged; built-in checks otherwise
        from security.upload_guard import UploadRejected, validate_upload
        try:
            validate_upload(data)
            return data
        except UploadRejected as e:
            raise HTTPException(getattr(e, "status_code", 422), {"error": "upload_rejected", "field": field, "message": str(e)})
    except ImportError:
        pass
    if not data:
        raise HTTPException(422, {"error": "upload_rejected", "field": field, "message": "empty file"})
    if len(data) > MAX_BYTES:
        raise HTTPException(413, {"error": "upload_rejected", "field": field, "message": "file larger than 12 MB"})
    is_heic = data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftyphevc")
    is_webp = data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    if not (data.startswith(MAGIC[:2]) or is_webp or is_heic):
        raise HTTPException(422, {"error": "upload_rejected", "field": field,
                                  "message": "only JPEG, PNG, WebP or HEIC images are accepted"})
    return data


def _rate_limit(ip: str):
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE_PER_HOUR:
        raise HTTPException(429, {"error": "rate_limited", "message": f"limit is {RATE_PER_HOUR} checks per hour"})
    q.append(now)


def _run(evidence, reference, claim, delivery):
    from pipeline.run import analyse
    with _lock:
        return analyse(evidence, reference, claim, delivery)


@app.on_event("startup")
def _warm():
    if os.environ.get("LENZIT_WARMUP", "1") == "1":
        try:
            import io
            buf = io.BytesIO()
            Image.new("RGB", (800, 600), "gray").save(buf, "JPEG")
            _run(buf.getvalue(), buf.getvalue(), "warm-up", None)   # loads CLIP + DINOv2 once
        except Exception as e:  # noqa: BLE001
            print(f"warm-up failed: {e}")


@app.get("/healthz")
def healthz():
    from pipeline.run import MODEL_VERSION
    return {"status": "ok", "model_version": MODEL_VERSION,
            **{f"{s}_mode": os.environ.get(f"LENZIT_{s.upper()}_MODE", "on" if s in ("s2", "s4") else "off")
               for s in ("s2", "s3", "s4")}}


@app.post("/v1/claims")
async def create_claim(request: Request, evidence: UploadFile = File(...), reference: UploadFile | None = File(None),
                       claim_text: str = Form(...), delivery_date: str | None = Form(None)):
    _rate_limit(request.client.host if request.client else "unknown")
    claim = claim_text.strip()
    if not 1 <= len(claim) <= 500:
        raise HTTPException(422, {"error": "bad_claim_text", "message": "claim_text must be 1-500 characters"})
    delivery = None
    if delivery_date:
        try:
            delivery = dt.date.fromisoformat(delivery_date)
        except ValueError:
            raise HTTPException(422, {"error": "bad_delivery_date", "message": "use YYYY-MM-DD"})
    ev = _validate(await evidence.read(MAX_BYTES + 1), "evidence")
    ref = _validate(await reference.read(MAX_BYTES + 1), "reference") if reference and reference.filename else None
    try:
        result = await run_in_threadpool(_run, ev, ref, claim, delivery)
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise HTTPException(422, {"error": "analysis_failed", "message": f"could not read the image ({type(e).__name__})"})
    heat = result["artifacts"].get("heatmap_png")
    result["artifacts"]["heatmap_png"] = base64.b64encode(heat).decode() if heat else None
    claim_id = secrets.token_urlsafe(12)
    result = {"claim_id": claim_id, "claim_text": claim, **result}
    _claims[claim_id] = result
    while len(_claims) > 200:
        _claims.popitem(last=False)
    return JSONResponse(result)


@app.get("/v1/claims/{claim_id}")
def get_claim(claim_id: str):
    if claim_id not in _claims:
        raise HTTPException(404, {"error": "not_found"})
    return _claims[claim_id]


@app.get("/", response_class=HTMLResponse)
def index():
    page = ROOT / "web" / "index.html"
    return page.read_text(encoding="utf-8") if page.exists() else FALLBACK


FALLBACK = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Lenzit</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;color:#1b1b1f;background:#fafafa}
label{display:block;margin:12px 0 4px;font-weight:600}input,textarea,button{font:inherit;width:100%;box-sizing:border-box}
textarea{min-height:60px}button{margin-top:16px;padding:10px;background:#1b1b1f;color:#fff;border:0;border-radius:6px;cursor:pointer}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin-top:20px}
.v{font-size:1.4em;font-weight:700}.likely_genuine{color:#1a7f37}.needs_verification{color:#9a6700}.likely_manipulated{color:#cf222e}
table{width:100%;border-collapse:collapse}td{border-top:1px solid #eee;padding:6px 4px;vertical-align:top}img{max-width:100%;border-radius:6px}
small{color:#666}</style></head><body>
<h1>Lenzit</h1><p>Verify the claim, not just the pixels. Upload the buyer's damage photo, the seller's reference photo and the claim.</p>
<form id="f"><label>Buyer's evidence photo</label><input type="file" name="evidence" accept="image/*" required>
<label>Seller's reference photo (optional)</label><input type="file" name="reference" accept="image/*">
<label>Claim text</label><textarea name="claim_text" maxlength="500" required>Item arrived damaged</textarea>
<button>Check this claim</button></form><div id="out"></div>
<p><small>Verification, not accusation: a "needs verification" result asks the buyer for a fresh photo; it never accuses anyone.
Evaluation data: FraudBench (CC BY-NC-SA 4.0).</small></p>
<script>
const f=document.getElementById('f'),out=document.getElementById('out');
const label={likely_genuine:'Likely genuine',needs_verification:'Needs verification',likely_manipulated:'Likely manipulated'};
f.onsubmit=async e=>{e.preventDefault();out.innerHTML='<div class="card">Checking… (first check after idle can take ~1 min)</div>';
const fd=new FormData(f);if(!fd.get('reference')||!fd.get('reference').size)fd.delete('reference');
try{const r=await fetch('/v1/claims',{method:'POST',body:fd});const j=await r.json();
if(!r.ok){out.innerHTML='<div class="card">Error: '+(j.detail&&j.detail.message||r.status)+'</div>';return;}
let rows='';for(const[k,s]of Object.entries(j.signals)){const na=j.unavailable_signals.includes(k);
rows+=`<tr><td>${k}</td><td>${na?'—':s.score.toFixed(2)}</td><td>${s.reason}</td></tr>`;}
out.innerHTML=`<div class="card"><div class="v ${j.verdict}">${label[j.verdict]}</div>
<p>Risk score ${j.risk_score.toFixed(2)} · next step: ${j.next_step.replaceAll('_',' ')} · ${j.latency_ms} ms</p>
${j.rules_applied.length?'<p>Rules: '+j.rules_applied.join('; ')+'</p>':''}
${j.artifacts.heatmap_png?'<p><b>Where the photo differs</b></p><img src="data:image/png;base64,'+j.artifacts.heatmap_png+'">':''}
<table>${rows}</table><p><small>${j.model_version} · claim ${j.claim_id}</small></p></div>`;}
catch(err){out.innerHTML='<div class="card">Network error: '+err+'</div>';}};
</script></body></html>"""
