import io

from fastapi.testclient import TestClient
from PIL import Image


def _jpeg(color="gray"):
    buf = io.BytesIO()
    Image.new("RGB", (900, 700), color).save(buf, "JPEG")
    return buf.getvalue()


def _client(monkeypatch):
    for k, v in {"LENZIT_S2_MODE": "off", "LENZIT_S4_MODE": "off", "LENZIT_WARMUP": "0"}.items():
        monkeypatch.setenv(k, v)
    from serving.api import app
    return TestClient(app)


def test_claim_roundtrip(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/v1/claims", files={"evidence": ("e.jpg", _jpeg(), "image/jpeg"),
                                    "reference": ("r.jpg", _jpeg("white"), "image/jpeg")},
               data={"claim_text": "Mug arrived cracked", "delivery_date": "2026-10-01"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["verdict"] in {"likely_genuine", "needs_verification", "likely_manipulated"}
    assert "global_synthetic" in j["unavailable_signals"]
    assert c.get(f"/v1/claims/{j['claim_id']}").status_code == 200
    assert c.get("/healthz").json()["status"] == "ok"
    assert c.get("/").status_code == 200


def test_rejects_non_image(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/v1/claims", files={"evidence": ("x.txt", b"hello world, not an image", "text/plain")},
               data={"claim_text": "broken"})
    assert r.status_code == 422


def test_rejects_bad_claim(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/v1/claims", files={"evidence": ("e.jpg", _jpeg(), "image/jpeg")}, data={"claim_text": "  "})
    assert r.status_code == 422
