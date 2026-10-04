import io

from PIL import Image

from pipeline.run import analyse


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1600, 1200), "gray").save(buf, "JPEG")
    return buf.getvalue()


def test_analyse_contract():
    r = analyse(_jpeg(), _jpeg(), "Mug arrived with a cracked handle")
    assert r["verdict"] in {"likely_genuine", "needs_verification", "likely_manipulated"}
    assert 0.0 <= r["risk_score"] <= 1.0
    assert set(r["signals"]) == {"provenance", "global_synthetic", "local_edit", "reference", "plausibility"}
    assert "heatmap_png" in r["artifacts"]
    for key in ("unavailable_signals", "next_step", "model_version", "latency_ms"):
        assert key in r
