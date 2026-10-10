"""Deploy the Lenzit API on Modal (CPU; S2 + S4 + decision policy; S3 TruFor off in the cloud).

  modal deploy serving/modal_app.py      -> prints the public URL (https://<workspace>--lenzit-api-web.modal.run)
  modal serve  serving/modal_app.py      -> temporary dev URL with live reload

Model weights (CLIP ViT-L/14, DINOv2) are downloaded into the image at build time, so cold starts only load them.
Keep one container warm during judging:  LENZIT_MIN_CONTAINERS=1 modal deploy serving/modal_app.py
"""
import os

import modal

MIN_CONTAINERS = int(os.environ.get("LENZIT_MIN_CONTAINERS", "1"))


def _download_weights():
    import open_clip
    import timm
    open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("fastapi", "uvicorn", "python-multipart", "pillow", "numpy", "open_clip_torch",
                 "scikit-learn", "timm", "pillow-heif", "opencv-python-headless", "c2pa-python")
    .run_function(_download_weights)
    .env({"LENZIT_S2_MODE": "on", "LENZIT_S4_MODE": "on", "LENZIT_S3_MODE": "off"})
    .add_local_python_source("pipeline", "serving", "security")
    .add_local_dir("models", "/root/models")
    .add_local_dir("web", "/root/web")
)

app = modal.App("lenzit-api", image=image)


@app.function(cpu=4.0, memory=8192, timeout=180, min_containers=MIN_CONTAINERS, scaledown_window=900)
@modal.asgi_app()
def web():
    from serving.api import app as fastapi_app
    return fastapi_app
