"""M0 smoke test: load CLIP ViT-L/14 and time feature extraction on one image (CPU and GPU)."""
import sys, time
import torch, open_clip
from PIL import Image

path = sys.argv[1]
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
model.eval()
x = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)

devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devices:
    m, xi = model.to(dev), x.to(dev)
    with torch.no_grad():
        m.encode_image(xi)  # warm-up
        if dev == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        feats = m.encode_image(xi)
        if dev == "cuda": torch.cuda.synchronize()
    print(f"{dev}: feature shape {tuple(feats.shape)}, {1000*(time.perf_counter()-t):.0f} ms per image")
print("CUDA available:", torch.cuda.is_available())
