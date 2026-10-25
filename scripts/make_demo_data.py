"""Generate a small synthetic image base (10 classes) to test the whole pipeline.

    python scripts/make_demo_data.py --out data/Objects --per-class 40
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def base(rng, h, w, color, jitter=25):
    c = np.clip(np.array(color) + rng.integers(-jitter, jitter + 1, 3), 0, 255)
    return np.ones((h, w, 3), np.uint8) * c.astype(np.uint8)


def gen(cls, rng, h=150, w=200):
    if cls == "sunset":
        img = np.zeros((h, w, 3), np.float32)
        g = np.linspace(0, 1, h)[:, None]
        img[..., 0] = 255 * (0.6 + 0.4 * g)
        img[..., 1] = 140 * (1 - g) + 40
        img[..., 2] = 60 * g
        cv2.circle(img, (int(rng.integers(40, w - 40)), int(rng.integers(30, 80))), 22, (255, 230, 120), -1)
        img = img + rng.normal(0, 6, img.shape)
    elif cls == "ocean":
        y = np.arange(h)[:, None]
        wave = 20 * np.sin(y / 5.0 + rng.random() * 6)
        img = np.stack([30 + 0 * y + 0 * np.arange(w), 90 + wave + 0 * np.arange(w), 190 + wave + 0 * np.arange(w)], -1) \
            .repeat(1, 1).astype(np.float32)
        img = np.broadcast_to(img, (h, w, 3)).copy() + rng.normal(0, 8, (h, w, 3))
    elif cls == "forest":
        img = base(rng, h, w, (30, 110, 40)).astype(np.float32) + rng.normal(0, 40, (h, w, 3))
    elif cls == "desert":
        img = base(rng, h, w, (210, 170, 100)).astype(np.float32) + rng.normal(0, 12, (h, w, 3))
        for _ in range(3):
            cv2.ellipse(img, (int(rng.integers(0, w)), int(rng.integers(h // 2, h))), (90, 14), 0, 0, 180, (190, 140, 80), -1)
    elif cls == "checker":
        s = int(rng.integers(12, 20))
        yy, xx = np.indices((h, w))
        m = ((yy // s + xx // s) % 2).astype(np.float32)
        img = np.stack([m * 230 + 20] * 3, -1) + rng.normal(0, 5, (h, w, 3))
    elif cls == "stripes_v":
        s = int(rng.integers(6, 12))
        xx = np.indices((h, w))[1]
        m = ((xx // s) % 2).astype(np.float32)
        img = np.stack([m * 200 + 30, m * 60 + 40, 160 - m * 100], -1) + rng.normal(0, 5, (h, w, 3))
    elif cls == "stripes_h":
        s = int(rng.integers(6, 12))
        yy = np.indices((h, w))[0]
        m = ((yy // s) % 2).astype(np.float32)
        img = np.stack([m * 200 + 30, m * 60 + 40, 160 - m * 100], -1) + rng.normal(0, 5, (h, w, 3))
    elif cls == "circles":
        img = base(rng, h, w, (240, 240, 240), 10).astype(np.float32)
        for _ in range(int(rng.integers(5, 9))):
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(10, 30)), (200, 30, 40), -1)
    elif cls == "triangles":
        img = base(rng, h, w, (20, 20, 40), 10).astype(np.float32)
        for _ in range(int(rng.integers(4, 8))):
            x, y = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
            pts = np.array([[x, y - 18], [x - 16, y + 12], [x + 16, y + 12]])
            cv2.fillPoly(img, [pts], (240, 220, 30))
    elif cls == "snow":
        img = base(rng, h, w, (235, 240, 250), 8).astype(np.float32) + rng.normal(0, 4, (h, w, 3))
        for _ in range(60):
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), 2, (255, 255, 255), -1)
    else:
        raise ValueError(cls)
    return np.clip(img, 0, 255).astype(np.uint8)


CLASSES = ["sunset", "ocean", "forest", "desert", "checker", "stripes_v",
           "stripes_h", "circles", "triangles", "snow"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/Objects")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for cls in CLASSES:
        d = Path(a.out) / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.per_class):
            cv2.imwrite(str(d / f"{cls}_{i:03d}.jpg"), cv2.cvtColor(gen(cls, rng), cv2.COLOR_RGB2BGR))
    print(f"wrote {len(CLASSES) * a.per_class} images to {a.out}")
