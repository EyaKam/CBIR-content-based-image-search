"""Image descriptors: colour (HSV histogram), texture (LBP), shape (HOG), deep (VGG16).

Every function takes an RGB uint8 image (H x W x 3) and returns a 1-D float32 vector.
"""
import io
import threading

import cv2
import numpy as np
from PIL import Image
from skimage import feature

from . import config as C


def load_rgb(path):
    """Read an image file as RGB uint8 with PIL (None if unreadable)."""
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:
        return None


def decode_rgb(data):
    """Decode uploaded image bytes as RGB uint8 (None if they are not an image)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:
        return None


# --------------------------------------------------------------------- colour
def hsv_hist(rgb, bins=C.HSV_BINS):
    """3-D HSV histogram, L1-normalised (sums to 1) -> 8*8*8 = 512 dims.

    OpenCV hue lives in [0,180) (the course notebook used 256, wasting bins).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    h = h.flatten().astype(np.float32)
    return h / (h.sum() + 1e-10)


# -------------------------------------------------------------------- texture
def lbp_hist(rgb, points=C.LBP_POINTS, radius=C.LBP_RADIUS, method=C.LBP_METHOD):
    """Histogram of Local Binary Patterns on the grey image.

    nri_uniform with P=8 -> 59 bins (58 uniform patterns + 1 'other').
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lbp = feature.local_binary_pattern(gray, points, radius, method=method)
    n_bins = int(points * (points - 1) + 3) if method == "nri_uniform" else int(points + 2)
    h, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    h = h.astype(np.float32)
    return h / (h.sum() + 1e-10)


# ---------------------------------------------------------------------- shape
def hog_desc(rgb):
    """HOG on the resized grey image: 64x128, 8x8 cells, 9 bins, 2x2 blocks -> 3780 dims."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = C.HOG_SIZE
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    fd = feature.hog(gray, orientations=C.HOG_ORIENT, pixels_per_cell=C.HOG_CELL,
                     cells_per_block=C.HOG_BLOCK, block_norm="L2-Hys")
    fd = fd.astype(np.float32)
    return fd / (np.linalg.norm(fd) + 1e-10)


# ----------------------------------------------------------------------- deep
_vgg = None
_vgg_lock = threading.Lock()


def get_vgg():
    """Load VGG16 once (needs tensorflow). Safe to call from several threads."""
    global _vgg
    with _vgg_lock:
        if _vgg is None:
            from .feature_extractor import FeatureExtractor  # lazy: heavy import
            _vgg = FeatureExtractor()
        return _vgg


def vgg_desc(rgb):
    """VGG16 'fc1' features (4096 dims, unit L2 norm)."""
    fe = get_vgg()
    with _vgg_lock:
        return fe.extract(rgb).astype(np.float32)


def vgg_batch(images, batch_size=32):
    """VGG16 features of a list of RGB arrays -> (n, 4096) float32."""
    fe = get_vgg()
    with _vgg_lock:
        return fe.extract_batch(images, batch_size).astype(np.float32)


EXTRACTORS = {"hsv": hsv_hist, "lbp": lbp_hist, "hog": hog_desc, "vgg": vgg_desc}
LABELS = {
    "hsv": "Colour (HSV histogram)",
    "lbp": "Texture (LBP)",
    "hog": "Shape (HOG)",
    "vgg": "Deep (VGG16)",
}


def extract(rgb, names):
    """Return {name: vector} for the requested descriptors."""
    return {n: EXTRACTORS[n](rgb) for n in names}
