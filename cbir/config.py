"""Central configuration. Override with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folder that contains one sub-folder per class: Objects/<class>/*.jpg
IMAGE_ROOT = Path(os.environ.get("CBIR_IMAGE_ROOT", ROOT / "data" / "Objects"))
# Where the precomputed index (features, paths, labels) is stored
INDEX_DIR = Path(os.environ.get("CBIR_INDEX_DIR", ROOT / "index"))
# Optional CSV with extra keywords: "relative/path.jpg,tag1 tag2 tag3"
TAGS_FILE = os.environ.get("CBIR_TAGS_FILE", "")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

# --- descriptor parameters -------------------------------------------------
HSV_BINS = (8, 8, 8)          # 512 dims (professor's value)
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "nri_uniform"    # P=8 -> 59 bins (skimage "uniform" gives only 10)
HOG_SIZE = (128, 64)          # (height, width) as in the slides
HOG_ORIENT = 9
HOG_CELL = (8, 8)
HOG_BLOCK = (2, 2)            # -> 3780 dims

# --- deep descriptor / search behaviour (same values as the reference project) -------------
# Minimum cosine similarity for a VGG-only image query (the reference project's `threshold`;
# Elasticsearch's min_score = threshold + 1 because it scores cosine + 1).
VGG_THRESHOLD = float(os.environ.get("CBIR_VGG_THRESHOLD", 0.2))
MAX_TOTAL_HITS = 10000        # like Elasticsearch's track_total_hits default

# default metric per descriptor (sqrt-chi2 and L2 are true metrics -> VP-tree OK)
DEFAULT_METRIC = {"hsv": "chi2", "lbp": "chi2", "hog": "l2", "vgg": "l2", "ehd": "l1", "htd": "l2"}

# --- the two image bases ----------------------------------------------------------------------
# A descriptor only exists in one base: signatures of different descriptors live in different
# spaces and cannot be compared, so the web page greys out the descriptors of the other base.
MIRFLICKR_RAW = Path(os.environ.get("CBIR_MIRFLICKR_RAW", ROOT / "data" / "mirflickr_raw"))
BASES = {
    "objects": {"title": "Objects", "index_dir": INDEX_DIR, "image_root": IMAGE_ROOT,
                "es_index": "objects"},
    "mirflickr": {"title": "MIRFLICKR 1M (thumbnails)", "index_dir": Path(os.environ.get(
                      "CBIR_INDEX_1M_DIR", ROOT / "index_1m")),
                  "image_root": MIRFLICKR_RAW, "es_index": "mirflickr"},
}
DESCRIPTORS = {
    "hsv": {"base": "objects", "title": "Colour", "note": "HSV histogram, 512 dims"},
    "lbp": {"base": "objects", "title": "Texture", "note": "Local binary patterns, 59 dims"},
    "hog": {"base": "objects", "title": "Shape", "note": "Histogram of oriented gradients, 3780 dims"},
    "vgg": {"base": "objects", "title": "Deep features", "note": "VGG16, 4096 dims"},
    "ehd": {"base": "mirflickr", "title": "Edge histogram", "note": "MPEG-7 EHD, 150 dims"},
    "htd": {"base": "mirflickr", "title": "Homogeneous texture", "note": "MPEG-7 HTD, 43 dims"},
}

# --- Elasticsearch (keywords + metadata of the images) ---------------------------------------
ES_URL = os.environ.get("CBIR_ES_URL", "http://localhost:9200")
# thumbnails of the 1M base: directory k (1..100) of the zips holds im{(k-1)*10000+1 .. k*10000}.jpg
THUMB_ZIPS = ["thumbs1-33.zip", "thumbs34-66.zip", "thumbs67-100.zip"]
