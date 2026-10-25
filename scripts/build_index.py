"""Offline phase.   python scripts/build_index.py --root path/to/Objects [--desc hsv lbp hog vgg]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cbir import config as C
from cbir.indexing import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.IMAGE_ROOT), help="folder with one sub-folder per class")
    ap.add_argument("--index", default=str(C.INDEX_DIR))
    ap.add_argument("--desc", nargs="+", default=["hsv", "lbp", "hog"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tags", default=C.TAGS_FILE, help="optional keywords: lines \"relative/path.jpg,tag1 tag2\" or a Flickr metadata CSV (header id,...,tags,...)")
    a = ap.parse_args()
    build_index(a.root, a.index, tuple(a.desc), a.workers, a.tags)
