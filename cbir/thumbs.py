"""Thumbnails of the 1M MIRFLICKR base, read straight from the downloaded zip files.

Extracting 1,000,000 tiny files is slow, so the server opens the zips once and reads the
requested JPEG when the web page asks for it. Image id i (0-based, the row of the index) is the
file `thumbnails/<d>/im<i+1>.jpg` with d = ceil((i+1) / 10000); d = 1..33 is in the first zip,
34..66 in the second, 67..100 in the third.

A zip that has not been downloaded is simply skipped: its images answer "not found" and the
others work (the search itself never needs the thumbnails).
"""
import threading
import zipfile
from pathlib import Path


class ThumbStore:
    def __init__(self, folder, zip_names):
        self.zips = []
        for n in zip_names:
            p = Path(folder) / n
            self.zips.append(zipfile.ZipFile(p) if p.exists() else None)
        if not any(self.zips):
            raise FileNotFoundError(f"none of {list(zip_names)} found in {folder}")
        self.lock = threading.Lock()

    @staticmethod
    def name(i):
        k = int(i) + 1
        return k, f"thumbnails/{(k - 1) // 10000 + 1}/im{k}.jpg"

    def read(self, i):
        """JPEG bytes of image i; raises KeyError if its zip is missing or lacks the file."""
        k, name = self.name(i)
        d = (k - 1) // 10000 + 1
        z = self.zips[0 if d <= 33 else 1 if d <= 66 else 2]
        if z is None:
            raise KeyError(f"thumbnail zip for {name} is not downloaded")
        with self.lock:
            return z.read(name)
