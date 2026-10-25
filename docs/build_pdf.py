"""Convert docs/GUIDE_DEMARRAGE.md to docs/GUIDE_DEMARRAGE.pdf (Markdown -> HTML -> Edge/Chrome).

    pip install markdown
    python docs/build_pdf.py
Needs Microsoft Edge or Google Chrome (used headless to print the page to PDF).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).parent
SRC, OUT = HERE / "GUIDE_DEMARRAGE.md", HERE / "GUIDE_DEMARRAGE.pdf"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1c2430; }
h1 { font-size: 21pt; color: #1f3a5f; border-bottom: 2px solid #1f3a5f; padding-bottom: 5px; margin-top: 0; }
h2 { font-size: 15pt; color: #1f3a5f; margin-top: 22px; border-bottom: 1px solid #c9d3df; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 12pt; color: #1f3a5f; page-break-after: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9.2pt; background: #eef1f5; padding: 1px 4px; border-radius: 3px; }
pre { background: #f3f1ea; border: 1px solid #d9d5c7; border-radius: 4px; padding: 8px 10px; white-space: pre-wrap; word-break: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.6pt; page-break-inside: avoid; }
th { background: #dce6f1; text-align: left; }
th, td { border: 1px solid #aab4c0; padding: 4px 7px; vertical-align: top; }
blockquote { margin: 10px 0; padding: 6px 12px; background: #eef4fb; border-left: 4px solid #2f6db5; }
blockquote p { margin: 3px 0; }
hr { border: 0; border-top: 1px solid #c9d3df; margin: 18px 0; }
"""


def find_browser():
    for b in BROWSERS:
        if Path(b).exists():
            return b
    sys.exit("No Edge/Chrome found: edit BROWSERS in docs/build_pdf.py")


if __name__ == "__main__":
    html = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    page = f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Guide de démarrage</title><style>{CSS}</style></head><body>{html}</body></html>'
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "guide.html"
        f.write_text(page, encoding="utf-8")
        subprocess.run([find_browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT}", f.as_uri()], check=True, capture_output=True, timeout=120)
    print("written", OUT, OUT.stat().st_size // 1024, "KB")
