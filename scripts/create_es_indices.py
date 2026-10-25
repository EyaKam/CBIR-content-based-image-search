"""Create the Elasticsearch indices `objects` and `mirflickr` with their mapping (run once,
before the Logstash ingestion).   python scripts/create_es_indices.py [--recreate]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cbir import config as C
from cbir.esclient import ES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="delete the indices first")
    ap.add_argument("--es", default=C.ES_URL)
    a = ap.parse_args()
    es = ES(a.es)
    if not es.alive():
        raise SystemExit(f"Elasticsearch not reachable at {a.es}: docker compose up -d elasticsearch")
    for b in C.BASES.values():
        es.create_index(b["es_index"], recreate=a.recreate)
        print(f"index {b['es_index']} ready")
