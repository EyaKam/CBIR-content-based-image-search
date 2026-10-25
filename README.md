
# 🔎 Content-Based Image & Text Search Engine with MPEG-7, VGG16, VP-Tree, Elasticsearch, Logstash, FastAPI and Docker

This project is a **content-based image retrieval (CBIR) and keyword search engine** built for the workshop *"Indexation et recherche d'images par le contenu"*. It searches **two image bases**: the **Objects** base (descriptors computed here: colour histogram, LBP texture, HOG shape, optional **VGG16**) and the **1M MIRFLICKR** base (the provided **MPEG-7** descriptors, with its 64×64 thumbnails). Signatures are compared with a similarity metric and a **VP-tree** speeds up the search. **Elasticsearch** (loaded by **Logstash**) stores the tags and metadata of every image and answers keyword queries; **FastAPI** exposes the REST service, a **web client** lets users search, and **Docker** runs the stack.

> 📘 **Nouveau sur le projet ?** Le guide pas à pas (clone, installation, données, Elasticsearch, base 1M) est dans
> [`docs/GUIDE_DEMARRAGE.md`](docs/GUIDE_DEMARRAGE.md).

---

## 📜 Table of Contents

- [📂 Project Architecture](#-project-architecture)
- [🚀 How It Works](#-how-it-works)
  - [Descriptors (signatures)](#1-descriptors-signatures)
  - [Index and VP-Tree](#2-index-and-vp-tree)
  - [Keyword Search](#3-keyword-search)
  - [FastAPI Backend](#4-fastapi-backend)
  - [Web Interface](#5-web-interface)
  - [Data: Objects base and 1M MIRFLICKR base](#6-data-objects-base-and-1m-mirflickr-base)
- [📸 Features](#-features)
- [🛠️ Technology Stack](#%EF%B8%8F-technology-stack)
- [🔧 How to Run the Project](#-how-to-run-the-project)
  - [Prerequisites](#prerequisites)
  - [Install](#1-install)
  - [Prepare the images](#2-prepare-the-images)
  - [Offline phase: build the signatures](#3-offline-phase-build-the-signatures)
  - [Load Elasticsearch with Logstash](#4-load-elasticsearch-with-logstash)
  - [Online phase: run the server](#5-online-phase-run-the-server)
  - [Access the Web Interface and API](#6-access-the-web-interface-and-api)
- [📂 Folder Structure](#-folder-structure)
- [🛡️ Environment Variables](#%EF%B8%8F-environment-variables)
- [⚙️ How the Components Work Together](#%EF%B8%8F-how-the-components-work-together)
- [🌐 REST API](#-rest-api)
- [🧪 Evaluation and Tests](#-evaluation-and-tests)
- [📊 Results](#-results)
- [💡 Design Choices](#-design-choices)

---

## 📂 **Project Architecture**

```
OFFLINE (once)
  Objects images ──► HSV / LBP / HOG (/ VGG16) ──► index/*.npy
  MIRFLICKR zips ──► MPEG-7 EHD, HTD files    ──► index_1m/*.npy
  tags (CSV files) ──► Logstash ──► Elasticsearch  (index objects, index mirflickr)

ONLINE (every query)
  Web client ──► FastAPI ──► keywords ──► Elasticsearch (BM25 / fuzzy)
                         ──► image id / upload ──► signature of the query
                                               ──► VP-tree or brute force over the .npy matrices
                         ──► ranked ids + tags (Elasticsearch) + image URLs ──► Web client
```

**Where things are stored**

| What | Where |
|---|---|
| Objects images | folder `data/Objects/` (one sub-folder per class) |
| 1M thumbnails (64×64) | read directly from `thumbs*.zip`, never extracted |
| Signatures | `index/*.npy` (Objects), `index_1m/*.npy` (MIRFLICKR) |
| Tags, labels, paths (information about each image) | Elasticsearch indices `objects` and `mirflickr` |

**A descriptor belongs to one base.** HSV, LBP, HOG and VGG16 search Objects; EHD and HTD search MIRFLICKR. Signatures of different descriptors live in different spaces and cannot be compared, so ticking a descriptor on the web page greys out the descriptors of the other base, and the API refuses a mix.

## 🚀 **How It Works**

### 1. **Descriptors (signatures)**
Every image becomes one or more vectors:
- **HSV colour histogram** – 8×8×8 bins (512 dims), L1-normalised.
- **LBP texture** – `nri_uniform`, P=8, R=1 (59 bins).
- **HOG shape** – grey image resized to 128×64, 9 orientations, 8×8 cells, 2×2 blocks (3780 dims).
- **VGG16 `fc1`** – 4096-dim deep feature, L2-normalised, compared with cosine similarity.
- **MPEG-7** (Edge Histogram, Homogeneous Texture, …) – precomputed vectors provided with the 1M base, imported as they are.

### 2. **Index and VP-Tree**
- Signatures are saved once as NumPy matrices (`index/<descriptor>.npy`).
- Distances: **L1, L2, χ²** (and cosine for VGG). Several descriptors can be combined with weights; each distance is divided by its mean pairwise distance so no descriptor dominates.
- A **VP-tree** (vantage-point tree) avoids comparing the query with every image. Only true metrics are used with the tree.

### 3. **Keyword Search (Elasticsearch + Logstash)**
- Each image is a document `{id, path, label, tags}` in an Elasticsearch index (`objects`, `mirflickr`).
- **Logstash** reads `index/objects.csv` and `index_1m/tags.csv` and loads them (`logstash/pipeline/*.conf`).
- Elasticsearch ranks with **BM25** (`match`, all/any word) and `fuzzy` (typo tolerance).
- A keyword query can be combined with an image query using **AND / OR**.
- If Elasticsearch is not running, the Objects base falls back to an in-process BM25 index.

### 4. **FastAPI Backend**
- Computes (or looks up) the query signature, searches, and returns **JSON** with the image URLs, labels and distances.
- Also answers the two requests of the reference project (`/search_by_text`, `/search_by_image/`) with the same JSON contract.

### 5. **Web Interface**
- Type keywords and/or pick an image (from the base, from a previous result, or upload one).
- Choose the descriptor(s), their weights, the distance, VP-tree or brute force, and the number of results.
- Click any result to launch a new search from it.

### 6. **Data: Objects base and 1M MIRFLICKR base**
- **Objects** (Caltech-101 style, one sub-folder per class, 9,145 images): descriptors are computed by `build_index.py`.
- **1M MIRFLICKR** (from `press.liacs.nl/mirflickr`): the MPEG-7 descriptors are provided, so nothing is computed. `import_mirflickr.py` reads the zips directly (no extraction): Edge Histogram (150 dims), Homogeneous Texture (43 dims) and the tags. Only the 64×64 **thumbnails** are used for display instead of the ten 12 GB image archives. Image `imK.jpg` goes with descriptor/tag file `K-1`.
- Only images already in the base can be queried in MIRFLICKR (no extractor for uploaded images).

---

## 📸 **Features**

- **Image Similarity Search**: query by an image of the base, a previous result, or an uploaded image.
- **Text-based Search**: keywords with BM25 ranking and fuzzy matching.
- **Combined Search**: keywords AND/OR image.
- **Choice of descriptors**: colour, texture, shape, deep features, MPEG-7, with weights.
- **Fast search**: VP-tree compared with brute force (timings shown in the UI).
- **Evaluation tools**: precision, recall, average precision, L1 vs L2 vs χ², speed benchmark.
- **Client/Server + Docker**: web client, REST API, containerised deployment.

---

## 🛠️ **Technology Stack**

| Component              | Technology                              |
|------------------------|-----------------------------------------|
| **Deep features**      | VGG16 (Keras / TensorFlow)              |
| **Classic descriptors**| OpenCV, scikit-image, NumPy             |
| **Index structure**    | VP-tree (own implementation), NumPy     |
| **Text search + metadata** | Elasticsearch 8.15 (BM25, fuzzy)    |
| **Data ingestion**     | Logstash 8.15 (CSV → Elasticsearch)     |
| **Backend API**        | FastAPI + Uvicorn                       |
| **Frontend Interface** | HTML / CSS / JavaScript                 |
| **Containerization**   | Docker, Docker Compose                  |

---

## 🔧 **How to Run the Project**

### Prerequisites
- Python 3.10 – 3.12 (TensorFlow does not support 3.13+ yet) and, for Elasticsearch / Logstash, Docker and Docker Compose.
- An image folder with one sub-folder per class (e.g. `Objects/`).

### 1. **Install**

```bash
pip install -r requirements.txt
```
The first VGG16 use downloads the ImageNet weights (~530 MB, cached in `~/.keras`).

### 2. **Prepare the images**

```
Objects/
  airplanes/  butterfly/  faces/  brain/ ...
```
No images yet? Create a small synthetic base:
```bash
python scripts/make_demo_data.py --out data/Objects
```

### 3. **Offline phase: build the signatures**

```bash
# Objects: descriptors computed here (add vgg to also compute VGG16, 530 MB weights download)
python scripts/build_index.py --root data/Objects --desc hsv lbp hog

# 1M MIRFLICKR: MPEG-7 descriptors + tags read from the zips in data/mirflickr_raw (about 5 minutes)
python scripts/import_mirflickr.py --raw data/mirflickr_raw --index index_1m
```
Download `features_edgehistogram.zip`, `features_homogeneoustexture.zip`, `tags.zip` and `thumbs*.zip` from `press.liacs.nl/mirflickr/mirdownload.html` into `data/mirflickr_raw/`.

### 4. **Load Elasticsearch with Logstash**

```bash
docker compose up -d elasticsearch
python scripts/create_es_indices.py                 # mappings of the indices objects and mirflickr
python scripts/make_objects_csv.py --index index    # index/objects.csv
docker compose run --rm logstash -f /usr/share/logstash/pipeline/objects.conf
docker compose run --rm logstash -f /usr/share/logstash/pipeline/mirflickr.conf
```
On Windows with Git Bash, prefix the `docker compose run` commands with `MSYS_NO_PATHCONV=1`.

### 5. **Online phase: run the server**

```bash
uvicorn server:app --host 0.0.0.0 --port 8000       # Elasticsearch on http://localhost:9200
```
Or everything in containers: `docker compose up -d --build cbir`.

### 6. **Access the Web Interface and API**
- **Web client:** [http://localhost:8000](http://localhost:8000)
- **Swagger docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 **Folder Structure**

```
├── server.py                 # FastAPI app: routes, image/thumbnail serving
├── requirements.txt          # Python dependencies
├── Dockerfile                # Image of the service
├── docker-compose.yml        # Elasticsearch, Logstash (ingest profile), cbir
├── logstash/pipeline/        # objects.conf, mirflickr.conf: CSV -> Elasticsearch
│
├── cbir/                     # Core library
│   ├── config.py             # Paths, descriptor parameters, thresholds (env variables)
│   ├── descriptors.py        # HSV, LBP, HOG, VGG16 descriptors
│   ├── feature_extractor.py  # VGG16 fc1 feature extractor
│   ├── distances.py          # L1, L2, chi2, cosine
│   ├── vptree.py             # VP-tree index
│   ├── textsearch.py         # In-process BM25 fallback when Elasticsearch is down
│   ├── esclient.py           # Elasticsearch client (search, mget, mapping)
│   ├── thumbs.py             # 1M thumbnails read from the zips
│   ├── indexing.py           # Offline phase: scan images, compute signatures, save index
│   └── engine.py             # Online phase: query -> distances -> ranked results
│
├── scripts/
│   ├── build_index.py        # Build the index from an image folder
│   ├── import_mirflickr.py   # 1M base: MPEG-7 descriptors + tags straight from the zips
│   ├── import_features.py    # Index from any precomputed descriptor files
│   ├── create_es_indices.py  # Elasticsearch mappings
│   ├── make_objects_csv.py   # index/objects.csv for Logstash
│   ├── evaluate.py           # Precision / recall / AP, metrics, speed
│   ├── benchmark_scale.py    # VP-tree vs brute force as N grows
│   └── make_demo_data.py     # Small synthetic image base
│
├── static/index.html         # Web client
├── tests/                    # Unit tests (text search, VP-tree, VGG plumbing, API contract)
├── data/                     # Image folders (not versioned)
└── index/                    # Generated signatures + meta.json
```

---

## 🛡️ **Environment Variables**

```bash
CBIR_IMAGE_ROOT=D:/path/to/Objects     # folder with the images
CBIR_INDEX_DIR=index                   # Objects signatures
CBIR_INDEX_1M_DIR=index_1m             # MIRFLICKR signatures
CBIR_MIRFLICKR_RAW=data/mirflickr_raw  # folder with thumbs*.zip
CBIR_ES_URL=http://localhost:9200      # Elasticsearch
CBIR_TAGS_FILE=photo_metadata.csv      # optional keywords for build_index.py
CBIR_VGG_THRESHOLD=0.2                 # minimum cosine similarity for a VGG-only query
CBIR_WARMUP=1                          # preload VGG16 at start-up (0 to disable)
```

---

## ⚙️ **How the Components Work Together**

- **Descriptors:** convert each image into numerical signatures (once, offline).
- **Index:** stores the signatures; the VP-tree organises them to prune the search.
- **FastAPI:** receives the query, computes the query signature with the same descriptor, and returns the ranked results.
- **Web client:** sends the query and displays the results; clicking one starts a new search.
- **Docker:** packages the server, the VGG16 weights cache and the index volumes.

---

## 🌐 **REST API**

`POST /api/search` (multipart): `keywords, kw_mode (and|or), combine (and|or), image_id | file, base (objects|mirflickr), descs=hsv,lbp,hog,vgg | ehd,htd, weights={"hsv":1}, metric (default|l1|l2|chi2), k, method (vptree|brute), threshold`
→ `{info, results:[{id, path, label, tags, distance}]}`. The base is deduced from `descs`; mixing descriptors of two bases returns 400.

Other routes: `GET /api/info`, `GET /api/browse?base=`, `GET /img/{base}/{id}?w=300` (image / thumbnail), `POST /search_by_text`, `POST /search_by_image/` (Objects base).

---

## 🧪 **Evaluation and Tests**

```bash
python scripts/evaluate.py --queries 200 --out results   # P@k, R@k, AP, L1 vs L2 vs chi2, speed
python scripts/benchmark_scale.py                         # VP-tree vs N and dimension
python -m pytest -q tests
```
Relevant = same class as the query (leave-one-out).

---

## 📊 **Results**

### Global Interface Description
The web page offers keyword search and image search side by side, with the descriptor choices, weights and distance options, and a result grid that shows each image with its distance.

<!-- add screenshot: images/interface.png -->

### 🔍 Search Features

#### Search by Text
- Enter keywords; all-words or any-word matching, with BM25 ranking.

<!-- add screenshot: images/text.png -->

#### Search by Image
- Pick an image of the base or upload one; results are ranked by similarity with the selected descriptors.

<!-- add screenshot: images/image.png -->

#### Evaluation
<!-- add precision/recall curve: results/precision_recall.png and the metrics tables -->

---

## 💡 **Design Choices**

- **VGG16 + cosine** captures semantics; HSV, LBP and HOG capture colour, texture and contours. They are compared in the evaluation and can be combined.
- **HSV 8×8×8**, **LBP nri_uniform (59 bins)**, **HOG 128×64 with 3780 dims**.
- **Metrics used with the VP-tree**: √χ², L1 and L2 satisfy the triangle inequality; KL divergence and histogram intersection do not.
- **Combining descriptors**: each distance is normalised by its mean pairwise distance, then weighted-summed.
- **Keywords + image**: AND = keyword matches ranked by visual similarity; OR = union of the best visual and keyword matches.
- **1M base**: precomputed MPEG-7 vectors plus thumbnails for display; queries by image id and keywords.
