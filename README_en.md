# 🍅 TomatoLens

> A self-hosted **reverse image search** engine for your PikPak cloud drive.

Upload an image and instantly find the most visually similar ones among hundreds
of thousands of files in your cloud drive. Also supports keyword search by
file/folder name and smart duplicate-folder detection.

> **中文说明：** [README.md](README.md)

---

## ✨ Features

- **Reverse image search**: dual matching with CLIP semantic vectors + ORB local features, grouped by similarity (1–5 groups).
- **Video cover indexing**: automatically fetches video thumbnails so searches can also match video files.
- **Keyword search**: multi-keyword support — space means AND, `-word` means NOT, e.g. `dinner tasty -video`.
- **Duplicate folder detection**: perceptual hash (pHash) + overlap coefficient + union-find to group duplicate/subset folders. Supports both full-drive and single-folder modes.
- **Incremental sync**: idempotent design; re-running only processes the diff, and failed downloads are retried next time.
- **Web UI**: build the index, search, deduplicate, keyword-search, and log in/out — all in the browser.

---

## 🛠️ Tech Stack

- Backend: Python + Flask + asyncio + httpx
- Vector search: OpenCLIP + FAISS
- Features / dedup: OpenCV (ORB) + perceptual hash (pHash)
- Storage: SQLite (metadata) + FAISS (vectors) + local thumbnail cache
- Cloud API: pikpakapi (unofficial PikPak API wrapper)

---

## 💻 Environment

Developed and tested on:

- OS: Windows 10/11
- Python: 3.12
- GPU: NVIDIA RTX 4060 (CUDA 12.6, used to accelerate CLIP inference; runs on CPU too, just slower)
- Vector search: FAISS runs on CPU (fast enough at the ~130k scale); only CLIP uses the GPU
- Data scale: ~200k files, ~130k images, ~20 GB thumbnail cache

Notes for other environments:

- **macOS / Linux**: replace `venv\Scripts\activate` with `source venv/bin/activate`; everything else is identical. Paths in the UI use `/` consistently.
- **CPU-only machines**: fully supported — the program auto-detects and falls back to CPU. CLIP inference is noticeably slower; consider lowering `CLIP_BATCH` in `config.py`.
- **No GPU but large libraries**: indexing is a one-time cost; you can build it folder by folder during idle hours.
- **Python version**: 3.10 – 3.12 is recommended; versions that are too new or too old may lack prebuilt wheels for torch / open_clip and other dependencies.

---

## 📦 Installation

### 1. Clone

```bash
git clone https://github.com/Martin22-Glitch/TomatoLens.git
cd TomatoLens
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

> ⚠️ **About torch / torchvision / faiss (important)**
> These three libraries are tightly coupled to your CUDA setup and are **not** included
> in `requirements.txt`. Install them separately:
> - **CPU users**:
>   pick the CPU command at [pytorch.org](https://pytorch.org) to install torch + torchvision
>   together, then run `pip install faiss-cpu`.
> - **GPU users**:
>   pick the command matching your CUDA version at [pytorch.org](https://pytorch.org) to install
>   torch + torchvision together, then run `pip install faiss-gpu`.
>
> If installation is slow or fails, use a PyPI mirror, e.g.
> `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>`;
> torch can also be downloaded as a `.whl` from the official site and installed locally with
> `pip install path.whl`.

### 3. Prepare the CLIP model

Place the OpenCLIP `ViT-B-32` weights file `open_clip_pytorch_model.bin` under `models/`.
Alternatively, leave it empty and the program will download it to `hf_cache/` on first
run (a HuggingFace mirror is pre-configured).

### 4. Configure your account

```bash
copy config_local.example.py config_local.py   # Windows
# cp config_local.example.py config_local.py    # macOS / Linux
```

Edit `config_local.py` with your PikPak email and device id. No working-directory setup needed.

---

## 🚀 Usage

### Start the server

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

### Login

Click "Login" in the top-right corner and enter your PikPak password. The token is saved
and auto-refreshed on expiry. You can also log in via CLI: `python pikpak_login.py`.

### Build / sync

In the "Build / Update" panel:

- **Full sync**: walks the entire drive for reconciliation (add/update/delete).
- **Index this folder**: pick from the folder tree or type a path (use `/`); incremental only, fast.

The first full build is a one-time cost.

### Search by image

Upload an image, choose the number of groups, and click search.

### Keyword search

Type directly; supports multi-word AND and `-` exclusion, e.g. `landscape beach -thumbnail`.

### Dedup

- Empty: scan the whole drive for duplicate folder groups.
- Path (e.g. `category/food/dinner/all`): compare it against the whole drive (subset/superset both count).

---

## ⚠️ Notes

- **Privacy**: `config_local.py` and `data/token.json` hold credentials. Never commit them; change your PikPak password if leaked.
- **Disk usage**: the thumbnail cache can reach tens of GB under `data/thumbnails/`.
- **First build cost**: one-time download + CLIP/ORB/pHash computation; afterwards incremental.
- **Rate limits**: thumbnails come from PikPak servers; speed depends on network and throttling. Batch runs at night are recommended.
- **Token expiry**: on `invalid refresh token`, just log in again.
- **Compliance**: for personal learning and managing your own cloud data only. Comply with PikPak's ToS and applicable laws.

---

## 🧰 Troubleshooting

- **torch / torchvision / faiss install fails**: install them separately per the section above; use a PyPI mirror, or download the `.whl` manually and run `pip install path.whl`.
- **CLIP model download slow/fails**: a HuggingFace mirror is pre-set; or download `ViT-B-32` weights manually into `models/`.
- **`database is locked`**: WAL mode and a 30s timeout are enabled; avoid heavy concurrent writes during sync, or retry.
- **Thumbnail 404**: the thumbnail failed to download or isn't indexed yet — re-sync that folder.
- **Login captcha / risk control**: log in from your usual network; retry later if the unofficial API triggers it.

---

> A personal learning project, not affiliated with PikPak.
