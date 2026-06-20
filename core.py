import config

import os
import io
import hashlib
import pickle
import shutil
import tempfile
import numpy as np
from PIL import Image
import torch
import open_clip
import faiss
import cv2

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 模型（模块加载时初始化一次）----------------
_model = None
_preprocess = None
_orb = None
_bf = None


def get_model():
    global _model, _preprocess
    if _model is None:
        print(f"[core] 加载 CLIP 模型到 {DEVICE} ...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            config.MODEL_NAME, pretrained=config.LOCAL_MODEL
        )
        _model = _model.to(DEVICE).eval()
        print(f"[core] HEIC 支持: {'是' if HEIC_OK else '否'}")
    return _model, _preprocess


def get_orb():
    global _orb, _bf
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=config.ORB_FEATURES)
        _bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    return _orb, _bf


# ---------------- ID ----------------
def fid_to_int(file_id: str) -> int:
    """PikPak file_id 字符串 -> int64 主键。"""
    return int(hashlib.sha256(file_id.encode()).hexdigest()[:15], 16)


# ---------------- 特征提取 ----------------
@torch.no_grad()
def embed_images_batch(imgs):
    """批量 CLIP 推理。imgs: list[PIL.Image] -> np.ndarray (N, dim)"""
    model, preprocess = get_model()
    batch = torch.stack([preprocess(im.convert("RGB")) for im in imgs]).to(DEVICE)
    feats = model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


@torch.no_grad()
def embed_image(img):
    """单张 CLIP 推理 -> (1, dim)"""
    return embed_images_batch([img])


def extract_orb_pil(img, max_side=1024):
    orb, _ = get_orb()
    arr = np.array(img.convert("L"))
    h, w = arr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)))
    kp, des = orb.detectAndCompute(arr, None)
    if des is None or len(kp) == 0:
        return None, None
    return np.float32([k.pt for k in kp]), des


def orb_score(q_pts, q_des, t_pts, t_des):
    _, bf = get_orb()
    if q_des is None or t_des is None or len(q_des) < 2 or len(t_des) < 2:
        return 0, 0.0
    matches = bf.knnMatch(q_des, t_des, k=2)
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < config.LOWE_RATIO * n.distance:
                good.append(m)
    if len(good) < 4:
        return len(good), 0.0
    src = np.float32([q_pts[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([t_pts[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0, 0.0
    inl = int(mask.sum())
    return inl, inl / len(good)


# ---------------- ORB 缓存（按文件存）----------------
def orb_cache_path(fid):
    return os.path.join(config.ORB_CACHE_DIR, f"{fid}.pkl")


def save_orb(fid, pts, des):
    with open(orb_cache_path(fid), "wb") as f:
        pickle.dump((pts, des), f)


def load_orb(fid):
    p = orb_cache_path(fid)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None, None


def delete_orb(fid):
    p = orb_cache_path(fid)
    if os.path.exists(p):
        os.remove(p)


# ---------------- 缩略图本地缓存 ----------------
def thumb_path(fid):
    return os.path.join(config.THUMB_DIR, f"{fid}.jpg")


def save_thumb(fid, data):
    with open(thumb_path(fid), "wb") as f:
        f.write(data)


def delete_thumb(fid):
    p = thumb_path(fid)
    if os.path.exists(p):
        os.remove(p)


# ---------------- Faiss 读写（绕开中文路径）----------------
def load_faiss():
    if os.path.exists(config.FAISS_PATH):
        tmp = os.path.join(tempfile.gettempdir(), "imgsearch_load.faiss")
        shutil.copyfile(config.FAISS_PATH, tmp)
        idx = faiss.read_index(tmp)
        os.remove(tmp)
        return idx
    return faiss.IndexIDMap(faiss.IndexFlatIP(config.CLIP_DIM))


def save_faiss(index):
    tmp = os.path.join(tempfile.gettempdir(), "imgsearch_save.faiss")
    faiss.write_index(index, tmp)
    shutil.move(tmp, config.FAISS_PATH)

# ---------------- 感知哈希 pHash（用于去重）----------------
def phash(img, hash_size=8):
    """返回 64 位整数的感知哈希。img: PIL.Image"""
    import numpy as _np
    # 缩到 32x32 灰度，做 DCT 取低频
    im = img.convert("L").resize((32, 32))
    arr = _np.asarray(im, dtype=_np.float32)
    dct = _dct2(arr)
    low = dct[:hash_size, :hash_size]
    med = _np.median(low[1:].flatten())  # 排除 DC 分量
    bits = (low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _dct2(a):
    import numpy as _np
    N = a.shape[0]
    # 一维 DCT 矩阵
    k = _np.arange(N)
    M = _np.cos(_np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * N))
    M[0, :] *= 1 / _np.sqrt(2)
    M *= _np.sqrt(2 / N)
    return M @ a @ M.T


def phash_hamming(a, b):
    return bin(a ^ b).count("1")
