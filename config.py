import os

# ---------------- 项目根目录 ----------------
# 可被 config.local.py 覆盖
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------- PikPak 账号（敏感，放 config.local.py）----------------
PIKPAK_USERNAME = "your_email@example.com"
DEVICE_ID = "changeme0000fixeddeviceid0000000"  # 任意固定32位字符串

# ---------------- 模型 ----------------
MODEL_NAME = "ViT-B-32"
CLIP_DIM = 512

# ---------------- 索引范围 ----------------
INDEX_ROOT_PATH = None  # None=全盘；或 ["Folder","SubFolder"]

# ---------------- 性能 / 限速 ----------------
DOWNLOAD_CONCURRENCY = 32
CLIP_BATCH = 32
REQUEST_DELAY = 0.0
LIST_PAGE_SIZE = 200

# ---------------- 检索参数 ----------------
RECALL_K = 50
ORB_FEATURES = 800
MIN_INLIERS = 12
MIN_INLIER_RATIO = 0.45
DOMINANCE_RATIO = 3.0
LOWE_RATIO = 0.70
RESULT_GROUPS_DEFAULT = 2

# ---------------- 关键词搜索 ----------------
KEYWORD_PAGE_SIZE = 50        # 每页条数
KEYWORD_PAGINATE_OVER = 50    # 超过这个数才分页

# ---------------- 去重 ----------------
OVERLAP_RATIO = 0.8           # 重叠系数阈值
MIN_OVERLAP_COUNT = 5         # 绝对交集下限

# ---------------- 是否把视频封面也入库 ----------------
INDEX_VIDEO_COVER = True

# ---------------- 本地覆盖（敏感信息）----------------
try:
    from config_local import *   # noqa
except ImportError:
    pass

# ---------------- 依赖路径（基于最终 BASE_DIR）----------------
os.makedirs(DATA_DIR, exist_ok=True)
LOCAL_MODEL = os.path.join(BASE_DIR, "models", "open_clip_pytorch_model.bin")
TOKEN_PATH = os.path.join(DATA_DIR, "token.json")
DB_PATH = os.path.join(DATA_DIR, "index.db")
FAISS_PATH = os.path.join(DATA_DIR, "index.faiss")
ORB_CACHE_DIR = os.path.join(DATA_DIR, "orb_cache")
THUMB_DIR = os.path.join(DATA_DIR, "thumbnails")
os.makedirs(ORB_CACHE_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(BASE_DIR, "hf_cache"))
