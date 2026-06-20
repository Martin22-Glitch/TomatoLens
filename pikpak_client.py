import config

import json
import os
import httpx
from pikpakapi import PikPakApi

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
            ".heic", ".heif", ".tiff"}


def _save_token_dict(client):
    with open(config.TOKEN_PATH, "w", encoding="utf-8") as f:
        json.dump(client.to_dict(), f, ensure_ascii=False, indent=2)


def load_client() -> PikPakApi:
    if not os.path.exists(config.TOKEN_PATH):
        raise RuntimeError("找不到 token.json，请先登录")
    with open(config.TOKEN_PATH, "r", encoding="utf-8") as f:
        client = PikPakApi.from_dict(json.load(f))
    # 关键：token 自动刷新后写回文件，避免下次用旧 token 失效
    client.token_refresh_callback = lambda c, *a, **k: _save_token_dict(c)
    return client


def is_image(f: dict) -> bool:
    if f.get("kind", "").endswith("folder"):
        return False
    if (f.get("mime_type") or "").lower().startswith("image"):
        return True
    return os.path.splitext((f.get("name") or "").lower())[1] in IMG_EXTS


def is_indexable(f: dict) -> bool:
    """图片，或带封面缩略图的视频，都可入库。"""
    if f.get("kind", "").endswith("folder"):
        return False
    if is_image(f):
        return True
    if config.INDEX_VIDEO_COVER:
        mime = (f.get("mime_type") or "").lower()
        if mime.startswith("video") and f.get("thumbnail_link"):
            return True
    return False



async def resolve_folder_id(client, path_parts):
    """按路径逐级找到文件夹 id，并返回 (folder_id, 绝对路径前缀)。
    绝对路径前缀形如 '杂 未整理/上传文件类/飞机图/'。"""
    parent_id = None
    abs_prefix = ""
    for part in path_parts:
        found, token = None, None
        while True:
            res = await client.file_list(size=config.LIST_PAGE_SIZE,
                                         parent_id=parent_id, next_page_token=token)
            for f in res.get("files", []):
                if f.get("kind", "").endswith("folder") and f.get("name") == part:
                    found = f.get("id")
                    break
            if found:
                break
            token = res.get("next_page_token")
            if not token:
                break
        if not found:
            raise RuntimeError(f"找不到文件夹: {part}")
        parent_id = found
        abs_prefix += part + "/"
    return parent_id, abs_prefix


async def list_top_folders(client):
    """只列根目录的文件夹（用于 Web 展示网盘结构，开销很小）。"""
    folders = []
    token = None
    while True:
        res = await client.file_list(size=config.LIST_PAGE_SIZE,
                                     parent_id=None, next_page_token=token)
        for f in res.get("files", []):
            if f.get("kind", "").endswith("folder"):
                folders.append(f.get("name"))
        token = res.get("next_page_token")
        if not token:
            break
    return folders


async def walk_folders_streaming(client, folder_id=None, prefix=""):
    """流式广度优先遍历：每遍历完一个目录就 yield (绝对路径前缀, 文件列表)。"""
    queue = [(folder_id, prefix)]
    while queue:
        fid, pfx = queue.pop(0)
        token = None
        files_here, subfolders = [], []
        while True:
            res = await client.file_list(size=config.LIST_PAGE_SIZE,
                                         parent_id=fid, next_page_token=token)
            for f in res.get("files", []):
                if f.get("kind", "").endswith("folder"):
                    subfolders.append((f.get("id"), pfx + f.get("name", "") + "/"))
                else:
                    files_here.append(f)
            token = res.get("next_page_token")
            if not token:
                break
        if files_here:
            yield pfx, files_here
        queue.extend(subfolders)


async def fetch_thumb_bytes(hc, url, sem):
    async with sem:
        r = await hc.get(url, timeout=30)
        r.raise_for_status()
        return r.content


async def fetch_one_thumb_by_fileid(file_id):
    """实时取单文件缩略图，多接口回退。"""
    client = load_client()
    try:
        url = None
        # 优先 file_list 拿不到单文件，用 offline_file_info
        for getter in ("offline_file_info", "get_download_url"):
            try:
                info = await getattr(client, getter)(file_id)
                url = (info.get("thumbnail_link")
                       or (info.get("file", {}) or {}).get("thumbnail_link"))
                if url:
                    break
            except Exception:
                continue
        if not url:
            return None
        async with httpx.AsyncClient(timeout=30) as hc:
            r = await hc.get(url)
            return r.content if r.status_code == 200 else None
    finally:
        try:
            await client.httpx_client.aclose()
        except Exception:
            pass


async def list_subfolders(client, parent_id=None):
    """只列某一层的子文件夹（懒加载用），返回 [{name,id,has_child}]。"""
    folders = []
    token = None
    while True:
        res = await client.file_list(size=config.LIST_PAGE_SIZE,
                                     parent_id=parent_id, next_page_token=token)
        for f in res.get("files", []):
            if f.get("kind", "").endswith("folder"):
                folders.append({"name": f.get("name"), "id": f.get("id")})
        token = res.get("next_page_token")
        if not token:
            break
    # 判断每个子文件夹是否还有下级（只取第一页判断，省开销）
    for fo in folders:
        sub = await client.file_list(size=1, parent_id=fo["id"])
        fo["has_child"] = any(x.get("kind","").endswith("folder")
                              for x in sub.get("files", []))
    return folders
