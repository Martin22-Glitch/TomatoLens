import config
import core
from pikpak_client import (load_client, walk_folders_streaming,
                           resolve_folder_id, fetch_thumb_bytes,
                           is_image, is_indexable)

import asyncio
import io
import sqlite3
import time
import numpy as np
from PIL import Image
import httpx


def init_db():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            faiss_id  INTEGER PRIMARY KEY,
            file_id   TEXT UNIQUE,
            name      TEXT,
            path      TEXT,
            modified  TEXT,
            size      INTEGER,
            phash     INTEGER
        )
    """)
    try:
        conn.execute("ALTER TABLE images ADD COLUMN phash INTEGER")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn


async def _process_batch(hc, sem, batch, conn, index, counters):
    async def dl(f):
        url = f.get("thumbnail_link")
        if not url:
            return f, None
        for attempt in range(3):
            try:
                return f, await fetch_thumb_bytes(hc, url, sem)
            except Exception:
                await asyncio.sleep(0.8 * (attempt + 1))
        return f, None

    results = await asyncio.gather(*[dl(f) for f in batch])
    imgs, metas, datas = [], [], []
    for f, data in results:
        if data is None:
            counters["failed"] += 1
            continue
        try:
            img = Image.open(io.BytesIO(data)); img.load()
            imgs.append(img); metas.append(f); datas.append(data)
        except Exception:
            counters["failed"] += 1
    if not imgs:
        return
    vecs = core.embed_images_batch(imgs)
    for img, f, data, vec in zip(imgs, metas, datas, vecs):
        fid = core.fid_to_int(f.get("id"))
        index.add_with_ids(vec.reshape(1, -1), np.array([fid], dtype="int64"))
        pts, des = core.extract_orb_pil(img)
        if des is not None:
            core.save_orb(fid, pts, des)
        core.save_thumb(fid, data)
        try:
            ph = core.phash(img)
        except Exception:
            ph = None
        conn.execute("INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?)",
                     (fid, f.get("id"), f.get("name"), f.get("_path"),
                      f.get("modified_time"), f.get("size"),
                      str(ph) if ph is not None else None))
        counters["added"] += 1



async def run_sync(target_path=None, progress_cb=None):
    """
    target_path: None=全盘；或路径列表如 ["杂 未整理","上传文件类","飞机图"]=指定文件夹。
    progress_cb: 可选回调 progress_cb(dict)，用于 Web 实时进度。
    """
    # ✅ 关键修复：counters / seen_fids 必须在任何 report() 调用之前定义
    counters = {"added": 0, "failed": 0, "seen": 0, "updated": 0, "deleted": 0}
    seen_fids = set()
    sem = asyncio.Semaphore(config.DOWNLOAD_CONCURRENCY)   # ← 在当前循环里建

    def report(stage, **kw):
        if progress_cb:
            progress_cb({"stage": stage, **kw, **counters})

    t_start = time.time()
    client = load_client()
    conn = init_db()
    index = core.load_faiss()

    # 起点 + 绝对路径前缀
    if target_path:
        report("resolving", msg=f"定位文件夹 {' / '.join(target_path)}")
        root_id, abs_prefix = await resolve_folder_id(client, target_path)
        scope_full = False
    else:
        root_id, abs_prefix = None, ""
        scope_full = True

    # 已知 file_id -> (name, path)
    known = {}
    for fid_str, name, path in conn.execute("SELECT file_id, name, path FROM images"):
        known[fid_str] = (name, path)

    pending = []
    report("walking", msg="开始遍历网盘")

    async with httpx.AsyncClient(
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=32),
            timeout=30) as hc:
        async for folder_path, files in walk_folders_streaming(client, root_id, abs_prefix):
            for f in files:
                if not is_indexable(f):
                    continue
                fid_str = f.get("id")
                new_path = folder_path + f.get("name", "")
                seen_fids.add(fid_str)
                counters["seen"] += 1
                if fid_str not in known:
                    f["_path"] = new_path
                    pending.append(f)
                    known[fid_str] = (f.get("name"), new_path)
                else:
                    old_name, old_path = known[fid_str]
                    if old_name != f.get("name") or old_path != new_path:
                        conn.execute(
                            "UPDATE images SET name=?, path=?, modified=? WHERE file_id=?",
                            (f.get("name"), new_path, f.get("modified_time"), fid_str))
                        known[fid_str] = (f.get("name"), new_path)
                        counters["updated"] += 1

            while len(pending) >= config.CLIP_BATCH:
                batch, pending = pending[:config.CLIP_BATCH], pending[config.CLIP_BATCH:]
                await _process_batch(hc, sem, batch, conn, index, counters)
                conn.commit(); core.save_faiss(index)
                report("indexing", msg=f"正在处理: {folder_path[:40]}")
                if config.REQUEST_DELAY > 0:
                    await asyncio.sleep(config.REQUEST_DELAY)

        if pending:
            await _process_batch(hc, sem, pending, conn, index, counters)
            conn.commit(); core.save_faiss(index)
            report("indexing", msg="处理收尾批次")

    # 删除检测：仅全盘
    if scope_full:
        db_fids = {r[0] for r in conn.execute("SELECT file_id FROM images")}
        to_delete = db_fids - seen_fids
        if to_delete:
            report("deleting", msg=f"清理已删除的 {len(to_delete)} 张")
            for fid_str in to_delete:
                fid = core.fid_to_int(fid_str)
                try:
                    index.remove_ids(np.array([fid], dtype="int64"))
                except Exception:
                    pass
                core.delete_orb(fid); core.delete_thumb(fid)
                conn.execute("DELETE FROM images WHERE file_id=?", (fid_str,))
                counters["deleted"] += 1
            conn.commit(); core.save_faiss(index)

    conn.commit(); core.save_faiss(index)
    elapsed = time.time() - t_start
    conn.close()
    await client.httpx_client.aclose()
    result = {"stage": "done", "elapsed": round(elapsed, 1),
              "total": index.ntotal, **counters}
    report("done", elapsed=round(elapsed, 1), total=index.ntotal)
    return result


if __name__ == "__main__":
    # 命令行：默认全盘；传文件夹路径用逗号分隔
    import sys
    tp = None
    if len(sys.argv) > 1:
        tp = [p.strip() for p in sys.argv[1].split(",") if p.strip()]
    def cli_progress(p):
        print(f"  [{p['stage']}] 扫描{p.get('seen',0)} 新增{p.get('added',0)} "
              f"更新{p.get('updated',0)} 失败{p.get('failed',0)} {p.get('msg','')}")
    res = asyncio.run(run_sync(tp, cli_progress))
    print(f"\n✅ 完成: {res}")
