import logging
logging.getLogger("werkzeug").setLevel(logging.WARNING)
import config
import core
from pikpak_client import fetch_one_thumb_by_fileid, load_client, list_subfolders
import sync as sync_mod
import dedup

import io, os, sqlite3, time, asyncio, threading
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file, render_template

core.get_model(); core.get_orb()
STATE = {"index": core.load_faiss()}
print(f"[app] 索引已加载，向量数: {STATE['index'].ntotal}")

SYNC = {"running": False, "progress": {}, "last": None, "cancel": False}


def _db():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def do_search(img, groups=2):
    groups = max(1, min(groups, 5))
    topk = groups * 5
    t0 = time.time()
    conn = _db()
    vec = core.embed_image(img)
    scores, ids = STATE["index"].search(vec, max(config.RECALL_K, topk))
    cands = []
    for s, fid in zip(scores[0], ids[0]):
        if fid == -1: continue
        fid = int(fid)
        row = conn.execute("SELECT name, path, file_id FROM images WHERE faiss_id=?",
                           (fid,)).fetchone()
        cands.append({"fid": fid, "fid_str": str(fid), "clip": float(s),
                      "path": row[1] if row else "?"})
    q_pts, q_des = core.extract_orb_pil(img)
    for c in cands:
        tp, td = core.load_orb(c["fid"])
        inl, rat = core.orb_score(q_pts, q_des, tp, td)
        c["orb"], c["ratio"] = inl, rat
    cands.sort(key=lambda x: (x["orb"], x["ratio"]), reverse=True)
    second = cands[1]["orb"] if len(cands) > 1 else 0
    for idx, c in enumerate(cands):
        same = (c["orb"] >= config.MIN_INLIERS and c["ratio"] >= config.MIN_INLIER_RATIO
                and (idx > 0 or c["orb"] >= max(config.MIN_INLIERS, second*config.DOMINANCE_RATIO)))
        c["verdict"] = ("★同图/局部" if same else
                        "疑似" if c["orb"] >= config.MIN_INLIERS and c["ratio"] >= config.MIN_INLIER_RATIO
                        else "相似")
        c.pop("fid", None)
    conn.close()
    return cands[:topk], time.time() - t0


def sync_worker(path_list):
    SYNC["running"] = True; SYNC["cancel"] = False
    SYNC["progress"] = {"stage": "starting"}
    def cb(p):
        SYNC["progress"] = p
        if SYNC["cancel"]:
            raise KeyboardInterrupt("用户取消")
    try:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        res = loop.run_until_complete(sync_mod.run_sync(path_list, cb))
        SYNC["last"] = res
        STATE["index"] = core.load_faiss()
    except KeyboardInterrupt:
        SYNC["progress"] = {"stage": "cancelled", "msg": "已暂停/结束（已处理部分已保存）"}
        SYNC["last"] = {**SYNC.get("progress", {}), "total": STATE["index"].ntotal,
                        "cancelled": True}
        STATE["index"] = core.load_faiss()
    except Exception as e:
        SYNC["progress"] = {"stage": "error", "msg": str(e)}
        SYNC["last"] = {"error": str(e), "total": STATE["index"].ntotal}
    finally:
        SYNC["running"] = False


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/stat")
def api_stat():
    return jsonify({"total": STATE["index"].ntotal})

@app.route("/api/subfolders")
def api_subfolders():
    pid = request.args.get("parent_id") or None
    try:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        client = load_client()
        folders = loop.run_until_complete(list_subfolders(client, pid))
        loop.run_until_complete(client.httpx_client.aclose())
        return jsonify({"folders": folders})
    except Exception as e:
        return jsonify({"folders": [], "error": str(e)})

@app.route("/api/search", methods=["POST"])
def api_search():
    groups = int(request.form.get("groups", config.RESULT_GROUPS_DEFAULT))
    f = request.files["image"]
    img = Image.open(io.BytesIO(f.read()))
    results, elapsed = do_search(img, groups)
    return jsonify({"results": results, "elapsed": elapsed})

@app.route("/api/sync", methods=["POST"])
def api_sync():
    if SYNC["running"]:
        return jsonify({"error": "已有同步任务在运行"})
    data = request.get_json(force=True)
    path = (data.get("path") or "").strip()
    # 路径统一用 / 分隔
    path = path.replace("\\", "/")
    path_list = [p.strip() for p in path.split("/") if p.strip()] if path else None
    threading.Thread(target=sync_worker, args=(path_list,), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/sync_status")
def api_sync_status():
    return jsonify({"running": SYNC["running"], **SYNC["progress"], "last": SYNC["last"]})

@app.route("/api/sync_cancel", methods=["POST"])
def api_sync_cancel():
    if SYNC["running"]:
        SYNC["cancel"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "msg": "当前无运行任务"})

@app.route("/api/reload", methods=["POST"])
def api_reload():
    STATE["index"] = core.load_faiss()
    return jsonify({"total": STATE["index"].ntotal})

@app.route("/thumb/<fid>")
def thumb(fid):
    try:
        fid = int(fid)
    except ValueError:
        return "", 404
    local = core.thumb_path(fid)
    if os.path.exists(local):
        return send_file(local, mimetype="image/jpeg")
    conn = _db()
    row = conn.execute("SELECT file_id FROM images WHERE faiss_id=?", (fid,)).fetchone()
    conn.close()
    if not row: return "", 404
    try:
        data = asyncio.run(fetch_one_thumb_by_fileid(row[0]))
    except Exception:
        data = None
    if not data: return "", 404
    core.save_thumb(fid, data)
    return send_file(io.BytesIO(data), mimetype="image/jpeg")

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    pwd = data.get("password", "")
    if not pwd:
        return jsonify({"ok": False, "msg": "请输入密码"})
    try:
        from pikpakapi import PikPakApi
        import json
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        client = PikPakApi(username=config.PIKPAK_USERNAME,
                           password=pwd, device_id=config.DEVICE_ID)
        loop.run_until_complete(client.login())
        loop.run_until_complete(client.get_quota_info())
        with open(config.TOKEN_PATH, "w", encoding="utf-8") as f:
            json.dump(client.to_dict(), f, ensure_ascii=False, indent=2)
        loop.run_until_complete(client.httpx_client.aclose())
        return jsonify({"ok": True, "msg": "登录成功，token 已保存"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    try:
        if os.path.exists(config.TOKEN_PATH):
            os.remove(config.TOKEN_PATH)
        return jsonify({"ok": True, "msg": "已登出，token 已删除"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/api/login_status")
def api_login_status():
    return jsonify({"logged_in": os.path.exists(config.TOKEN_PATH)})


def _parse_keywords(q):
    """解析多关键词：空格分隔=AND；前缀 - =排除。返回 (includes, excludes)。"""
    inc, exc = [], []
    for tok in q.split():
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("-") and len(tok) > 1:
            exc.append(tok[1:].lower())
        else:
            inc.append(tok.lower())
    return inc, exc


def _match(text, inc, exc):
    t = (text or "").lower()
    if any(e in t for e in exc):
        return False
    return all(i in t for i in inc)


@app.route("/api/keyword")
def api_keyword():
    """多关键词：空格=同时满足(AND)，-词=排除(NOT)。文件夹优先，文件名分页。"""
    q = (request.args.get("q") or "").strip()
    page = int(request.args.get("page", 1))
    inc, exc = _parse_keywords(q)
    if not inc and not exc:
        return jsonify({"folders": [], "files": [], "total": 0, "page": 1, "pages": 1})

    conn = _db()
    rows = conn.execute("SELECT faiss_id, name, path FROM images").fetchall()
    conn.close()

    # 文件夹：对每条 path 的每一级目录段做匹配
    folder_count = {}
    for fid, name, path in rows:
        if not path:
            continue
        segs = path.split("/")[:-1]
        cur = ""
        for s in segs:
            cur += s + "/"
            if _match(s, inc, exc):
                folder_count[cur] = folder_count.get(cur, 0) + 1
    folders = [{"path": k, "count": v}
               for k, v in sorted(folder_count.items(), key=lambda x: -x[1])]

    # 文件名匹配
    files_all = [{"fid_str": str(fid), "name": name, "path": path}
                 for fid, name, path in rows if _match(name, inc, exc)]
    total = len(files_all)
    psize = config.KEYWORD_PAGE_SIZE
    if total > config.KEYWORD_PAGINATE_OVER:
        pages = (total + psize - 1) // psize
        page = max(1, min(page, pages))
        files = files_all[(page - 1) * psize: page * psize]
    else:
        pages, page, files = 1, 1, files_all
    return jsonify({"folders": folders, "files": files,
                    "total": total, "page": page, "pages": pages})


@app.route("/api/dedup", methods=["POST"])
def api_dedup():
    data = request.get_json(force=True)
    scope = (data.get("scope") or "").strip()
    try:
        if not scope:
            groups = dedup.find_duplicate_folders()
            return jsonify({"ok": True, "mode": "full",
                            "groups": groups, "count": len(groups)})
        target = scope.replace("\\", "/").strip("/") + "/"
        res = dedup.find_duplicates_of(target)
        if res.get("error") == "not_exist":
            return jsonify({"ok": False, "msg": res["msg"]})
        return jsonify({"ok": True, "mode": "single", **res})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


if __name__ == "__main__":
    print("✅ 打开 http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)
