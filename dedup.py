import config
import sqlite3
from collections import defaultdict


class _UF:
    def __init__(self):
        self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _folder_of(path):
    if "/" in path:
        return path.rsplit("/", 1)[0] + "/"
    return "(根目录)/"


def _load_folder_hashes():
    """读全盘，返回 {文件夹路径: set(phash)}。"""
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    rows = conn.execute(
        "SELECT path, phash FROM images "
        "WHERE phash IS NOT NULL AND phash != '0' AND phash != 0").fetchall()
    conn.close()
    folder_hashes = defaultdict(set)
    for path, ph in rows:
        try:
            folder_hashes[_folder_of(path)].add(int(ph))
        except (ValueError, TypeError):
            continue
    return folder_hashes


def _is_dup(inter, sa, sb):
    """重叠系数判定：交集 / 较小集 >= 阈值，且绝对交集达标。"""
    if sa == 0 or sb == 0:
        return False, 0.0
    overlap = inter / min(sa, sb)
    return (inter >= config.MIN_OVERLAP_COUNT
            and overlap >= config.OVERLAP_RATIO), overlap


def find_duplicate_folders():
    """全盘模式：所有文件夹两两查重，返回重复组列表。"""
    folder_hashes = _load_folder_hashes()
    folders = list(folder_hashes.keys())

    # 倒排索引，只比有共同 phash 的文件夹对
    inv = defaultdict(list)
    for f in folders:
        for h in folder_hashes[f]:
            inv[h].append(f)
    pair_overlap = defaultdict(int)
    for h, flist in inv.items():
        if len(flist) < 2:
            continue
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                a, b = sorted((flist[i], flist[j]))
                pair_overlap[(a, b)] += 1

    uf = _UF()
    for (a, b), inter in pair_overlap.items():
        dup, _ = _is_dup(inter, len(folder_hashes[a]), len(folder_hashes[b]))
        if dup:
            uf.union(a, b)

    comp = defaultdict(list)
    for f in folders:
        if f in uf.p:
            comp[uf.find(f)].append(f)

    groups = []
    for root, members in comp.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda f: len(folder_hashes[f]), reverse=True)
        grp = [{"path": f, "count": len(folder_hashes[f]),
                "role": "keep" if i == 0 else "dup"}
               for i, f in enumerate(members)]
        groups.append(grp)
    groups.sort(key=lambda g: sum(x["count"] for x in g), reverse=True)
    return groups


def find_duplicates_of(target_folder):
    """
    单文件夹模式：把 target_folder 和全盘其他文件夹比对，
    找出与它重复的文件夹（它作为子集或超集都算）。
    target_folder: 形如 'a整理分类/好吃的/晚饭/总/'（结尾带/）。
    返回：{"found": bool, "target": {...}, "matches": [...], "error": str|None}
    """
    folder_hashes = _load_folder_hashes()

    # 容错：用户可能没带结尾 /，或大小写/空格，先精确找，再宽松找
    if target_folder not in folder_hashes:
        # 尝试补/或去/
        alt = target_folder.rstrip("/") + "/"
        if alt in folder_hashes:
            target_folder = alt
        else:
            return {"found": False, "error": "not_exist",
                    "msg": f"未找到该文件夹（或它没有已索引的图片）：{target_folder}"}

    tset = folder_hashes[target_folder]
    if not tset:
        return {"found": False, "error": "empty",
                "msg": "该文件夹没有可用于比对的图片（pHash 为空）"}

    matches = []
    for f, hs in folder_hashes.items():
        if f == target_folder:
            continue
        inter = len(tset & hs)
        if inter == 0:
            continue
        dup, overlap = _is_dup(inter, len(tset), len(hs))
        if dup:
            # 判断关系
            r_target = inter / len(tset)   # 目标被覆盖的比例
            r_other = inter / len(hs)      # 对方被覆盖的比例
            if r_target >= 0.99 and r_other >= 0.99:
                rel = "完全相同"
            elif r_target >= 0.99:
                rel = "目标是对方的子集"
            elif r_other >= 0.99:
                rel = "对方是目标的子集"
            else:
                rel = "高度重叠"
            matches.append({"path": f, "count": len(hs),
                            "inter": inter, "overlap": round(overlap, 2),
                            "relation": rel})
    matches.sort(key=lambda x: x["overlap"], reverse=True)
    return {"found": len(matches) > 0,
            "target": {"path": target_folder, "count": len(tset)},
            "matches": matches, "error": None}
