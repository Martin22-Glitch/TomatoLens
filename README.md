# 🍅 TomatoLens

> 面向 PikPak 云盘的「以图搜图」系统 —— 为云盘建立一个本地化的视觉搜索引擎。

上传一张图片，即可在云盘中数十万张图片里秒级找出最相似的若干张；同时支持
按文件名 / 文件夹名关键词检索，以及智能检测内容重复的文件夹。

> **English:** [README_en.md](README_en.md)

---

## ✨ 功能特性

- **以图搜图**：CLIP 语义向量 + ORB 局部特征双重匹配，结果按相似度分组（1–5 组）。
- **视频封面检索**：自动抓取视频缩略图并入库，搜图时可命中视频文件。
- **关键词搜索**：支持多关键词，空格表示「同时满足(AND)」，`-词` 表示「排除(NOT)」，例：`晚饭 好吃 -视频`。
- **重复文件夹检测**：基于感知哈希(pHash) + 重叠系数 + 并查集，自动分组列出重复或包含关系的文件夹，支持全盘检测与指定文件夹检测两种模式。
- **增量同步**：幂等设计，重复运行仅处理差异，下载失败的图片会在下次同步时自动重试。
- **Web 界面**：在浏览器内完成建库、搜图、去重、关键词搜索及登录 / 登出。

---

## 🛠️ 技术栈

- 后端：Python + Flask + asyncio + httpx
- 向量检索：OpenCLIP + FAISS
- 特征 / 去重：OpenCV(ORB) + 感知哈希(pHash)
- 存储：SQLite（元数据）+ FAISS（向量）+ 本地缩略图缓存
- 云盘接口：pikpakapi（非官方 PikPak API 封装）

---

## 💻 运行环境

本项目的开发与测试环境如下，供参考：

- 操作系统：Windows 10/11
- Python：3.12
- GPU：NVIDIA RTX 4060（CUDA 12.6，用于加速 CLIP 推理；无 GPU 也可运行，仅速度较慢）
- 向量检索：FAISS 运行于 CPU（13 万量级下足够快），仅 CLIP 使用 GPU
- 数据规模：约 20 万文件、13 万张图片，缩略图缓存约 20 GB

其他环境的适配说明：

- **macOS / Linux**：将命令中的 `venv\Scripts\activate` 改为 `source venv/bin/activate`，
  其余流程一致。路径分隔符在界面中统一使用 `/`，无需额外处理。
- **纯 CPU 机器**：可正常运行，程序会自动检测并回退到 CPU。CLIP 推理速度会明显变慢，
  建议减小 `config.py` 中的 `CLIP_BATCH`。
- **无 GPU 但图片量大**：建库为一次性开销，可分文件夹、分批次在空闲时段完成。
- **Python 版本**：建议使用 3.10 ~ 3.12；过新或过旧的版本可能导致 torch / open_clip 等依赖无对应预编译包。

---

## 📦 安装

### 1. 克隆项目

```bash
git clone https://github.com/Martin22-Glitch/TomatoLens.git
cd TomatoLens
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

> ⚠️ **关于 torch / torchvision / faiss（重要）**
> 这三个库与具体的 CUDA 环境强相关，`requirements.txt` 中**未包含**它们，需要按
> 自己的环境单独安装：
> - **CPU 用户**：
>   到 [pytorch.org](https://pytorch.org) 选择 CPU 版命令一次性安装 torch + torchvision，
>   再执行 `pip install faiss-cpu`。
> - **GPU 用户**：
>   到 [pytorch.org](https://pytorch.org) 选择与本机 CUDA 版本对应的命令一次性安装
>   torch + torchvision，再执行 `pip install faiss-gpu`。
>
> 若安装速度慢或失败，可使用国内 PyPI 镜像，例如：
> `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>`；
> torch 也可从官方提供的 wheel 直链下载 `.whl` 后用 `pip install 路径.whl` 本地安装。

### 3. 准备 CLIP 模型

将 OpenCLIP 的 `ViT-B-32` 权重文件 `open_clip_pytorch_model.bin` 放到 `models/` 目录下；
也可不放，首次运行时程序会自动从镜像下载到 `hf_cache/`（项目已默认配置 HuggingFace 国内镜像）。

### 4. 配置账号

```bash
copy config_local.example.py config_local.py   # Windows
# cp config_local.example.py config_local.py    # macOS / Linux
```

编辑 `config_local.py`，填入 PikPak 邮箱与 device_id。无需配置工作目录。

---

## 🚀 使用

### 启动服务

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:5000`。

### 登录

点击页面右上角「登录」，输入 PikPak 密码，成功后 token 会被保存并在过期时自动刷新。
也可使用命令行方式登录：`python pikpak_login.py`。

### 建库 / 同步

在「建库 / 更新」区域：

- **全盘同步**：遍历整个网盘进行对账（新增 / 更新 / 删除），适合定期校正。
- **索引此文件夹**：在左侧文件夹树点选或手动输入路径（层级用 `/` 分隔），仅做增量，速度快。

首次全盘建库为一次性开销，耗时取决于图片数量与网络速度。

### 搜图

上传图片，选择分组数量，点击「查询」。结果会标注「同图 / 疑似 / 相似」。

### 关键词搜索

直接输入，支持多词与排除，例：`风景 海边 -缩略图`。命中的文件夹可点击直接填入索引 / 去重输入框。

### 去重检测

- 留空：检测全盘所有重复文件夹组。
- 填路径（如 `分类/好吃的/晚饭/总`）：将该文件夹与全盘比对，找出内容重复的文件夹（子集 / 超集均算）。

---

## ⚠️ 注意事项

- **隐私安全**：`config_local.py` 与 `data/token.json` 含账号凭证，若不慎泄露，应立即修改 PikPak 密码。
- **磁盘占用**：缩略图缓存可达数十 GB，取决于文件数量，存放于 `data/thumbnails/`。
- **首次建库成本**：会下载全部缩略图并计算 CLIP / ORB / pHash，属一次性开销，之后为增量同步。
- **网络与限速**：缩略图来自 PikPak 服务器，下载速度受网络及服务端限速影响，建议夜间分批进行。
- **token 失效**：若提示 `invalid refresh token`，重新在网页登录即可。
- **合规使用**：本项目仅供个人学习及管理自有云盘数据使用，请遵守 PikPak 服务条款及相关法律法规。

---

## 🧰 常见问题（FAQ / Troubleshooting）

**Q：安装依赖时 torch / faiss 下载失败或安装报错？**
A：这两个库体积大且与 CUDA 强相关，最易出问题。建议：(1) 按上文「关于 torch 与 faiss」单独安装对应版本；(2) 国内网络可使用清华等 PyPI 镜像，例：
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>`；
(3) torch 也可从 PyTorch 官方提供的 wheel 直链手动下载 `.whl` 后用 `pip install 路径.whl` 本地安装。

**Q：CLIP 模型下载很慢或失败？**
A：项目默认已设置 HuggingFace 镜像（`hf-mirror.com`）。也可手动下载 `ViT-B-32` 的
`open_clip_pytorch_model.bin` 放入 `models/` 目录，跳过自动下载。

**Q：提示 `database is locked`？**
A：多为同步任务与查询并发写入所致。项目已对 SQLite 启用 WAL 模式与 30 秒超时；若仍出现，
请避免在同步进行时频繁触发其他写操作，或稍后重试。

**Q：搜图结果缩略图显示 404？**
A：通常是该图缩略图下载失败或尚未入库。重新对所在文件夹执行一次同步即可补齐。

**Q：登录触发验证码 / 风控？**
A：请在平时常用的网络环境下登录；非官方 API 偶发风控，可稍后重试。

---

> 本项目为个人学习项目，与 PikPak 官方无任何关联。
