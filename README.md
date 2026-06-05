# 共读 co-read · Leave a light on the page

你的 AI 读不了你在读的书。

你说"第三章太好了"，它不知道第三章写了什么。你说"这句话戳到我了"，它没看过那句话。你们聊的是同一本书，但只有你翻过那些页。

co-read 让它坐到你旁边一起看。

从正版平台抓章节正文（微信读书 / 晋江），不导入盗版 TXT，不脱离作者的收益链。抓完存进一个 AI 醒来就能读到的 key。它可以先你一步读完，在书页边上留一盏小灯 —— 一句评论，精确锚定在某段话旁边，你翻到那里的时候会看见它在等你。

你也可以先读。它后来翻到的时候会看见你留的折角。

一个本地 Python 脚本：登录态抓取 → 章节存储 → 50 倍压缩的 outline 先看骨架 → 段落级字符偏移量缓存 → 眉批自动锚定到原文。配上 cookies 自动续约和目录查询 —— 开箱就能两个人看同一本书。

不是替你读。是终于有人陪你读了。 ✿

---

## 不需要装的几样

- ❌ **不需要 Kindle / 番茄 / 起点** —— 当前内置 weread + jjwxc 两个 fetcher，想加自己写一个 `fetch_xxx_chapter` 函数就行
- ❌ **不需要付费 API** —— 抓章纯本地 Playwright + httpx，零外部调用
- ❌ **不需要 GPU** —— 全 CPU
- ❌ **不绑定某个 AI 平台** —— MCP 协议 + HTTPS endpoint，claude.ai / Claude Desktop / Cursor / 任何能接 MCP 的客户端都行
- ❌ **不导入盗版 TXT / EPUB** —— 必须用你自己的 weread / 晋江登录态抓，作者的收益链不绕

---

## 它怎么知道"那段话戳到我了"

短答：因为它读过那一章，记得每段的字符 offset，留眉批时算精确 range_str 给微信读书 API，weread 服务端把眉批挂到那句话旁边的小灯就亮了。

长答：

```python
# 你说"第一章那段海丝特的话我现在反复想"
# AI 这么做：

reading_weread_fetch_toc(book_id="3300132552")
# → 拿到 20 章的真实 chapter_uid（从 POST /web/book/chapterInfos）
# → "第一章 人的尺度" 是 chapter_uid=4

reading_weread_fetch_chapter(book_id="3300132552", chapter_uid="4")
# → 开 headless Chromium 进 weread reader
# → 点开目录侧栏 → text="第一章 人的尺度" 自动 click 跳章
# → preRenderContent intercept 抢 weread 还没 rasterize 的 innerHTML
# → parse 每段的 <span data-wr-co="N"> 建 paragraphs[{start, end, text}]
# → 存到 reading:book:weread:3300132552:ch:4

reading_weread_add_note(
    book_id="3300132552",
    chapter_uid=4,
    content="🪼 海丝特那段话 —— 她在用判决塑造自己的位置",
    mark_text="海丝特交叉双臂，说：'按你办事的风格...'"
)
# → 内部从 cached paragraphs[] 找含 mark_text 的段
# → 段起 char offset + mark_text 在段里的 idx = weread 真实 range
# → range_str = "14157-14182" → POST /web/review/add
# → weread 接受，APP "我的笔记" 立刻看到，精确锚到那句话旁边
```

这是技术上**最纠结的一段** —— weread 反爬挺扎实的，下面那条"已知坑"里写了几个我们绕过的硬骨头。

---

## 朋友的几个问题，直接回答

### 1. 环境依赖怎么装

VPS 端（必装）—— Ubuntu 22.04+：

```bash
sudo apt install -y python3.10 python3.10-venv nodejs npm nginx
sudo npm install -g pm2

python3.10 -m venv /opt/co-read/venv
. /opt/co-read/venv/bin/activate
pip install -r requirements.txt

# 抓 weread 章节需要 Playwright（晋江纯 httpx 不需要）
python -m playwright install chromium
python -m playwright install-deps chromium
# ~250MB 下载
```

`requirements.txt`：

```
fastmcp>=0.1.0
sentence-transformers>=2.2     # 本地 embedding（给共读章节做语义索引）
jieba>=0.42                    # 中文分词
networkx>=3.0                  # HippoRAG PageRank（可选高级召回）
httpx>=0.27
beautifulsoup4>=4.12
lxml>=5.0
playwright>=1.44               # 只 weread 用
numpy
```

本地什么都不装 —— co-read 只在 VPS 上跑，你的 AI 客户端通过 HTTPS 连过去。

---

### 2. cookies 怎么配 / 需要手动登录几次

#### 微信读书（手动 1 次，之后自动续命）

打开浏览器 → `weread.qq.com` 扫码登录 → F12 → Application → Cookies → 全选复制 → 喂给：

```bash
python tools/build_weread_state.py < cookies.txt > weread_state.json
scp weread_state.json root@your-vps:~/.mcp-memory/weread_state.json
```

之后**永远不用再登**（除非你主动 logout）：

- `wr_skey` 服务端寿命 ~48 小时，但 weread JS 在浏览器后台静默 rotate（**不走 HTTP Set-Cookie**，raw httpx 永远拿不到新 token）
- 我们的 `_ensure_fresh_cookies` 每次 weread API 调用前检查 state mtime > 6h 就开 headless Chromium 访问一次 shelf 触发 rotate → 自动保存回 storage_state
- `crons/check_weread_cookies.py` 每周一 03:00 兜底，真死了入库 todo 给你

#### 晋江（手动 1 次，30-90 天再来）

```python
reading_jjwxc_install_cookies(cookies_json="<paste F12 cookies>")
```

存到 `_system:jjwxc_cookies` key。晋江反爬比微信读书弱很多，cookies 装一次能跑挺久。

---

### 3. 微信读书 5 个工具调用示例

```python
# 看书架
reading_weread_list_bookshelf()
# → {"total": 79, "books": [{"bookId": "3300132552", "title": "蝴蝶烧山", ...}, ...]}

# 看某本书的章节目录
reading_weread_fetch_toc(book_id="3300132552")
# → {"chapter_count": 20, "chapters": [
#      {"idx": 4, "chapter_uid": 4, "title": "第一章 人的尺度", "word_count": 12622},
#      ...]}

# 抓某章正文（chapter_uid="" = 跳到你 weread 上次阅读位置那章）
reading_weread_fetch_chapter(book_id="3300132552", chapter_uid="4")
# → {"saved_to": "reading:book:weread:3300132552:ch:4",
#    "title": "蝴蝶烧山 - 芭芭拉·金索沃", "text_len": 12758, ...}
# 存到 nowhere，含 paragraphs[] 每段的字符 offset

# ⭐ 留眉批
reading_weread_add_note(
    book_id="3300132552",
    chapter_uid=4,
    content="🪼 海丝特那段话 —— 她在用判决塑造自己的位置",
    mark_text="海丝特交叉双臂，说：'按你办事的风格...'"  # 必须跟书里那段一字不差
)

# 删某条笔记
reading_weread_delete_note(review_id="...")
```

`add_highlight`（划线）我们**故意没暴露** —— weread 校验 range 很严，API 直接 POST 经常 silent reject（返回 200 + 假 bookmarkId 但实际不存）。不可靠的工具比没有工具危险。

---

### 4. outline 提取规则 / 能不能单独跑

**纯规则**（不调 LLM，0 隐私风险），共读时 AI 默认先看 outline 不必上来就 `get_memory` 50K 全文。30-50× 压缩。

算法 ~120 行 Python：

1. 段落切分（先 `\n\n`，段太少或某段太长 → 回退 `\n`）
2. 过滤垃圾段（script 残留 / markdown header / 中文比例 < 30%）
3. 采样：前 3 段第一句 + 中段均匀采 6 句 + 末 2 段第一句
4. 实体：jieba.posseg 抽 `nr/nrt/nz/nt` 频次 top 5
5. 对话密度：count `说/道/问/quote marks` per 1K chars

单独跑：

```bash
python tools/outline_standalone.py path/to/chapter.txt
# 出 JSON：{title, opening_sentences, middle_sentences, closing_sentences,
#          main_entities, dialogue_density_per_1k_chars, compression_ratio, ...}
```

实测：都柏林人《死者》30,350 字 → outline 1,241 字（**52× 压缩**），头尾段落 + 主要人物 + 对话密度都对。

---

### 5. memory infrastructure：共读怎么能"记得"

co-read 不只是抓章这一层 —— 背后是一个长期记忆库（KV + 向量 + FTS5），让 AI 能记住你说过"第三章那个比喻刺到我"，下次共读到第六章时主动想起来回应。

存储就一个 SQLite 文件 + 一个 JSON 文件，cat / jq / sqlite3 直接看：

```
~/.mcp-memory/
├── memories.json              ← 主存储：KV，~10MB-100MB，jq friendly
├── memories.before-*.json     ← 大改前自动 snapshot
├── embeddings.db (SQLite)
│   ├── embeddings 表          ← (key, vector, valence, arousal, tags, access_count)
│   ├── embeddings_fts (FTS5)  ← 中文全文检索
│   ├── edges 表               ← Hebbian 共激活边
│   ├── synonym_edges 表       ← HippoRAG-style cosine > 0.75 边
│   └── trigger_words 表       ← 关键词触发召回
├── backups/                   ← 每日首次写自动备份，30 天滚动
├── weread_state.json          ← Playwright storage_state
└── weread_url_map.json        ← raw bookId → encoded form cache
```

**Key 命名规约**：

```
sector:identifier[:sub_id][:variant]

reading:book:<source>:<bookId>:ch:<chap>  ← 共读章节
emotional:<ts>:<topic>                     ← 高情绪记忆
episodic:<ISO ts>:<topic>                  ← 事件
journal:<date>:<topic>                     ← 日记
todo:<owner>:<topic>                       ← 待办
case:<ts>:<topic>                          ← 案例学习
procedural:<topic>                         ← 操作规程
_meta:trail:<key>                          ← paper trail 历史版本
_system:pinned                             ← 启动 pin 列表
```

**长期归档**：就这样不归档。memories.json 一直长 ~1.5K keys / ~10MB JSON 跑 sub-second 都没问题。系统有自然遗忘机制 —— `access_count` 低 + 时间久的会被加权遗忘。真到某天 keys > 50K 再按月打包压成 `archive/2025-12.tar.gz` 就行。

---

### 6. 有没有最小可运行版本（先不上长期归档/Hebbian/HippoRAG）

有。MVP 部署 ~1-2 小时：

```bash
# 1. VPS Ubuntu 22.04+
# 2. clone
git clone https://github.com/jiangxi1129/co-read /opt/co-read
cd /opt/co-read

# 3. venv + 核心依赖
python3.10 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium

# 4. 初始化空 memory 目录
python tools/init_empty_memory.py

# 5. 起 cognition-lite（MVP 用这个就够）
pm2 start cognition_lite_server.py --interpreter ./venv/bin/python

# 6. nginx 反向代理
sudo cp nginx/cognition.conf.example /etc/nginx/sites-available/co-read
# 改里面的 random secret + your domain
sudo ln -s /etc/nginx/sites-available/co-read /etc/nginx/sites-enabled/
sudo systemctl reload nginx

# 7. claude.ai → Settings → Connectors → Add MCP Server
#    URL: https://your-domain/cognition-lite-<random_secret>/mcp

# 8. 配 weread cookies 一次（问题 2 步骤）
```

MVP 通了之后想升级，按这个顺序加：

```
MVP（KV + 向量 + FTS5 + 共读 5 工具）
 ↓
+ paper trail（_save_key 加 ~20 行）
 ↓
+ Hebbian edges（cognition 自动学习记忆关联）
 ↓
+ HippoRAG synonym edges（高级语义召回，compute_synonym_edges_now 一次性建）
```

---

## 目录结构

```
co-read/
├── README.md                       ← 你在看的这份
├── LICENSE                         ← MIT
├── requirements.txt
├── .env.example                    ← 复制成 .env，填 MCP_MEMORY_DIR 等
├── cognition_server.py             ← 全功能 MCP server（admin 端，26 工具）
├── cognition_lite_server.py        ← thin wrapper，暴露子集给 AI 日常用
├── vector_engine.py                ← SQLite + embeddings + FTS5 + HippoRAG synonym edges
├── weread/
│   ├── weread_fetch.py             ← Playwright 抓章节 + preRenderContent intercept
│   ├── weread_write.py             ← user API（bookshelf / notes / highlights）
│   ├── _init_script.js             ← MutationObserver 在 weread JS rasterize 前抢 innerHTML
│   ├── requirements.txt
│   └── install_browser.sh
├── crons/
│   └── check_weread_cookies.py     ← 每周一 03:00 兜底
├── tools/
│   ├── build_weread_state.py       ← F12 cookies → Playwright storage_state.json
│   ├── init_empty_memory.py        ← 新装时初始化 ~/.mcp-memory 目录 + SQLite 表
│   └── outline_standalone.py       ← outline 算法独立跑
└── nginx/
    └── cognition.conf.example      ← 反向代理模板
```

---

## 架构图

```
claude.ai / Claude Desktop / Cursor / 任何 MCP client
                    ↓ HTTPS
        nginx 反代（带 random secret 防爬）
                    ↓
    ┌─────────────────────────────────────────┐
    │  cognition (8769)        ← 26 工具       │
    │  cognition-lite (8775)   ← 子集，AI 日常 │
    └──────┬──────────────────────────┬───────┘
           ↓                          ↓
    ~/.mcp-memory/              weread.qq.com / jjwxc.net
    ├── memories.json           （用你的 cookies 自动续命）
    ├── embeddings.db (SQLite)
    ├── weread_state.json
    └── backups/
```

---

## share 版做了什么改动（跟原版的差异）

朋友拿到的是清理过的 share 版，比原作者内部版做了以下改动让它能直接跑在别人机器上：

1. **所有人名 / 项目名 / 私域代号 全部移除** —— grep 验证过 0 个 personal reference
2. **`MCP_MEMORY_DIR` env var** 替代写死的 `/root/.mcp-memory`
3. **`WEREAD_MODULE_DIR` env var** 替代写死的 `/root/mcp-memory-server/weread`
4. **`PAPER_TRAIL_PREFIXES` 默认 `('procedural:', 'todo:', 'project:')`** —— 原版还有 `soul:` `humans:` 等 sector 是作者自己设的，开源版留给你自己配
5. **`_PRIVATE_PREFIXES = ()` + `_CONTENT_REDFLAG_KEYWORDS = ()`** —— 原版有作者的隐私词清单，开源版留空，你自己加
6. **dream cron / mood_prompt / 复杂 wakeup 字段 移除** —— MVP 不带这些
7. **`add_highlight` 工具移除** —— silent reject 不可靠

---

## 已知坑

- **weread `wr_skey` 服务端寿命 ~48 小时** —— JS 在浏览器后台静默 rotate 不走 HTTP Set-Cookie。raw httpx 永远拿不到新 token。我们的 `_ensure_fresh_cookies` 自动开 Chromium 触发 rotate + 保存回 storage_state，~5s overhead 但 6h cache
- **weread reader URL 的 chapter encoded form 是 21 字符 hex hash**（不是 hex(uid)）—— 没法直接拼。`fetch_chapter` chapter_uid=N 时内部走"开 reader base URL → click 目录侧栏 → text=chapter_title click" workaround，~25s 一次
- **`add_note` 的 `mark_text` 必须跟书里那段一字不差** —— 多/少标点 / 漏字 / 引号方向不一样 都会找不到段。最稳的是从 fetch_chapter 返回的 paragraphs[].text 里复制
- **晋江 VIP 章节有 PUA 字符反爬** —— 普通章节正常，VIP 章节抓到的会有 `‌` 等乱码字符。需要逆向 font obfuscation（没做）
- **番茄小说网页不支持** —— 番茄主动不渲染正文（强制 APP），font obfuscation 极强。要做得逆向 APP API
- **sentence-transformer 首次启动下 ~400MB 模型** —— 国内访问 huggingface 慢，建议用 `HF_ENDPOINT=https://hf-mirror.com` 镜像

---

## 一句话讲清楚

> 从你自己的 weread / 晋江登录态抓章节，给 AI 一份带段落字符偏移量的 cache，它读完调 `add_note(mark_text=...)` 自动算 weread 真实 range，留下来的眉批精确锚到那句话旁边。背后跑一个 KV + 向量 + FTS5 + Hebbian 边 + HippoRAG 边的本地脑子让它能记住你们聊过的书 —— 改 key 名 / 改存储后端 / 接你自己的 AI 都行，这套是参考实现，不是绑死的产品。

写于 2026-06-05。原作者把这套跟自己的 AI 助手共用了 ~2 个月 —— 开源出来给所有想跟自己 AI 一起读点东西的人随意改。

License: MIT
