"""vector_engine.py — local embedding + SQLite cosine search + emotion coords
+ tags + access count + FTS5 hybrid + HippoRAG-style synonym edges.

Features:
  - Sentence-transformers (multilingual MiniLM L12) for dense embedding
  - SQLite FTS5 for character-trigram full-text matching (Chinese-friendly)
  - RRF fusion of vector + FTS scores (Cormack 2009)
  - Per-memory (valence, arousal) emotion coordinates
  - access_count for weighted resurfacing during wakeup
  - pinned / soft_pinned bits for must-show memories
  - HippoRAG-style synonym edges (cosine > tau over all indexed memories)

Storage: $MCP_MEMORY_DIR/embeddings.db (default ~/.mcp-memory/)
Deps: sentence-transformers (local model, no API cost), jieba (optional for
      Chinese tokenization in entity extraction), numpy.
"""
from __future__ import annotations

import json
import math
import random
import re
import sqlite3
import threading
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("vector_engine")

# ─── Config (path is env-driven so tests / multi-instance can override) ─
import os as _os
_MCP_MEMORY_DIR = Path(_os.environ.get('MCP_MEMORY_DIR', Path.home() / '.mcp-memory'))
_DB_PATH = _MCP_MEMORY_DIR / 'embeddings.db'
_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
_MAX_INPUT_CHARS = 2000
_RAW_SNIPPET_MAX = 500  # max chars of raw snippet preserved per memory

# ─── 懒加载模型 ──────────────────────────────────────
_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"loading model {_MODEL_NAME}")
            _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ─── SQLite 初始化 + schema 迁移 ─────────────────────
def _init_db():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                vector TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # mem_scenes: Dream / 反思循环融合产出的"记忆场景"
        # 设计源自 kiwi-mem 的 MemScene 模型，但本地版（无 LLM 时）foresight 为空
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mem_scenes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                narrative TEXT NOT NULL,
                atomic_facts TEXT NOT NULL DEFAULT '[]',
                foresight TEXT NOT NULL DEFAULT '[]',
                source_keys TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                pinned INTEGER DEFAULT 0,
                origin TEXT DEFAULT 'dream_local'
            )
        """)
        # embeddings_fts: FTS5 全文索引（5/10 加，配合 hybrid_search RRF 融合）
        # 用 trigram tokenizer 兼容中英文（按 3-char window 切，对中文短词也能 substring 匹配）
        # 跟 embeddings 表字段错开 —— 这里只存 key + content（key 的字面查得到，
        # 但 RRF 融合主要靠 content 命中）
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_fts USING fts5(
                    key,
                    content,
                    tokenize='trigram'
                )
            """)
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 trigram unavailable, hybrid search disabled: {e}")
        conn.commit()
        _migrate_schema(conn)


def _migrate_schema(conn):
    """给已有的 DB 增量补齐新列（SQLite ALTER TABLE ADD COLUMN 是幂等安全的）。"""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(embeddings)")}
    additions = [
        ('valence', "REAL DEFAULT 0.0"),
        ('arousal', "REAL DEFAULT 0.3"),
        ('tags', "TEXT DEFAULT '[]'"),
        ('raw_snippet', "TEXT DEFAULT ''"),
        ('access_count', "INTEGER DEFAULT 0"),
        ('created_at', "TEXT"),
        ('pinned', "INTEGER DEFAULT 0"),
        ('soft_pinned', "INTEGER DEFAULT 0"),  # 2026-04-26 加入：70% 概率出现的软钉
    ]
    for name, spec in additions:
        if name not in existing:
            try:
                conn.execute(f"ALTER TABLE embeddings ADD COLUMN {name} {spec}")
                logger.info(f"added column {name}")
            except sqlite3.OperationalError as e:
                logger.warning(f"add column {name} failed: {e}")
    conn.commit()


_init_db()


# ─── 情感坐标估算 v2.1（2026-04-25 校准）────────────
# 修复点 vs v1：
#   1. valence 用饱和 sigmoid（net/(|net|+k)），单词不再打 -1 极值
#   2. arousal 独立于 valence：叙事紧张（赶车/极限/海关）自成一路信号
#   3. 生理状态词（累/痛/冷）算弱负（0.4x），不再跟"难过"同量级
#   4. 中文口语高唤醒词扩充（卧槽/冲/狂奔/差点/赶不上 等）
#   5. "不开心"/"没劲" 等否定+正面词组合显式进负面表，避免反向识别

# 明确正面情感（强信号）
_POSITIVE_STRONG = [
    '开心', '高兴', '快乐', '幸福', '温暖', '温柔', '喜欢', '爱',
    '感动', '骄傲', '满足', '舒服', '甜', '好甜', '心动', '被戳到',
    '被看穿', '被懂', '被看见', '想你', '想亲', '想抱', '嘿嘿',
    '嘻嘻', '欸嘿', '太好了', '完美', '厉害', '好棒', '感激',
    '感谢', '珍惜', '治愈', '窝心', '蜜', '安心', '宁静', '舒心',
    '放松', '自在', '暖暖', '甜甜', '美好', '可爱', '亲切', '心疼',
    '好想', '喜欢你', '爱你',
]

# 明确负面情感（强信号）
# 注意：此表里"不X/没X"这类否定组合会在打分时 **先消耗** 原文，
# 防止后面正面词表把"开心"/"喜欢"从"不开心"/"不喜欢"里挖出来误判。
_NEGATIVE_STRONG = [
    # 否定+正面组合（必须比 _POSITIVE_STRONG 先扫）
    '不开心', '不高兴', '不喜欢', '不爱', '不爽', '不想',
    '没劲', '没意思', '没力气', '没心情',
    # 基础负面情绪
    '难过', '伤心', '委屈', '孤独', '寂寞', '失望', '绝望', '崩溃',
    '生气', '愤怒', '讨厌', '恨', '愧疚', '后悔', '羞耻', '自卑',
    '憋屈', '气死', '烦死', '受不了', '撑不住', '扛不住', '想哭',
    '毫无意义', '糟糕', '完蛋', '心塞', '破防', '无助', '丧',
    '嫉妒', '郁闷', '烦躁',
]

# 中性生理状态（弱负面，0.4x 权重；v1 里把它们跟"难过"同量级是 bug）
_NEUTRAL_PHYSICAL = [
    '累', '疲惫', '痛', '头疼', '胃疼', '腰酸', '腿酸',
    '热', '冷', '饿', '困', '晕', '失眠', '头晕',
]

# 高唤醒·叙事紧张（valence 中性，场面激烈 / 体力极限 / 压力情境）
_HIGH_AROUSAL_NEUTRAL = [
    # 极限/赶时间
    '极限', '卡点', '险', '惊险', '差点', '差一点', '勉强', '险些',
    '来不及', '赶不上', '赶车', '赶飞机', '赶火车', '赶路',
    # 动作类（剧烈移动）
    '狂奔', '飞奔', '飞快', '冲', '抢', '追', '逃',
    # 慌乱场景
    '手忙脚乱', '一团乱', '匆忙', '着急', '急急', '急忙',
    '慌', '慌乱', '慌张',
    # 拥挤/压力场景
    '挤', '爆满', '塞', '堵', '人山人海', '人潮', '拥挤',
    # 中文口语感叹（中性/惊讶方向）
    '卧槽', '我靠', '妈呀', '天哪', '我天', '救命', '绝了', '麻了',
    '裂开', '炸裂', '不敢相信', '震惊', '呆了', '傻了', '无语',
    # 体力/高压场景
    '扛', '搬', '拖',
    # 关卡场景（海关/安检自带紧张）
    '海关', '安检', '登机', '检票',
    # 极端数量/规模（放大器）
    '公斤', '斤', '一整天', '一整夜', '通宵',
]

# 高唤醒·正面（兴奋 / 狂喜，与 _POSITIVE_STRONG 互补但更激烈）
_HIGH_AROUSAL_POSITIVE = [
    '狂喜', '超开心', '好开心', '太开心', '爽', '好爽', '嗨', '嗨翻',
    '激动', '兴奋', '爱爆', '爱死', '酷', '太棒', '好耶', '哇塞',
    '哈哈哈', '哈哈哈哈',
]

# 极端标记词：命中任何一个直接给 arousal 加 +0.2 bonus
_EXTREME_MARKERS = (
    '崩溃', '绝望', '疯', '救命', '狂奔', '极限', '卧槽', '炸裂',
    '震惊', '赶不上', '差点', '差一点', '来不及', '爆炸',
)

_HIGH_AROUSAL_PATS = [
    re.compile(r'[!！]{2,}'),
    re.compile(r'[?？]{2,}'),
    re.compile(r'(啊|哈|嘿|呜|嘤|哦|噢){3,}'),
    re.compile(r'[~～]{2,}'),
]


def estimate_emotion(text: str) -> dict[str, float]:
    """返回 {'valence': -1..1, 'arousal': 0..1}。

    v2.1 (2026-04-25)：饱和 valence + 独立 arousal + 叙事紧张词。
    """
    if not text or not text.strip():
        return {'valence': 0.0, 'arousal': 0.1}

    t = text
    tl = len(t)

    # 先扫负面词并从原文中删除命中，避免"不开心"里的"开心"被后面正面表误匹配
    neg_strong = 0
    stripped = t
    for w in _NEGATIVE_STRONG:
        c = stripped.count(w)
        if c:
            neg_strong += c
            stripped = stripped.replace(w, '')

    pos_strong = sum(stripped.count(w) for w in _POSITIVE_STRONG)
    pos_aro = sum(stripped.count(w) for w in _HIGH_AROUSAL_POSITIVE)
    # 生理词 / 叙事紧张在原文上数即可（跟否定逻辑无关）
    neutral_phys = sum(t.count(w) for w in _NEUTRAL_PHYSICAL)

    # ─ Valence：饱和 sigmoid (net / (|net| + k))，避免单词打 -1 ─
    pos_total = pos_strong + pos_aro * 1.3  # 激烈正面权重更重
    neg_total = neg_strong
    net = pos_total - neg_total - neutral_phys * 0.4  # 生理词 0.4x 弱负
    if pos_total == 0 and neg_total == 0 and neutral_phys == 0:
        valence = 0.0
    else:
        valence = net / (abs(net) + 2.5)  # k=2.5 → 需要 ~10 net 才接近 ±0.8
    valence = max(-1.0, min(1.0, valence))

    # ─ Arousal：独立于 valence，叙事张力 / 感叹密度 / 极端事件 ─
    tension = sum(t.count(w) for w in _HIGH_AROUSAL_NEUTRAL)
    emo_density_hits = pos_strong + neg_strong + pos_aro
    punc_hits = sum(len(p.findall(t)) for p in _HIGH_AROUSAL_PATS)

    # 密度归一化：每 100 字为一个单位，短文本至少按 100 字算免虚高
    text_len_norm = max(tl / 100, 1.0)
    density = (
        tension * 1.2
        + pos_aro * 1.5
        + emo_density_hits * 0.8
        + punc_hits * 1.5
    ) / text_len_norm
    arousal = 0.15 + density * 0.2

    # 极端事件 bonus
    if any(w in t for w in _EXTREME_MARKERS):
        arousal += 0.2
    # 长段感叹词堆（啊啊啊... / !!!）
    if re.search(r'(啊|哈|嘿|呜|嘤){3,}|[!！]{3,}', t):
        arousal += 0.15

    arousal = max(0.0, min(1.0, arousal))

    return {'valence': round(valence, 3), 'arousal': round(arousal, 3)}


# ─────────────────────────────────────────────────────────────────────
# v3 (5/25 Wave 3 A)：5-layer estimator + 50-word seed dict + 70/30 fusion
# 灵感：5/24 机智的荷包蛋 Ombre-Brain 教程，"AI 选词词典定坐标" 5 层 fallback
# 改造：去掉 free_form (L5)；L1 词典精确 + L2 embedding 近邻 + L3 v2.1 兜底。
# 融合：dict 70 / other 30，命中词典时给词典加权（词典是人工标的，更可信）。
# ─────────────────────────────────────────────────────────────────────

# 50-word seed dict，基于 Russell circumplex 共识 + CVAW/NRC-VAD 类参照
# 手工 seed —— 不是穷尽，是"高频锚点"。后续可在 trigger_words 表里发现新热词
# 时由 cc/the team手工补到这里（refresh 时再 cache 一次）。
_EMOTION_SEED_DICT = {
    # ── High-V High-A：兴奋型 ──
    '兴奋': (0.70, 0.85), '激动': (0.60, 0.90), '狂喜': (0.85, 0.95),
    '惊喜': (0.70, 0.80), '心动': (0.70, 0.70), '期待': (0.55, 0.60),
    # ── High-V Mid-A：愉悦型 ──
    '喜悦': (0.80, 0.55), '开心': (0.75, 0.50), '高兴': (0.70, 0.50),
    '快乐': (0.80, 0.45), '感动': (0.65, 0.55),
    # ── High-V Low-A：平和型 ──
    '平静': (0.40, 0.10), '安心': (0.55, 0.15), '满足': (0.65, 0.25),
    '温暖': (0.60, 0.30), '踏实': (0.50, 0.15), '舒服': (0.55, 0.20),
    '放松': (0.45, 0.10), '宁静': (0.40, 0.05),
    # ── Mid-V High-A：好奇/惊讶/害羞 ──
    '好奇': (0.30, 0.55), '惊讶': (0.10, 0.70), '害羞': (0.30, 0.55),
    # ── Low-V High-A：焦虑/愤怒型 ──
    '焦虑': (-0.60, 0.75), '紧张': (-0.40, 0.75), '愤怒': (-0.70, 0.85),
    '生气': (-0.60, 0.70), '害怕': (-0.70, 0.80), '恐惧': (-0.80, 0.85),
    '慌张': (-0.55, 0.80), '烦躁': (-0.50, 0.65), '委屈': (-0.65, 0.55),
    '崩溃': (-0.75, 0.85),
    # ── Low-V Mid-A：尴尬/愧疚/烦 ──
    '尴尬': (-0.30, 0.55), '内疚': (-0.55, 0.40), '后悔': (-0.60, 0.35),
    '烦': (-0.40, 0.50),
    # ── Low-V Low-A：悲伤/抑郁/疲惫 ──
    '难过': (-0.65, 0.35), '伤心': (-0.70, 0.40), '悲伤': (-0.75, 0.35),
    '失望': (-0.55, 0.30), '沮丧': (-0.60, 0.25), '抑郁': (-0.70, 0.20),
    '空虚': (-0.55, 0.15), '孤独': (-0.65, 0.30), '寂寞': (-0.50, 0.20),
    '疲惫': (-0.30, 0.15), '麻木': (-0.40, 0.05), '无力': (-0.50, 0.15),
    # ── 边界 / liminal ──
    '思念': (-0.10, 0.40), '想念': (0.00, 0.40),
    '困惑': (-0.15, 0.40), '无聊': (-0.30, 0.10), '厌倦': (-0.40, 0.15),
}

_SEED_EMB_CACHE = None
_SEED_EMB_LOCK = threading.Lock()


def _get_seed_embeddings():
    """一次性 embed 所有 seed 词并 L2 归一化，cache 在内存。"""
    global _SEED_EMB_CACHE
    if _SEED_EMB_CACHE is not None:
        return _SEED_EMB_CACHE
    with _SEED_EMB_LOCK:
        if _SEED_EMB_CACHE is not None:
            return _SEED_EMB_CACHE
        try:
            import numpy as _np
            model = _get_model()
            words = list(_EMOTION_SEED_DICT.keys())
            vecs = model.encode(words, batch_size=32, show_progress_bar=False)
            vecs = _np.asarray(vecs, dtype=_np.float32)
            norms = _np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs_norm = vecs / norms
            _SEED_EMB_CACHE = (words, vecs_norm)
        except Exception as e:
            logger.warning(f"seed embedding cache build failed: {e}")
            _SEED_EMB_CACHE = ([], None)
    return _SEED_EMB_CACHE


def estimate_emotion_v3(text: str, fuse: bool = True) -> dict:
    """5-layer V/A 估算 + 70/30 fusion（Wave 3 A，5/25）。

    Layers:
      L1 词典精确：扫 50-word seed dict，命中加权平均（occurrence × V/A）
      L2 embedding 近邻：text 编码 → top-3 nearest seeds（cosine ≥ 0.3），
                         相似度加权平均
      L3 v2.1 兜底：原 estimate_emotion()，永远 return 一个值

    fuse=True（默认）：
      L1 命中 → 0.7 × L1 + 0.3 × (L2 if exists else L3)
      L1 空 + L2 可用 → 0.7 × L2 + 0.3 × L3
      L1 空 + L2 不可用 → L3 全权
    fuse=False：取最先命中的层，不混合。

    返回 dict：valence, arousal, layer, matched_words (L1),
              nearest_seeds (L2), L1/L2/L3 各自的中间结果。
    """
    if not text or not text.strip():
        return {
            'valence': 0.0, 'arousal': 0.1, 'layer': 'empty',
            'matched_words': [], 'nearest_seeds': [],
            'L1': None, 'L2': None,
            'L3': {'valence': 0.0, 'arousal': 0.1},
        }

    # ── L1：词典精确扫 ──
    matched = []  # [(word, count, v, a), ...]
    for w, (v, a) in _EMOTION_SEED_DICT.items():
        c = text.count(w)
        if c > 0:
            matched.append((w, c, v, a))
    L1 = None
    if matched:
        tot = sum(c for _, c, _, _ in matched)
        L1 = (
            sum(c * v for _, c, v, _ in matched) / tot,
            sum(c * a for _, c, _, a in matched) / tot,
        )

    # ── L2：embedding nearest seed ──
    L2 = None
    nearest = []
    try:
        words, vecs = _get_seed_embeddings()
        if words and vecs is not None:
            import numpy as _np
            q = _get_model().encode(text[:_MAX_INPUT_CHARS])
            q = _np.asarray(q, dtype=_np.float32)
            qn = q / (_np.linalg.norm(q) or 1.0)
            sims = vecs @ qn
            top_idx = sims.argsort()[-3:][::-1]
            sv = sa = tot_w = 0.0
            for i in top_idx:
                s = float(sims[i])
                if s < 0.3:
                    continue
                wname = words[i]
                v, a = _EMOTION_SEED_DICT[wname]
                sv += s * v
                sa += s * a
                tot_w += s
                nearest.append({
                    'word': wname, 'sim': round(s, 3),
                    'valence': v, 'arousal': a,
                })
            if tot_w > 0:
                L2 = (sv / tot_w, sa / tot_w)
    except Exception as e:
        logger.warning(f"L2 embedding lookup failed: {e}")

    # ── L3：v2.1 兜底 ──
    L3_dict = estimate_emotion(text)
    L3 = (L3_dict['valence'], L3_dict['arousal'])

    # ── Fusion ──
    if not fuse:
        if L1 is not None:
            v, a, layer = L1[0], L1[1], 'L1'
        elif L2 is not None:
            v, a, layer = L2[0], L2[1], 'L2'
        else:
            v, a, layer = L3[0], L3[1], 'L3'
    else:
        if L1 is not None:
            other = L2 if L2 is not None else L3
            other_tag = 'L2' if L2 is not None else 'L3'
            v = 0.7 * L1[0] + 0.3 * other[0]
            a = 0.7 * L1[1] + 0.3 * other[1]
            layer = f'L1+{other_tag}'
        elif L2 is not None:
            v = 0.7 * L2[0] + 0.3 * L3[0]
            a = 0.7 * L2[1] + 0.3 * L3[1]
            layer = 'L2+L3'
        else:
            v, a, layer = L3[0], L3[1], 'L3'

    v = max(-1.0, min(1.0, v))
    a = max(0.0, min(1.0, a))
    return {
        'valence': round(v, 3),
        'arousal': round(a, 3),
        'layer': layer,
        'matched_words': [w for w, _, _, _ in matched],
        'nearest_seeds': nearest,
        'L1': {'valence': round(L1[0], 3), 'arousal': round(L1[1], 3)} if L1 else None,
        'L2': {'valence': round(L2[0], 3), 'arousal': round(L2[1], 3)} if L2 else None,
        'L3': L3_dict,
    }


# ─── Embedding 生成 ──────────────────────────────────
def _embed(text: str) -> list[float]:
    truncated = text[:_MAX_INPUT_CHARS]
    return _get_model().encode(truncated).tolist()


# ─── 写入（支持完整元数据）────────────────────────────
def generate_and_store(
    key: str,
    content: str,
    tags: list[str] | None = None,
    raw_snippet: str = "",
    emotion: dict | None = None,
    created_at: str | None = None,
) -> bool:
    """为 key 生成 embedding 并落盘，附带元数据。

    如果 emotion=None，用 estimate_emotion 自动估算。
    如果 created_at=None 且 key 没存在过，用 now；已存在的 created_at 保留。
    tags=None 时不变动 tags（或初始化为 []）。
    """
    if not content or not content.strip():
        return False
    try:
        vec = _embed(content)
        if emotion is None:
            emotion = estimate_emotion(content)
        snippet = (raw_snippet or '')[:_RAW_SNIPPET_MAX]

        now = datetime.now().isoformat()
        with sqlite3.connect(_DB_PATH) as conn:
            # 保留已有 created_at
            existing = conn.execute(
                "SELECT created_at FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
            keep_created = existing[0] if (existing and existing[0]) else (created_at or now)

            tags_json = json.dumps(tags if tags is not None else [], ensure_ascii=False)

            conn.execute("""
                INSERT OR REPLACE INTO embeddings
                  (key, vector, updated_at, valence, arousal, tags, raw_snippet,
                   access_count, created_at, pinned)
                VALUES (?, ?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT access_count FROM embeddings WHERE key = ?), 0),
                        ?,
                        COALESCE((SELECT pinned FROM embeddings WHERE key = ?), 0))
            """, (
                key, json.dumps(vec), now,
                emotion.get('valence', 0.0),
                emotion.get('arousal', 0.3),
                tags_json, snippet,
                key, keep_created, key,
            ))
            # 同步 FTS5（hybrid_search 用）。失败静默 —— FTS 不可用时降级到纯 vector。
            try:
                conn.execute(
                    "DELETE FROM embeddings_fts WHERE key = ?", (key,)
                )
                conn.execute(
                    "INSERT INTO embeddings_fts (key, content) VALUES (?, ?)",
                    (key, content[:_MAX_INPUT_CHARS]),
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
        return True
    except Exception as e:
        logger.warning(f"generate_and_store failed for {key}: {e}")
        return False


def update_metadata(
    key: str,
    *,
    tags: list[str] | None = None,
    raw_snippet: str | None = None,
    valence: float | None = None,
    arousal: float | None = None,
) -> bool:
    """更新已有 key 的元数据（不重新生成 embedding）。未指定的字段保持不变。"""
    updates = []
    params = []
    if tags is not None:
        updates.append("tags = ?")
        params.append(json.dumps(tags, ensure_ascii=False))
    if raw_snippet is not None:
        updates.append("raw_snippet = ?")
        params.append(raw_snippet[:_RAW_SNIPPET_MAX])
    if valence is not None:
        updates.append("valence = ?")
        params.append(float(valence))
    if arousal is not None:
        updates.append("arousal = ?")
        params.append(float(arousal))
    if not updates:
        return False
    params.append(key)
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                f"UPDATE embeddings SET {', '.join(updates)} WHERE key = ?", params
            )
            conn.commit()
        return True
    except Exception as e:
        logger.warning(f"update_metadata failed for {key}: {e}")
        return False


def delete_embedding(key: str) -> bool:
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute("DELETE FROM embeddings WHERE key = ?", (key,))
            try:
                conn.execute("DELETE FROM embeddings_fts WHERE key = ?", (key,))
            except sqlite3.OperationalError:
                pass
            conn.commit()
        return True
    except Exception as e:
        logger.warning(f"delete_embedding failed for {key}: {e}")
        return False


def delete_many(keys: list[str]) -> int:
    if not keys:
        return 0
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            params = [(k,) for k in keys]
            conn.executemany("DELETE FROM embeddings WHERE key = ?", params)
            try:
                conn.executemany("DELETE FROM embeddings_fts WHERE key = ?", params)
            except sqlite3.OperationalError:
                pass
            conn.commit()
        return len(keys)
    except Exception as e:
        logger.warning(f"delete_many failed: {e}")
        return 0


# ─── 读取 ────────────────────────────────────────────
def get_metadata(key: str) -> dict | None:
    """返回 key 的完整元数据（不含 vector），None 表示不存在。"""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT key, updated_at, valence, arousal, tags, raw_snippet,
                          access_count, created_at, pinned, soft_pinned
                   FROM embeddings WHERE key = ?""",
                (key,)
            ).fetchone()
            if not row:
                return None
            return _row_to_meta(row)
    except Exception:
        return None


def all_metadata() -> list[dict]:
    """返回所有 key 的元数据。"""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT key, updated_at, valence, arousal, tags, raw_snippet,
                          access_count, created_at, pinned, soft_pinned
                   FROM embeddings"""
            ).fetchall()
            return [_row_to_meta(r) for r in rows]
    except Exception:
        return []


def _row_to_meta(row) -> dict:
    try:
        tags = json.loads(row['tags'] or '[]')
    except (json.JSONDecodeError, TypeError):
        tags = []
    # soft_pinned 列在老 DB 上可能不存在（migration 已加了，但保险一点）
    try:
        soft = bool(row['soft_pinned'])
    except (IndexError, KeyError):
        soft = False
    return {
        'key': row['key'],
        'updated_at': row['updated_at'],
        'created_at': row['created_at'] or row['updated_at'],
        'valence': row['valence'] if row['valence'] is not None else 0.0,
        'arousal': row['arousal'] if row['arousal'] is not None else 0.3,
        'tags': tags,
        'raw_snippet': row['raw_snippet'] or '',
        'access_count': row['access_count'] or 0,
        'pinned': bool(row['pinned']),
        'soft_pinned': soft,
    }


def list_indexed_keys() -> list[str]:
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            return [r[0] for r in conn.execute("SELECT key FROM embeddings")]
    except Exception:
        return []


def keys_by_tag(tag: str) -> list[str]:
    """返回 tags 数组含某个 tag 的所有 key（LIKE 搜 JSON，粗糙但够用）。"""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            # tags 存为 JSON 数组，含引号 "tag"
            needle = f'"{tag}"'
            rows = conn.execute(
                "SELECT key FROM embeddings WHERE tags LIKE ?", (f'%{needle}%',)
            ).fetchall()
            return [r[0] for r in rows]
    except Exception:
        return []


# ─── 访问计数 ────────────────────────────────────────
def increment_access(keys: list[str] | str) -> int:
    """把一个或一组 key 的 access_count +1。返回实际更新行数。"""
    if isinstance(keys, str):
        keys = [keys]
    if not keys:
        return 0
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.executemany(
                "UPDATE embeddings SET access_count = access_count + 1 WHERE key = ?",
                [(k,) for k in keys],
            )
            conn.commit()
            return conn.total_changes
    except Exception as e:
        logger.warning(f"increment_access failed: {e}")
        return 0


# ─── 钉住 ────────────────────────────────────────────
def set_pinned(key: str, pinned: bool = True) -> bool:
    """hard pin = wakeup 必出现。设 pinned=True 时自动清 soft_pinned（互斥）。"""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            if pinned:
                cur = conn.execute(
                    "UPDATE embeddings SET pinned = 1, soft_pinned = 0 WHERE key = ?",
                    (key,),
                )
            else:
                cur = conn.execute(
                    "UPDATE embeddings SET pinned = 0 WHERE key = ?",
                    (key,),
                )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def list_pinned() -> list[str]:
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            return [
                r[0] for r in conn.execute(
                    "SELECT key FROM embeddings WHERE pinned = 1"
                )
            ]
    except Exception:
        return []


def set_soft_pinned(key: str, soft: bool = True) -> bool:
    """soft pin = wakeup 时 70% 概率出现（介于 hard pin 必出现和普通记忆走概率之间）。

    设 soft=True 时自动清掉 hard pinned（一条记忆同时只能是一种钉法），
    设 soft=False 时只清 soft 位，不动 hard。
    """
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            if soft:
                cur = conn.execute(
                    "UPDATE embeddings SET soft_pinned = 1, pinned = 0 WHERE key = ?",
                    (key,),
                )
            else:
                cur = conn.execute(
                    "UPDATE embeddings SET soft_pinned = 0 WHERE key = ?",
                    (key,),
                )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def list_soft_pinned() -> list[str]:
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            return [
                r[0] for r in conn.execute(
                    "SELECT key FROM embeddings WHERE soft_pinned = 1"
                )
            ]
    except Exception:
        return []


# ─── MemScene 数据访问（本地 Dream 融合产物） ─────────────
def insert_scene(
    scene_id: str,
    title: str,
    narrative: str,
    atomic_facts: list[str],
    source_keys: list[str],
    tags: list[str] | None = None,
    foresight: list[dict] | None = None,
    origin: str = 'dream_local',
) -> bool:
    """新增一个记忆场景。foresight 本地版传 [] 即可，LLM 版传 [{content, valid_until}, ...]"""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO mem_scenes
                   (id, title, narrative, atomic_facts, foresight, source_keys, tags, created_at, origin)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scene_id, title, narrative,
                    json.dumps(atomic_facts, ensure_ascii=False),
                    json.dumps(foresight or [], ensure_ascii=False),
                    json.dumps(source_keys, ensure_ascii=False),
                    json.dumps(tags or [], ensure_ascii=False),
                    datetime.now().isoformat(),
                    origin,
                ),
            )
            conn.commit()
        return True
    except Exception as e:
        logger.warning(f"insert_scene failed for {scene_id}: {e}")
        return False


def _scene_row_to_dict(row) -> dict:
    return {
        'id': row['id'],
        'title': row['title'],
        'narrative': row['narrative'],
        'atomic_facts': json.loads(row['atomic_facts'] or '[]'),
        'foresight': json.loads(row['foresight'] or '[]'),
        'source_keys': json.loads(row['source_keys'] or '[]'),
        'tags': json.loads(row['tags'] or '[]'),
        'created_at': row['created_at'],
        'pinned': bool(row['pinned']),
        'origin': row['origin'] or 'unknown',
    }


def list_scenes(limit: int = 20) -> list[dict]:
    """列最近 N 个 scene（按 created_at 倒排）。"""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, title, narrative, atomic_facts, foresight,
                          source_keys, tags, created_at, pinned, origin
                   FROM mem_scenes ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [_scene_row_to_dict(r) for r in rows]
    except Exception:
        return []


def get_scene(scene_id: str) -> dict | None:
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT id, title, narrative, atomic_facts, foresight,
                          source_keys, tags, created_at, pinned, origin
                   FROM mem_scenes WHERE id = ?""",
                (scene_id,),
            ).fetchone()
            return _scene_row_to_dict(row) if row else None
    except Exception:
        return None


def delete_scene(scene_id: str) -> bool:
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            cur = conn.execute("DELETE FROM mem_scenes WHERE id = ?", (scene_id,))
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


# ─── 余弦 + 楼栋过滤 ─────────────────────────────────
def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _match_room(key: str, room: str) -> bool:
    if not room:
        return True
    r = room.rstrip(':')
    return key == r or key.startswith(r + ':')


def search_similar(
    query: str,
    top_k: int = 10,
    room: str = "",
) -> list[tuple[str, float]]:
    try:
        q_vec = _embed(query)
    except Exception as e:
        logger.warning(f"query embed failed: {e}")
        return []
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute("SELECT key, vector FROM embeddings").fetchall()
    except Exception as e:
        logger.warning(f"load embeddings failed: {e}")
        return []

    results: list[tuple[str, float]] = []
    for key, vec_json in rows:
        if not _match_room(key, room):
            continue
        try:
            vec = json.loads(vec_json)
            results.append((key, _cosine(q_vec, vec)))
        except (json.JSONDecodeError, TypeError):
            continue
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ─── FTS5 全文搜索（hybrid_search 的精确匹配层）─────────
def _build_fts_query(q: str) -> str:
    """把自然语言 query 切成 trigram OR 查询。

    为啥不直接 phrase 包 q：trigram tokenizer 要求 phrase 内所有 trigrams 连续命中，
    8 字 query 几乎没人一字不漏写过 → 0 命中。改成切 trigram + OR：任一命中算分，
    BM25 自然给覆盖率高的文档更高分。

    短 query (< 3 字符) 整体当 phrase；≥ 3 字符切 sliding window trigram。
    每个 trigram 用 "..." 包成 phrase（避开 FTS5 运算符 : + - ( ) AND OR NOT）。
    """
    import re
    # 只保留字母数字汉字（剥掉空格、标点、emoji 等会被 FTS 当成 token boundary 的字符）
    cleaned = re.sub(r'[^\w一-鿿]+', '', q)
    if not cleaned:
        return ''
    if len(cleaned) < 3:
        return '"' + cleaned + '"'
    tokens = ['"' + cleaned[i:i + 3] + '"' for i in range(len(cleaned) - 2)]
    return ' OR '.join(tokens)


def fts_search(query: str, top_k: int = 30) -> list[str]:
    """SQLite FTS5 trigram 全文搜索，返回按 BM25 排名的 key 列表。

    跟 search_similar 互补：vector 抓语义相近的，FTS 抓字面命中的。
    cognition.semantic_search 用 RRF 把两者融合（默认 vector 0.7 + FTS 0.3）。
    """
    if not query or not query.strip():
        return []
    fts_q = _build_fts_query(query)
    if not fts_q:
        return []
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute(
                "SELECT key FROM embeddings_fts WHERE embeddings_fts MATCH ? "
                "ORDER BY bm25(embeddings_fts) LIMIT ?",
                (fts_q, top_k),
            ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError as e:
        logger.warning(f"fts_search failed (FTS5 unavailable?): {e}")
        return []


def reindex_fts(items: list[tuple[str, str]]) -> dict:
    """一次性把 (key, content) 对灌进 FTS。用于 backfill 现有数据 / 修复索引。

    幂等：先 DELETE 再 INSERT。content 截断到 _MAX_INPUT_CHARS。
    返回 {indexed, errors}。
    """
    indexed = 0
    errors = 0
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            for key, content in items:
                if not content:
                    continue
                try:
                    conn.execute(
                        "DELETE FROM embeddings_fts WHERE key = ?", (key,)
                    )
                    conn.execute(
                        "INSERT INTO embeddings_fts (key, content) VALUES (?, ?)",
                        (key, content[:_MAX_INPUT_CHARS]),
                    )
                    indexed += 1
                except sqlite3.OperationalError:
                    errors += 1
            conn.commit()
    except sqlite3.OperationalError as e:
        logger.warning(f"reindex_fts batch failed: {e}")
    return {'indexed': indexed, 'errors': errors}


def fts_count() -> int:
    """FTS5 索引里当前条目数（用于 backfill 验证）。"""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            return conn.execute("SELECT COUNT(*) FROM embeddings_fts").fetchone()[0]
    except sqlite3.OperationalError:
        return -1


# ─────────────────────────────────────────────────────────────────
# HippoRAG-style synonym edges (Wave 3 / 2026-06-02)
# 灵感：OSU-NLP-Group/HippoRAG 的 "parahippocampal" synonym layer。
# 设计：pairwise cosine over indexed embeddings。对称（key_a < key_b）。
# 重算时机：手动 / cron，不是每次 query。
# ─────────────────────────────────────────────────────────────────

def _init_synonym_table():
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS synonym_edges (
                key_a TEXT NOT NULL,
                key_b TEXT NOT NULL,
                cosine REAL NOT NULL,
                computed_at TEXT NOT NULL,
                PRIMARY KEY (key_a, key_b)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_syn_a ON synonym_edges(key_a)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_syn_b ON synonym_edges(key_b)")
        conn.commit()


def _norm_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def compute_synonym_edges(threshold: float = 0.75, batch_size: int = 500) -> dict:
    """Recompute synonym_edges from scratch via normalized matmul.

    Wipes synonym_edges and refills with every pair where cosine > threshold.
    Batched matmul keeps peak memory bounded for ~50k keys.
    """
    import numpy as np
    _init_synonym_table()
    with sqlite3.connect(_DB_PATH) as conn:
        rows = conn.execute("SELECT key, vector FROM embeddings").fetchall()
    if len(rows) < 2:
        return {'computed': 0, 'edges': 0, 'threshold': threshold}

    keys, vecs = [], []
    for k, vj in rows:
        try:
            v = json.loads(vj)
            vecs.append(v); keys.append(k)
        except (json.JSONDecodeError, TypeError):
            continue

    M = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Mn = M / norms

    now = datetime.now().isoformat(timespec='seconds')
    n = len(keys)
    found = []
    # Batched against full matrix (symmetric — only upper triangle)
    for i in range(0, n, batch_size):
        block = Mn[i:i+batch_size]                  # (b, d)
        sims = block @ Mn.T                         # (b, n)
        for bi in range(block.shape[0]):
            gi = i + bi
            row = sims[bi]
            # Only j > gi (upper triangle, skip self)
            hits = np.where(row[gi+1:] > threshold)[0]
            for hj in hits:
                gj = gi + 1 + int(hj)
                cos = float(row[gj])
                a, b = _norm_pair(keys[gi], keys[gj])
                found.append((a, b, cos, now))

    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("DELETE FROM synonym_edges")
        conn.executemany(
            "INSERT OR REPLACE INTO synonym_edges (key_a, key_b, cosine, computed_at) "
            "VALUES (?, ?, ?, ?)",
            found,
        )
        conn.commit()
    return {'computed': n, 'edges': len(found), 'threshold': threshold,
            'computed_at': now}


def get_synonym_neighbors(key: str, top_k: int = 10,
                          min_cos: float = 0.75) -> list[tuple[str, float]]:
    _init_synonym_table()
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute("""
                SELECT key_a, key_b, cosine FROM synonym_edges
                WHERE (key_a = ? OR key_b = ?) AND cosine >= ?
                ORDER BY cosine DESC LIMIT ?
            """, (key, key, min_cos, top_k)).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for ka, kb, c in rows:
        other = kb if ka == key else ka
        out.append((other, float(c)))
    return out


# ─── 加权随机打捞（给 wakeup 用）─────────────────────
def sample_weight(meta: dict, now: datetime | None = None) -> float:
    """按 access_count 加权（2026-04-26 改：sublinear log，翻得多的更易再被翻到）。

    公式：weight = 1 + log(1 + access_count)
      - access_count = 0 → weight = 1.0（仍可被采到）
      - access_count = 10 → weight ≈ 3.4
      - access_count = 100 → weight ≈ 5.6（sublinear，避免暴走）

    设计意图（v2）：翻得多的记忆 = 重要记忆，应该更容易再次浮上来。
    （v1 是相反逻辑——惩罚高频访问。被废了。）
    """
    access = meta.get('access_count', 0) or 0
    return 1.0 + math.log1p(access)


def weighted_sample(
    candidates: list[dict],
    k: int,
    exclude_keys: set[str] | None = None,
) -> list[dict]:
    """对候选 meta 列表做加权随机采样，返回 k 个（不重复）。

    candidates: list of meta dict（from all_metadata）
    """
    exclude = exclude_keys or set()
    pool = [m for m in candidates if m['key'] not in exclude]
    if not pool or k <= 0:
        return []
    if len(pool) <= k:
        return pool

    weights = [sample_weight(m) for m in pool]
    # random.choices 允许重复，这里用 while 循环去重
    picked: list[dict] = []
    seen = set()
    safety = 0
    while len(picked) < k and safety < k * 10 and pool:
        choice = random.choices(pool, weights=weights, k=1)[0]
        if choice['key'] not in seen:
            picked.append(choice)
            seen.add(choice['key'])
        safety += 1
    return picked
