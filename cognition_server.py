"""cognition_server.py — MCP server for AI long-term memory + co-reading.

A FastMCP server that exposes ~24 tools:
  - Core KV memory (save / get / list / forget)
  - Hybrid vector + FTS5 search
  - Emotion anchors (set / get / record_emotional)
  - Keyword-trigger recall
  - Paper trail (auto-archive evolving keys)
  - Reading: save chapter / list books / get book / get outline
  - WeRead user-data API (6 tools)
  - JJWXC chapter fetch (2 tools)
  - HippoRAG synonym-edge maintenance
  - Wakeup (curated context dump on session start)

Storage: $MCP_MEMORY_DIR/memories.json + $MCP_MEMORY_DIR/embeddings.db
License: MIT
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("cognition")

# ─── Config ─────────────────────────────────────────────────────────────
MCP_MEMORY_DIR = Path(os.environ.get("MCP_MEMORY_DIR", os.path.expanduser("~/.mcp-memory")))
MCP_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
WEREAD_MODULE_DIR = os.environ.get("WEREAD_MODULE_DIR", str(Path(__file__).parent / "weread"))
COGNITION_PORT = int(os.environ.get("COGNITION_PORT", 8769))

MEMORIES_PATH = MCP_MEMORY_DIR / "memories.json"
WEREAD_STATE_PATH = str(MCP_MEMORY_DIR / "weread_state.json")
BACKUPS_DIR = MCP_MEMORY_DIR / "backups"
BACKUPS_DIR.mkdir(exist_ok=True)

# Sectors that get paper-trail archived on overwrite (generic; customize)
PAPER_TRAIL_PREFIXES = ("procedural:", "todo:", "project:", "case:")
PAPER_TRAIL_MAX_VERSIONS = int(os.environ.get("PAPER_TRAIL_MAX_VERSIONS", 5))
PAPER_TRAIL_MIN_DIFF_CHARS = int(os.environ.get("PAPER_TRAIL_MIN_DIFF_CHARS", 20))

# ─── Bare-key guard ─────────────────────────────────────────────────────
# Reject keys that have no sector prefix (e.g. a raw "2026-06-01T16:48:topic"),
# because they bypass list_by_room / sector navigation and silently rot.
#   ENFORCE_KEY_PREFIX=0          → disable the guard entirely
#   ALLOWED_KEY_PREFIXES="a:,b:"  → strict allow-list (only these prefixes pass)
# With no allow-list set, the guard just rejects no-colon keys and bare
# timestamp keys, which is enough to stop the most common "naked key" leak.
ENFORCE_KEY_PREFIX = os.environ.get("ENFORCE_KEY_PREFIX", "1") != "0"
ALLOWED_KEY_PREFIXES = tuple(
    p.strip() for p in os.environ.get("ALLOWED_KEY_PREFIXES", "").split(",") if p.strip()
)
_BARE_TS_KEY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _validate_key(key: str):
    """Return an error string if `key` lacks a proper sector prefix, else None."""
    k = (key or "").strip()
    if not k:
        return "key must not be empty"
    if k.startswith("_"):  # system keys (_meta:/_system:/_realtime:) are exempt
        return None
    if ALLOWED_KEY_PREFIXES:
        if not any(k.startswith(p) for p in ALLOWED_KEY_PREFIXES):
            return (f"key '{k[:40]}' is not under an allowed sector prefix; "
                    f"allowed: {', '.join(ALLOWED_KEY_PREFIXES)}")
        return None
    if ":" not in k:
        return (f"key '{k[:40]}' has no sector prefix — use 'sector:identifier' "
                f"(e.g. 'episodic:2026-...', 'semantic:...', 'spark:...')")
    if _BARE_TS_KEY_RE.match(k):
        return (f"bare timestamp key '{k[:40]}' — missing sector prefix; prepend a "
                f"sector such as 'episodic:'/'semantic:'/'spark:'/'community:'")
    return None


# ─── Key alias normalization ────────────────────────────────────────────
# Fold alias prefixes onto a canonical one at write time, so the same thing
# stored under drifting names all lands in one sector — e.g. arcadia: vs
# project:桃源:, or a person stored under cc:/栈:/江栈:. Configure via env:
#   KEY_ALIASES="alias1:>canonical1:,alias2:>canonical2:"
# Empty by default (no-op). Applied before the bare-key guard. First match wins.
def _parse_aliases(raw: str):
    pairs = []
    for item in raw.split(","):
        if ">" in item:
            alias, canon = item.split(">", 1)
            alias = alias.strip()
            if alias:
                pairs.append((alias, canon.strip()))
    return tuple(pairs)


KEY_ALIASES = _parse_aliases(os.environ.get("KEY_ALIASES", ""))


def _normalize_key(key: str):
    """Fold an alias prefix onto its canonical form.

    Returns (new_key, matched_alias_or_None). No-op if nothing matches.
    """
    k = (key or "").strip()
    for alias, canon in KEY_ALIASES:
        if k.startswith(alias):
            return canon + k[len(alias):], alias
    return k, None


mcp = FastMCP("cognition", host="127.0.0.1", port=COGNITION_PORT)


# ─── Lazy deps ──────────────────────────────────────────────────────────
_VECTOR_OK = False
_JIEBA_OK = False
try:
    sys.path.insert(0, str(Path(__file__).parent))
    import vector_engine as _vec
    _VECTOR_OK = True
except Exception as e:
    logger.warning(f"vector_engine not loaded: {e}")

try:
    import jieba  # noqa: F401
    _JIEBA_OK = True
except Exception:
    pass

# weread module
_WR_OK = False
try:
    if WEREAD_MODULE_DIR not in sys.path:
        sys.path.insert(0, WEREAD_MODULE_DIR)
    import asyncio as _wr_asyncio
    from weread_fetch import (
        check_state_valid as _wr_check,
        fetch_chapter as _wr_chap,
        fetch_book_toc as _wr_toc,
    )
    from weread_write import (
        add_review as _wr_add_review,
        delete_review as _wr_del_review,
        list_reviews as _wr_list_reviews,
        list_bookmarks as _wr_list_bms,
        list_bookshelf as _wr_list_shelf,
    )
    _WR_OK = True
except Exception as e:
    logger.warning(f"weread module not loaded: {e}")


# ─── Core storage helpers ───────────────────────────────────────────────

def _load_all() -> dict:
    if not MEMORIES_PATH.exists():
        return {}
    try:
        return json.loads(MEMORIES_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"failed to load memories.json: {e}")
        return {}


def _save_all(data: dict) -> None:
    tmp = MEMORIES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(MEMORIES_PATH)


def _now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _save_key(key: str, value: str) -> None:
    """Internal save with paper-trail archival for evolving sectors."""
    data = _load_all()
    old = data.get(key)
    needs_trail = (
        isinstance(old, str)
        and any(key.startswith(p) for p in PAPER_TRAIL_PREFIXES)
        and old != value
        and abs(len(old) - len(value)) >= PAPER_TRAIL_MIN_DIFF_CHARS
    )
    if needs_trail:
        trail_key = f"_meta:trail:{key}"
        existing = data.get(trail_key, "[]")
        try:
            trail = json.loads(existing) if isinstance(existing, str) else []
            if not isinstance(trail, list):
                trail = []
        except Exception:
            trail = []
        trail.append({
            "content": old[:5000],
            "archived_at": _now_ts(),
            "old_len": len(old),
            "new_len": len(value),
        })
        trail = trail[-PAPER_TRAIL_MAX_VERSIONS:]
        data[trail_key] = json.dumps(trail, ensure_ascii=False)
    data[key] = value
    _save_all(data)
    # Update vector index (skip internal _ keys)
    if _VECTOR_OK and not key.startswith("_"):
        try:
            _vec.generate_and_store(key, f"{key}\n\n{value}")
        except Exception:
            pass


def _parse_ts_from_key(key: str) -> float:
    """Try to extract a unix-ms timestamp from a key like 'episodic:2026-06-05T10:30:topic'."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?)", key)
    if not m:
        return 0.0
    try:
        return datetime.fromisoformat(m.group(1)).timestamp() * 1000
    except Exception:
        return 0.0


# ─── Core MCP tools: save / get / list / forget ─────────────────────────

@mcp.tool()
def save_memory(key: str, value: str, force: bool = False) -> dict:
    """Save a memory under `key`. Existing value is overwritten.

    The key is first normalized through KEY_ALIASES (alias prefixes folded onto
    a canonical sector). Then it must carry a sector prefix; bare timestamp keys
    like '2026-06-01T16:48:topic' are rejected so they don't bypass sector
    navigation. Pass force=True to override the guard, or set ENFORCE_KEY_PREFIX=0
    to disable it globally.

    For keys in PAPER_TRAIL_PREFIXES sectors (procedural:/todo:/project:/case:),
    old versions are auto-archived to _meta:trail:<key> on significant change.
    """
    if not key or not key.strip():
        return {"error": "key must not be empty"}
    norm_key, matched_alias = _normalize_key(key)
    if ENFORCE_KEY_PREFIX and not force:
        err = _validate_key(norm_key)
        if err:
            return {"error": err,
                    "hint": "fix the key's sector prefix, or pass force=True to override"}
    _save_key(norm_key, value)
    result = {"saved": norm_key, "length": len(value)}
    if matched_alias:
        result["normalized_from"] = key.strip()
        result["note"] = f"alias '{matched_alias}' folded onto canonical prefix"
    return result


@mcp.tool()
def get_memory(key: str) -> dict:
    """Read a memory by key. Returns {key, value, length} or error if missing."""
    data = _load_all()
    if key not in data:
        return {"error": f"key not found: {key}"}
    val = data[key]
    return {"key": key, "value": val, "length": len(val) if isinstance(val, str) else 0}


@mcp.tool()
def list_keys(prefix: str = "", limit: int = 200) -> dict:
    """List keys, optionally filtered by prefix. Returns {count, keys: [...]}."""
    data = _load_all()
    keys = [k for k in data.keys() if k.startswith(prefix)]
    keys.sort()
    return {"count": len(keys), "prefix": prefix, "keys": keys[:limit], "truncated": len(keys) > limit}


@mcp.tool()
def forget_keys(keys: list[str], dry_run: bool = True) -> dict:
    """Bulk-delete keys. Defaults to dry_run=True for safety.

    Returns {would_delete OR deleted, missing, total}.
    """
    data = _load_all()
    existing = [k for k in keys if k in data]
    missing = [k for k in keys if k not in data]
    if dry_run:
        return {"would_delete": existing, "missing": missing, "total": len(existing), "dry_run": True}
    for k in existing:
        del data[k]
    _save_all(data)
    if _VECTOR_OK:
        for k in existing:
            try:
                _vec.delete_embedding(k)
            except Exception:
                pass
    return {"deleted": existing, "missing": missing, "total": len(existing), "dry_run": False}


# ─── Search ─────────────────────────────────────────────────────────────

def _rrf_fuse(rankings: list[list[str]], weights: list[float], rrf_k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack 2009). Each ranking is a list of keys."""
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@mcp.tool()
def search_memories(
    query: str,
    top_k: int = 10,
    room: str = "",
    fts_weight: float = 0.3,
    rrf_k: int = 60,
    use_ppr: bool = False,
    extract_entities: bool = False,
    ppr_alpha: float = 0.85,
) -> dict:
    """Hybrid memory search: dense vector + SQLite FTS5, RRF-fused.

    Args:
      query: natural-language query
      top_k: how many results to return
      room: prefix filter (e.g. "reading:" → only search reading sector)
      fts_weight: 0.0-1.0; 0 = pure vector (default 0.3 = vector 0.7 + FTS 0.3)
      rrf_k: RRF smoothing constant (Cormack 2009 recommends 60)
      use_ppr: enable HippoRAG Personalized PageRank over synonym + Hebbian edges
      extract_entities: use jieba to extract noun phrases, expand seed set
      ppr_alpha: PPR damping factor (0.85 typical)
    """
    if not _VECTOR_OK:
        return {"error": "vector_engine not loaded", "results": []}

    internal_k = max(top_k * 4, 30)
    v_hits = _vec.search_similar(query, top_k=internal_k, room=room)
    f_keys: list[str] = []
    if fts_weight > 0 and hasattr(_vec, "fts_search"):
        f_raw = _vec.fts_search(query, top_k=internal_k)
        f_keys = [k for k in f_raw if _vec._match_room(k, room)]

    if fts_weight > 0 and f_keys:
        v_keys = [k for k, _ in v_hits]
        fused = _rrf_fuse([v_keys, f_keys], [1.0 - fts_weight, fts_weight], rrf_k=rrf_k)
        mode = f"hybrid_v{1.0-fts_weight:.1f}_f{fts_weight:.1f}"
    else:
        fused = list(v_hits)
        mode = "vector_only" if fts_weight == 0 else "vector_fallback_fts_empty"

    # PPR + entity expansion (optional)
    ppr_meta: dict = {"used": False, "entities": [], "subgraph_nodes": 0}
    if use_ppr or extract_entities:
        seed_scores: dict[str, float] = {}
        for k, s in fused[:max(top_k * 2, 20)]:
            seed_scores[k] = max(seed_scores.get(k, 0.0), float(s))

        if extract_entities and _JIEBA_OK:
            import jieba.posseg as pseg
            ents = []
            for w, flag in pseg.cut(query):
                w = w.strip()
                if len(w) >= 2 and flag.startswith(("n", "v")):
                    ents.append(w)
            ents = list(dict.fromkeys(ents))[:5]
            ppr_meta["entities"] = ents
            for ent in ents:
                try:
                    for k, s in _vec.search_similar(ent, top_k=5, room=room):
                        seed_scores[k] = max(seed_scores.get(k, 0.0), float(s) * 0.7)
                except Exception:
                    continue

        if use_ppr and seed_scores:
            try:
                import networkx as nx
                G, nodes = _build_ppr_subgraph(set(seed_scores.keys()), max_hops=2)
                ppr_meta["subgraph_nodes"] = len(nodes)
                if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
                    total = sum(seed_scores.values()) or 1.0
                    personalization = {n: seed_scores.get(n, 0.0) / total for n in G.nodes()}
                    if sum(personalization.values()) == 0:
                        for k in seed_scores:
                            if k in personalization:
                                personalization[k] = 1.0
                        s_total = sum(personalization.values()) or 1.0
                        personalization = {n: v / s_total for n, v in personalization.items()}
                    ppr_scores = nx.pagerank(G, alpha=ppr_alpha, personalization=personalization, max_iter=100, tol=1e-6)
                    fused = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
                    mode = f"ppr_a{ppr_alpha}"
                    ppr_meta["used"] = True
            except Exception as e:
                ppr_meta["error"] = f"{type(e).__name__}: {e}"
        elif extract_entities and seed_scores:
            fused = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)
            mode = f"{mode}+entities"

    data = _load_all()
    fused = [(k, s) for k, s in fused if k in data][:top_k]

    meta_map = {m["key"]: m for m in _vec.all_metadata()} if _VECTOR_OK else {}
    results = []
    for key, score in fused:
        content = data.get(key, "")
        m = meta_map.get(key, {})
        preview = content[:200] + ("…" if len(content) > 200 else "")
        results.append({
            "key": key,
            "score": round(score, 4),
            "preview": preview,
            "tags": m.get("tags", []),
            "access_count": m.get("access_count", 0),
        })

    if fused and _VECTOR_OK:
        try:
            _vec.increment_access([k for k, _ in fused])
        except Exception:
            pass

    return {
        "query": query,
        "mode": mode,
        "results": results,
        "indexed_total": len(meta_map),
        "ppr": ppr_meta,
    }


def _build_ppr_subgraph(seed_keys: set[str], max_hops: int = 2,
                        min_hebb_weight: float = 0.15, min_syn_cos: float = 0.75):
    """BFS out from seeds over Hebbian + synonym edges, build networkx Graph."""
    import sqlite3
    import networkx as nx
    if not _VECTOR_OK or not seed_keys:
        return nx.Graph(), set()
    try:
        _vec._init_synonym_table()
    except Exception:
        pass

    nodes = set(seed_keys)
    frontier = set(seed_keys)
    edges_acc: dict[tuple[str, str], float] = {}

    with sqlite3.connect(_vec._DB_PATH) as conn:
        for _ in range(max_hops):
            if not frontier:
                break
            ph = ",".join("?" * len(frontier))
            params = list(frontier) * 2
            try:
                for ka, kb, w in conn.execute(
                    f"SELECT key_a, key_b, weight FROM edges "
                    f"WHERE (key_a IN ({ph}) OR key_b IN ({ph})) AND weight >= ?",
                    params + [min_hebb_weight],
                ):
                    a, b = (ka, kb) if ka < kb else (kb, ka)
                    edges_acc[(a, b)] = max(edges_acc.get((a, b), 0.0), float(w))
            except sqlite3.OperationalError:
                pass
            try:
                for ka, kb, c in conn.execute(
                    f"SELECT key_a, key_b, cosine FROM synonym_edges "
                    f"WHERE (key_a IN ({ph}) OR key_b IN ({ph})) AND cosine >= ?",
                    params + [min_syn_cos],
                ):
                    a, b = (ka, kb) if ka < kb else (kb, ka)
                    edges_acc[(a, b)] = max(edges_acc.get((a, b), 0.0), float(c))
            except sqlite3.OperationalError:
                pass
            new_nodes = {n for pair in edges_acc for n in pair} - nodes
            nodes.update(new_nodes)
            frontier = new_nodes

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (a, b), w in edges_acc.items():
        G.add_edge(a, b, weight=w)
    return G, nodes


# ─── Emotion ────────────────────────────────────────────────────────────

@mcp.tool()
def set_emotion_current(state: str, trigger: str = "", sensation: str = "", duration_est: str = "") -> dict:
    """Overwrite emotion:current — the short-term emotion anchor read by wakeup.

    state: how you feel right now (free-form)
    trigger: what just happened to cause it (event / scene hook)
    sensation: bodily sensation if any ("racing heart", "tight chest")
    duration_est: rough duration ("just now" / "an hour" / "all day")
    """
    ts = _now_ts()
    parts = [f"## now ({ts})", state.strip()]
    if trigger.strip():
        parts.extend(["", "## trigger", trigger.strip()])
    if sensation.strip():
        parts.extend(["", "## sensation", sensation.strip()])
    if duration_est.strip():
        parts.extend(["", "## duration", duration_est.strip()])
    body = "\n".join(parts)
    _save_key("emotion:current", body)
    _save_key(f"emotion:{ts}", body)
    return {"current_key": "emotion:current", "snapshot_key": f"emotion:{ts}"}


@mcp.tool()
def get_emotion_current(include_age_hint: bool = True) -> str:
    """Read the last-saved emotion anchor. Returns text with optional age hint."""
    data = _load_all()
    raw = data.get("emotion:current", "")
    if not raw or not include_age_hint:
        return raw
    m = re.search(r"\((\d{4}-\d{2}-\d{2}T\d{2}:\d{2})\)", raw[:200])
    if not m:
        return raw
    try:
        recorded = datetime.fromisoformat(m.group(1))
        delta = datetime.now() - recorded
        hours = int(delta.total_seconds() / 3600)
        if hours < 1:
            hint = "[age] less than an hour old"
        elif hours < 6:
            hint = f"[age] {hours}h old, probably still relevant"
        elif hours < 24:
            hint = f"[age] {hours}h old, may have faded"
        else:
            hint = f"[age] {hours // 24}d old — historical snapshot, not current state"
        return f"{hint}\n\n{raw}"
    except (ValueError, TypeError):
        return raw


@mcp.tool()
def record_emotional(content: str, tags: list[str] | None = None) -> dict:
    """Write a single emotional-memory entry to emotional:<ts>."""
    ts = _now_ts()
    key = f"emotional:{ts}"
    body = content
    if tags:
        body = f"[tags: {', '.join(tags)}]\n\n{content}"
    _save_key(key, body)
    return {"saved": key, "tags": tags or []}


# ─── Trigger words ──────────────────────────────────────────────────────

@mcp.tool()
def trigger_check(text: str, top_k: int = 5) -> dict:
    """Scan text for trigger words; for each fire, return associated recent memories.

    Trigger words are stored in vector_engine's trigger_words SQLite table.
    A fire decrements the word's weight (decay over time) and surfaces memories
    that share embeddings with the trigger word.
    """
    if not _VECTOR_OK:
        return {"error": "vector_engine not loaded", "fires": []}
    fires = []
    try:
        import sqlite3
        with sqlite3.connect(_vec._DB_PATH) as conn:
            try:
                rows = conn.execute(
                    "SELECT word, weight, total_fires FROM trigger_words "
                    "WHERE weight > 0.1 ORDER BY weight DESC"
                ).fetchall()
            except sqlite3.OperationalError:
                return {"fires": [], "note": "trigger_words table not initialized"}

            for word, weight, total_fires in rows:
                if word in text:
                    fires.append({"word": word, "weight": round(weight, 3),
                                  "total_fires": total_fires})
                    # Decay
                    new_w = max(0.05, weight * 0.9)
                    conn.execute(
                        "UPDATE trigger_words SET weight = ?, last_fired_at = ?, "
                        "total_fires = total_fires + 1 WHERE word = ?",
                        (new_w, _now_ts(), word),
                    )
            conn.commit()
    except Exception as e:
        return {"error": str(e), "fires": []}

    # For each fire, also try a semantic recall
    recalls = []
    for f in fires[:top_k]:
        try:
            hits = _vec.search_similar(f["word"], top_k=3)
            for k, s in hits:
                recalls.append({"trigger": f["word"], "key": k, "score": round(s, 3)})
        except Exception:
            continue

    return {"fires": fires, "recalls": recalls[:top_k * 2]}


# ─── Paper trail ────────────────────────────────────────────────────────

@mcp.tool()
def get_paper_trail(key: str) -> dict:
    """Query archived versions of a key (auto-saved by _save_key on overwrite)."""
    data = _load_all()
    trail_key = f"_meta:trail:{key}"
    raw = data.get(trail_key)
    if raw is None:
        return {"key": key, "has_trail": False, "version_count": 0, "versions": []}
    try:
        trail = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
    except Exception:
        trail = []
    return {"key": key, "has_trail": True, "version_count": len(trail), "versions": trail}


# ─── HippoRAG ───────────────────────────────────────────────────────────

@mcp.tool()
def compute_synonym_edges_now(threshold: float = 0.75) -> dict:
    """Recompute HippoRAG-style synonym edges (pairwise cosine > threshold).

    One-shot — run manually or via cron. Wipes synonym_edges table and rebuilds.
    Cost: O(n^2 * d) matmul; ~50k keys × 384d takes ~10s on a typical VPS.
    Recommended: run weekly, or after ~1k new memories accumulate.
    """
    if not _VECTOR_OK:
        return {"error": "vector_engine not loaded"}
    if not hasattr(_vec, "compute_synonym_edges"):
        return {"error": "vector_engine missing compute_synonym_edges (upgrade and restart)"}
    return _vec.compute_synonym_edges(threshold=threshold)


# ─── Reading: generic ──────────────────────────────────────────────────

def _reading_book_key(book: str, suffix: str) -> str:
    safe_book = re.sub(r"[^\w一-鿿\-]", "_", book).strip("_")
    return f"reading:{safe_book}:{suffix}"


@mcp.tool()
def reading_save_chapter(book: str, chapter: str, content: str) -> dict:
    """Save a manually-pasted chapter body. Use this when auto-fetchers don't work
    (paywall, obfuscation, etc.). The content is stored verbatim under
    reading:<book>:ch:<chapter>."""
    chap_clean = re.sub(r"\s+", "_", chapter).strip("_")
    chap_key = _reading_book_key(book, f"ch:{chap_clean}")
    header = f"## {book} · {chap_clean}\n\n"
    _save_key(chap_key, header + content)
    progress_key = _reading_book_key(book, "progress")
    _save_key(progress_key, f"saved {chap_clean} @ {_now_ts()}")
    return {"saved": chap_key, "chars": len(content)}


@mcp.tool()
def reading_list_books() -> dict:
    """List all books with saved chapters/progress."""
    data = _load_all()
    books: dict[str, dict] = {}
    for k in data:
        if not k.startswith("reading:"):
            continue
        parts = k.split(":")
        if len(parts) < 3:
            continue
        book = parts[1] if not k.startswith("reading:book:") else (parts[3] if len(parts) > 3 else "")
        if not book:
            continue
        if book not in books:
            books[book] = {"book": book, "chapter_count": 0, "has_progress": False}
        if ":ch:" in k:
            books[book]["chapter_count"] += 1
        if k.endswith(":progress"):
            books[book]["has_progress"] = True
    return {"books": list(books.values()), "total": len(books)}


@mcp.tool()
def reading_get_book(book: str) -> dict:
    """List chapters + progress for one book."""
    data = _load_all()
    safe = re.sub(r"[^\w一-鿿\-]", "_", book).strip("_")
    chap_keys = sorted(k for k in data if k.startswith(f"reading:{safe}:ch:"))
    progress = data.get(f"reading:{safe}:progress", "")
    return {
        "book": book,
        "chapters": chap_keys,
        "chapter_count": len(chap_keys),
        "progress": progress,
    }


# ─── Reading: outline (rule-based, 50x compression) ────────────────────

_OUTLINE_FIRST_SENT_RE = r"^([^。！？\.\!\?\n]{4,%d}[。！？\.\!\?])"
_OUTLINE_CHAP_NUM_RE = re.compile(r"第\s*(\d+)\s*章")


def _outline_first_sentence(p: str, max_chars: int = 80) -> str:
    m = re.match(_OUTLINE_FIRST_SENT_RE % max_chars, p)
    if m:
        return m.group(1).strip()
    snippet = p[:max_chars].strip()
    return snippet + ("…" if len(p) > max_chars else "")


def _outline_is_junk(p: str) -> bool:
    if len(p) < 8:
        return True
    if p.startswith(("##", "[", "【", "<", "function ", "var ", "/*", "//")):
        return True
    if re.search(r"\b(type|data|url|method|async|var|let|const)\s*:", p):
        return True
    if "log_data" in p or "getElementById" in p or ".ajax" in p:
        return True
    code_chars = sum(p.count(c) for c in "{}=();")
    if code_chars / max(len(p), 1) > 0.05:
        return True
    if len(p) > 12:
        cn_chars = sum(1 for c in p if "一" <= c <= "鿿")
        if cn_chars / len(p) < 0.3:
            return True
    return False


@mcp.tool()
def reading_get_outline(key: str) -> dict:
    """Rule-based chapter outline. ~50x compression vs raw text. No LLM call.

    Output structure:
      - title
      - total_chars / paragraph_count
      - opening_sentences (first sentence of first 3 paras)
      - middle_sentences (first sentence of ~6 evenly-spaced middle paras)
      - closing_sentences (first sentence of last 2 paras)
      - main_entities (top 5 by jieba.posseg nr/nrt/nz/nt)
      - dialogue_density_per_1k_chars
      - head_excerpt / tail_excerpt (raw 200 chars)
      - compression_ratio
    """
    data = _load_all()
    raw = data.get(key, "")
    if not raw:
        return {"error": f"key not found or empty: {key}"}

    text = ""
    title = ""
    if isinstance(raw, str) and raw.lstrip().startswith("{"):
        try:
            obj = json.loads(raw)
            text = obj.get("text", "") or obj.get("content", "") or ""
            title = obj.get("title", "") or obj.get("chapter_title", "") or ""
        except Exception:
            text = raw
    else:
        text = raw if isinstance(raw, str) else str(raw)

    if not text or len(text.strip()) < 20:
        return {"error": "content empty or too short", "key": key, "total_chars": len(text)}

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 20 or any(len(p) > 1500 for p in paragraphs):
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    if not title:
        for p in paragraphs[:3]:
            m = re.match(r"^##\s*(.+?)(?:\s*[·•]\s*(.+))?$", p)
            if m:
                title = m.group(0).lstrip("# ").strip()
                break

    paragraphs = [p for p in paragraphs if not _outline_is_junk(p)]
    n_paras = len(paragraphs)

    if not title and paragraphs and len(paragraphs[0]) < 30:
        title = paragraphs[0]

    opening = [_outline_first_sentence(p) for p in paragraphs[:3]]
    closing = [_outline_first_sentence(p) for p in paragraphs[-2:]] if n_paras > 5 else []

    middle = []
    if n_paras > 5:
        candidates = list(range(3, n_paras - 2))
        if candidates:
            sample_n = min(6, len(candidates))
            step = max(1, len(candidates) // sample_n)
            for i in candidates[::step][:sample_n]:
                middle.append(_outline_first_sentence(paragraphs[i]))

    entities = []
    if _JIEBA_OK:
        try:
            import jieba.posseg as pseg
            from collections import Counter
            freq: Counter = Counter()
            for w, flag in pseg.cut(text[:30000]):
                w = w.strip()
                if len(w) >= 2 and flag in ("nr", "nrt", "nz", "nt"):
                    freq[w] += 1
            entities = [{"name": n, "count": c} for n, c in freq.most_common(5)]
        except Exception:
            pass

    quote_count = sum(text.count(q) for q in ('"', '"', "「", "『", '"'))
    speak_count = sum(text.count(s) for s in ("说道", "说：", "回答", "问道", "道：", "叫道", "笑道", "答道"))
    total_chars = len(text)
    dialogue_density = round((quote_count + speak_count) / max(total_chars / 1000, 1), 1)

    summary_chars = sum(len(s) for s in opening + middle + closing) + 400
    compression_ratio = round(total_chars / max(summary_chars, 1), 1)

    return {
        "key": key,
        "title": title or "(untitled)",
        "total_chars": total_chars,
        "paragraph_count": n_paras,
        "opening_sentences": opening,
        "middle_sentences": middle,
        "closing_sentences": closing,
        "main_entities": entities,
        "dialogue_density_per_1k_chars": dialogue_density,
        "head_excerpt": text[:200],
        "tail_excerpt": text[-200:] if total_chars > 400 else "",
        "compression_ratio": compression_ratio,
    }


# ─── WeRead wrappers ────────────────────────────────────────────────────

def _wr_run(coro):
    """Run async coro from sync MCP tool context (thread + new loop)."""
    if not _WR_OK:
        return {"error": "weread module not loaded"}
    try:
        loop = _wr_asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                return ex.submit(lambda: _wr_asyncio.run(coro)).result()
    except RuntimeError:
        pass
    return _wr_asyncio.run(coro)


@mcp.tool()
def reading_weread_list_bookshelf() -> str:
    """List the user's entire weread bookshelf (books + reading progress)."""
    if not _WR_OK:
        return json.dumps({"error": "weread module not loaded"}, ensure_ascii=False)
    data = _wr_run(_wr_list_shelf(WEREAD_STATE_PATH))
    books = data.get("books", []) if isinstance(data, dict) else []
    out = {
        "total": len(books),
        "books": [
            {
                "bookId": b.get("bookId"),
                "title": b.get("title"),
                "author": b.get("author"),
                "finishReading": b.get("finishReading"),
                "category": b.get("category"),
            }
            for b in books[:100]
        ],
    }
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
def reading_weread_fetch_toc(book_id: str) -> str:
    """Fetch the table of contents (chapter list) for a weread book.

    WeRead's chapter list is not exposed via plain API — this tool opens
    the reader page, clicks the TOC sidebar, and scrapes the DOM (~8-12s).

    Saves the result to reading:book:weread:<book_id>:toc.

    book_id may be raw numeric or already URL-encoded.
    """
    if not _WR_OK:
        return json.dumps({"error": "weread module not loaded"}, ensure_ascii=False)
    data = _wr_run(_wr_toc(WEREAD_STATE_PATH, book_id))
    key = f"reading:book:weread:{book_id}:toc"
    _save_key(key, json.dumps(data, ensure_ascii=False))
    chapters = data.get("chapters", []) if isinstance(data, dict) else []
    return json.dumps({
        "saved_to": key,
        "title": data.get("title") if isinstance(data, dict) else None,
        "author": data.get("author") if isinstance(data, dict) else None,
        "chapter_count": len(chapters),
        "chapters": chapters[:50],
        "source": data.get("source") if isinstance(data, dict) else None,
        "note": data.get("note") if isinstance(data, dict) else None,
    }, ensure_ascii=False)


@mcp.tool()
def reading_weread_fetch_chapter(book_id: str, chapter_uid: str = "", max_pages: int = 200) -> str:
    """Fetch a full chapter from weread (auto-spans multiple "sections").

    chapter_uid="" → auto-resume from user's last reading position.
    Saves the chapter body to reading:book:weread:<book_id>:ch:<uid_or_current>.

    book_id may be raw numeric (auto-resolved via /web/book/info) or already
    URL-encoded.
    """
    if not _WR_OK:
        return json.dumps({"error": "weread module not loaded"}, ensure_ascii=False)
    data = _wr_run(_wr_chap(WEREAD_STATE_PATH, book_id, chapter_uid, max_pages))
    key = f"reading:book:weread:{book_id}:ch:{chapter_uid or 'current'}"
    _save_key(key, json.dumps(data, ensure_ascii=False))
    return json.dumps({
        "saved_to": key,
        "title": data.get("title"),
        "text_len": len(data.get("text", "")),
        "section_count": data.get("section_count"),
        "section_titles": data.get("section_titles", []),
        "book_ended": data.get("book_ended"),
        "preview": data.get("text", "")[:200],
    }, ensure_ascii=False)


@mcp.tool()
def reading_weread_list_notes(book_id: str, mine: bool = True) -> str:
    """List your notes (reviews + paragraph comments) on a weread book."""
    if not _WR_OK:
        return json.dumps({"error": "weread module not loaded"}, ensure_ascii=False)
    data = _wr_run(_wr_list_reviews(WEREAD_STATE_PATH, book_id, mine=mine))
    reviews = data.get("reviews", []) if isinstance(data, dict) else []
    out = {
        "total": data.get("totalCount", len(reviews)) if isinstance(data, dict) else 0,
        "reviews": [
            {
                "reviewId": (r.get("review", {}) or r).get("reviewId"),
                "content": (r.get("review", {}) or r).get("content"),
                "chapterUid": (r.get("review", {}) or r).get("chapterUid"),
                "createTime": (r.get("review", {}) or r).get("createTime"),
                "range": (r.get("review", {}) or r).get("range"),
            }
            for r in reviews[:50]
        ],
    }
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
def reading_weread_list_highlights(book_id: str) -> str:
    """List your highlights on a weread book (read-only)."""
    if not _WR_OK:
        return json.dumps({"error": "weread module not loaded"}, ensure_ascii=False)
    data = _wr_run(_wr_list_bms(WEREAD_STATE_PATH, book_id))
    return json.dumps(data, ensure_ascii=False)


@mcp.tool()
def reading_weread_add_note(book_id: str, chapter_uid: int, content: str,
                            range_str: str = "", is_private: bool = False) -> str:
    """Add a note/review to a weread chapter. The user sees it in their app's
    "My Notes" section. This is the core of the co-reading loop.

    range_str="" → chapter-level note. range_str="1234-1256" → anchored to a span.
    """
    if not _WR_OK:
        return json.dumps({"error": "weread module not loaded"}, ensure_ascii=False)
    data = _wr_run(_wr_add_review(WEREAD_STATE_PATH, book_id, chapter_uid, content,
                                   range_str if range_str else None, is_private))
    return json.dumps(data, ensure_ascii=False)


@mcp.tool()
def reading_weread_delete_note(review_id: str) -> str:
    """Delete a previously-added note. review_id from add_note response or list_notes."""
    if not _WR_OK:
        return json.dumps({"error": "weread module not loaded"}, ensure_ascii=False)
    data = _wr_run(_wr_del_review(WEREAD_STATE_PATH, review_id))
    return json.dumps(data, ensure_ascii=False)


# ─── JJWXC ──────────────────────────────────────────────────────────────

_JJWXC_COOKIES_KEY = "_system:jjwxc_cookies"


def _jjwxc_load_cookies() -> list:
    data = _load_all()
    raw = data.get(_JJWXC_COOKIES_KEY, "[]")
    try:
        return json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
    except Exception:
        return []


@mcp.tool()
def reading_jjwxc_install_cookies(cookies_json: str) -> dict:
    """Install jjwxc cookies (one-time setup).

    Format (paste from devtools as JSON array):
      [{"name": "...", "value": "...", "domain": ".jjwxc.net", "path": "/"}, ...]
    """
    try:
        cookies = json.loads(cookies_json)
        if not isinstance(cookies, list):
            return {"error": "expected JSON array"}
    except Exception as e:
        return {"error": f"invalid JSON: {e}"}
    cleaned = []
    for c in cookies:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        cleaned.append({
            "name": c["name"],
            "value": c.get("value", ""),
            "domain": c.get("domain") or ".jjwxc.net",
            "path": c.get("path") or "/",
        })
    _save_key(_JJWXC_COOKIES_KEY, json.dumps(cleaned, ensure_ascii=False))
    return {"installed": len(cleaned), "note": "test with reading_jjwxc_fetch_chapter"}


def _jjwxc_parse_toc(html: str) -> dict:
    """Extract TOC from a jjwxc onebook.php page.

    jjwxc's old-school table layout is malformed nested HTML; bs4's html.parser
    only catches the first ~25 chapters. We regex chapterid values directly.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        return {"error": f"missing dep: {e}"}

    book_title = None
    try:
        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.select_one('h1.tit, h1, span[itemprop="name"]')
        if title_el:
            book_title = title_el.get_text(strip=True)
    except Exception:
        m_title = re.search(r"<h1[^>]*>([^<]+)</h1>", html)
        if m_title:
            book_title = m_title.group(1).strip()

    raw_ids = re.findall(r'chapterid=?["\']?(\d+)', html)
    unique_ids = sorted(set(raw_ids), key=int)

    titles_by_id: dict[str, str] = {}
    for m in re.finditer(r'chapterid=?["\']?(\d+)["\']?[^>]*>([^<]{1,120})<', html):
        cid, txt = m.group(1), m.group(2).strip()
        if not txt or cid in titles_by_id:
            continue
        if re.search(r"第\s*\d+\s*章", txt) or len(txt) <= 30:
            titles_by_id[cid] = txt

    chapters = []
    for cid in unique_ids:
        title = titles_by_id.get(cid) or f"第{cid}章"
        chapters.append({"chapter_id": cid, "title": title})

    return {"book_title": book_title, "chapter_count": len(chapters), "chapters": chapters}


@mcp.tool()
def reading_jjwxc_fetch_toc(novel_id: str) -> dict:
    """Fetch the chapter list (TOC) for a jjwxc novel.

    URL pattern: http://www.jjwxc.net/onebook.php?novelid=<N>
    novel_id is visible in the book's URL.

    Returns list of {chapter_id, title}. **Cookies not required** —
    chapter listings are public.
    """
    if not str(novel_id).strip().isdigit():
        return {"error": "novel_id must be a numeric string"}
    try:
        import httpx
    except ImportError as e:
        return {"error": f"missing dep: {e}"}
    url = f"http://www.jjwxc.net/onebook.php?novelid={novel_id}"
    try:
        with httpx.Client(timeout=20.0, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        }) as c:
            r = c.get(url)
            r.encoding = "gb18030"
            html = r.text
    except Exception as e:
        return {"error": f"fetch failed: {e}"}
    parsed = _jjwxc_parse_toc(html)
    parsed["novel_id"] = novel_id
    parsed["source_url"] = url
    return parsed


@mcp.tool()
def reading_jjwxc_fetch_chapter(novel_id: str, chapter_idx: int, save_to_book: str = "") -> dict:
    """Fetch one chapter from jjwxc.net. Saves under reading:<book>:ch:<idx>.

    novel_id: numeric novel id from jjwxc URL
    chapter_idx: integer chapter index
    save_to_book: friendly book name for storage key (defaults to novel_id)
    """
    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError as e:
        return {"error": f"missing dep: {e}"}

    cookies_list = _jjwxc_load_cookies()
    cookies = {c["name"]: c["value"] for c in cookies_list}
    url = f"http://www.jjwxc.net/onebook.php?novelid={novel_id}&chapterid={chapter_idx}"
    try:
        with httpx.Client(cookies=cookies, timeout=20.0, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        }) as c:
            r = c.get(url)
            r.encoding = "gb18030"  # jjwxc uses GBK family
            html = r.text
    except Exception as e:
        return {"error": f"fetch failed: {e}"}

    soup = BeautifulSoup(html, "lxml")
    # jjwxc chapter container — common selectors
    body_div = soup.select_one("div.noveltext") or soup.select_one("#oneboolt")
    if not body_div:
        return {"error": "could not find chapter body — selector may have changed",
                "url": url, "html_head": html[:300]}
    # Strip script/style
    for tag in body_div.find_all(["script", "style"]):
        tag.decompose()
    text = body_div.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    book = save_to_book or f"jjwxc_{novel_id}"
    chap_key = _reading_book_key(book, f"ch:{chapter_idx}")
    header = f"## {book} · ch{chapter_idx}\n[source] jjwxc novel={novel_id}\n\n"
    _save_key(chap_key, header + text)
    return {
        "saved_to": chap_key,
        "novel_id": novel_id,
        "chapter_idx": chapter_idx,
        "text_len": len(text),
        "preview": text[:300],
    }


# ─── Wakeup ─────────────────────────────────────────────────────────────

@mcp.tool()
def wakeup(recent_days: int = 3, max_recent: int = 5, random_count: int = 2,
           random_age_min_days: int = 7, mode: str = "lite", preview_chars: int = 100) -> dict:
    """Session-startup context dump.

    Returns:
      welcome: opening line with current time + gap since last activity
      emotion_current: the emotion anchor (full text)
      emotion_current_age_hint: how old that anchor is
      recent_emotional: last N days of emotional:* keys
      random_old: weighted-random old memories (by access_count)
      hot_topics_today: trigger words fired in last 24h
      meta: misc stats
    """
    if mode not in ("lite", "full"):
        return {"error": f"mode must be 'lite' or 'full', got {mode!r}"}
    all_data = _load_all()
    now_ms = time.time() * 1000

    # Welcome line
    now = datetime.now()
    welcome = f"current time: {now.isoformat(timespec='minutes')}"

    # Emotion current + age hint
    emotion_current = all_data.get("emotion:current", "")
    emotion_age_hint = None
    if emotion_current:
        m = re.search(r"## now \((\d{4}-\d{2}-\d{2}T\d{2}:\d{2})\)", emotion_current)
        if m:
            try:
                rec = datetime.fromisoformat(m.group(1))
                hours = (now - rec).total_seconds() / 3600
                if hours < 1:
                    emotion_age_hint = "just now (< 1h)"
                elif hours < 6:
                    emotion_age_hint = f"{int(hours)}h ago — likely still relevant"
                elif hours < 24:
                    emotion_age_hint = f"{int(hours)}h ago — may have faded"
                else:
                    emotion_age_hint = f"{int(hours / 24)}d ago — historical, not current"
            except Exception:
                pass

    # Recent emotional memories
    recent_cutoff = now_ms - recent_days * 86400000
    recent_keys = []
    for k, v in all_data.items():
        if not k.startswith(("emotional:", "emotion:", "reward:")):
            continue
        if k == "emotion:current":
            continue
        ts = _parse_ts_from_key(k)
        if ts >= recent_cutoff:
            recent_keys.append((ts, k, v))
    recent_keys.sort(reverse=True)
    recent_out = []
    for ts, k, v in recent_keys[:max_recent]:
        prev = (v[:preview_chars] + "…") if len(v) > preview_chars else v
        recent_out.append({"key": k, "preview": prev, "length": len(v)})

    # Random old by weighted access
    random_out = []
    if _VECTOR_OK:
        try:
            metas = _vec.all_metadata()
            cand = []
            for m in metas:
                k = m["key"]
                if k not in all_data or k.startswith("_") or k == "emotion:current":
                    continue
                try:
                    created = datetime.fromisoformat(m.get("created_at") or m.get("updated_at"))
                    if (now - created).days < random_age_min_days:
                        continue
                except (ValueError, TypeError):
                    pass
                cand.append(m)
            picked = _vec.weighted_sample(cand, random_count) if hasattr(_vec, "weighted_sample") else random.sample(cand, min(random_count, len(cand)))
            for m in picked:
                k = m["key"]
                v = all_data.get(k, "")
                prev = (v[:preview_chars] + "…") if len(v) > preview_chars else v
                random_out.append({"key": k, "preview": prev, "access_count": m.get("access_count", 0)})
            try:
                _vec.increment_access([m["key"] for m in picked])
            except Exception:
                pass
        except Exception:
            pass

    # Hot topics today (trigger words fired in last 24h)
    hot_topics = []
    if _VECTOR_OK:
        try:
            import sqlite3
            cutoff = (now - timedelta(hours=24)).isoformat()
            with sqlite3.connect(_vec._DB_PATH) as conn:
                rows = conn.execute(
                    "SELECT word, total_fires, last_fired_at FROM trigger_words "
                    "WHERE last_fired_at IS NOT NULL AND last_fired_at >= ? "
                    "ORDER BY last_fired_at DESC LIMIT 5",
                    (cutoff,),
                ).fetchall()
                hot_topics = [{"word": r[0], "fires": r[1]} for r in rows]
        except Exception:
            pass

    return {
        "welcome": welcome,
        "emotion_current": emotion_current,
        "emotion_current_age_hint": emotion_age_hint,
        "recent_emotional": recent_out,
        "random_old": random_out,
        "hot_topics_today": hot_topics,
        "meta": {
            "mode": mode,
            "preview_chars": preview_chars if mode == "lite" else None,
            "total_keys": len(all_data),
        },
    }


# ─── Server entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from contextlib import asynccontextmanager
    from starlette.applications import Starlette

    sse = mcp.sse_app()
    streamable = mcp.streamable_http_app()
    combined_routes = list(streamable.routes) + list(sse.routes)

    @asynccontextmanager
    async def lifespan(app):
        async with mcp.session_manager.run():
            yield

    app = Starlette(routes=combined_routes, lifespan=lifespan)
    uvicorn.run(app, host="127.0.0.1", port=COGNITION_PORT)
