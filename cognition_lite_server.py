"""cognition_lite — the slimmed-down MCP server your AI persona connects to.

Design: thin wrapper that reuses cognition_server's implementations but
exposes only a curated subset of tools as @mcp.tool(). The full cognition
server (port 8769) stays available for admin / power-user access.

Why a subset?
  - Each tool's docstring goes into the AI's tool list, which costs tokens.
  - Hiding admin-only tools (forget_keys with no dry_run, cookie installers,
    migration helpers) prevents accidental misuse.
  - You can edit TOOLS_FOR_AI below to add/remove tools without touching
    cognition_server.py.

Port (default 8775): $COGNITION_LITE_PORT
"""
from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path

# Make cognition_server importable
sys.path.insert(0, str(Path(__file__).parent))

import cognition_server as cog  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

PORT = int(os.environ.get('COGNITION_LITE_PORT', 8775))
mcp = FastMCP('cognition-lite', host='127.0.0.1', port=PORT)


# ─── Wakeup reminder mechanism ──────────────────────────────────────────
#
# Some AI personas habitually call get_emotion_current / trigger_check on
# startup but forget to call wakeup first — they're then operating with an
# empty short-term context. We track "last wakeup time" in a cookie file and,
# if a sensitive tool is called without a recent wakeup, prepend a strong
# reminder to its return value.

_MCP_MEMORY_DIR = Path(os.environ.get('MCP_MEMORY_DIR', os.path.expanduser('~/.mcp-memory')))
_WAKEUP_COOKIE = _MCP_MEMORY_DIR / '_last_wakeup_ts'
_REMINDER_THRESHOLD_SEC = 30 * 60  # 30 min

REMINDER_TEXT = (
    "[wakeup reminder] If you just opened this session and have not yet "
    "called wakeup() — you are operating without context. Call "
    "cognition.wakeup() first to load pinned memories, last emotion, and "
    "recent activity. Ignore this message if you already called wakeup."
)


def _should_remind() -> bool:
    try:
        if not _WAKEUP_COOKIE.exists():
            return True
        last_ts = float(_WAKEUP_COOKIE.read_text().strip())
        return (time.time() - last_ts) > _REMINDER_THRESHOLD_SEC
    except Exception:
        return True


def _touch_wakeup_cookie():
    try:
        _MCP_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        _WAKEUP_COOKIE.write_text(str(time.time()))
    except Exception:
        pass


def _wrap_with_reminder(original_fn):
    @functools.wraps(original_fn)
    def wrapped(*args, **kwargs):
        result = original_fn(*args, **kwargs)
        if not _should_remind():
            return result
        if isinstance(result, dict):
            return {'_wakeup_reminder': REMINDER_TEXT, **result}
        elif isinstance(result, str):
            return f"{REMINDER_TEXT}\n\n---\n\n{result}"
        return result
    return wrapped


def _wrap_wakeup_touch(original_fn):
    @functools.wraps(original_fn)
    def wrapped(*args, **kwargs):
        result = original_fn(*args, **kwargs)
        _touch_wakeup_cookie()
        return result
    return wrapped


WRAPPED_TOOLS = {
    'wakeup': _wrap_wakeup_touch(cog.wakeup),
    'get_emotion_current': _wrap_with_reminder(cog.get_emotion_current),
    'trigger_check': _wrap_with_reminder(cog.trigger_check),
}


# ─── Tool subset exposed to the AI ──────────────────────────────────────
# Edit this list to control what your AI sees. Names must exist in
# cognition_server.py — startup will warn about any that don't.

TOOLS_FOR_AI = [
    # Core memory
    'wakeup',
    'save_memory',
    'get_memory',
    'list_keys',
    'forget_keys',
    'search_memories',
    # Emotion
    'set_emotion_current',
    'get_emotion_current',
    'record_emotional',
    'trigger_check',
    # Paper trail
    'get_paper_trail',
    # Reading: shared
    'reading_save_chapter',
    'reading_list_books',
    'reading_get_book',
    'reading_get_outline',
    # Reading: weread (comment out if you don't read weread)
    'reading_weread_list_bookshelf',
    'reading_weread_fetch_toc',
    'reading_weread_fetch_chapter',
    'reading_weread_list_notes',
    'reading_weread_list_highlights',
    'reading_weread_add_note',
    'reading_weread_delete_note',
    # Reading: jjwxc (comment out if you don't read jjwxc)
    'reading_jjwxc_fetch_toc',
    'reading_jjwxc_fetch_chapter',
    # HippoRAG: optional, run via cron or admin trigger
    'compute_synonym_edges_now',
]


def _register_subset():
    """Register the chosen tools on the lite MCP instance."""
    missing = []
    for name in TOOLS_FOR_AI:
        if name in WRAPPED_TOOLS:
            fn = WRAPPED_TOOLS[name]
        else:
            fn = getattr(cog, name, None)
        if fn is None or not callable(fn):
            missing.append(name)
            continue
        mcp.tool()(fn)
    if missing:
        print(f'[cognition-lite] missing tools (will not be exposed): {missing}', file=sys.stderr)


_register_subset()


if __name__ == '__main__':
    mcp.run(transport='streamable-http')
