"""WeRead user data API: bookshelf, bookmarks (highlight + short note),
reviews (full book / paragraph note).

Uses weread's own web API — no rendering hacks. Zero sign / zero CSRF — cookies
are enough.

Endpoints reverse-engineered via probe:
  POST /web/review/add           {bookId, chapterUid, content, type=1, range?, isPrivate?}
  POST /web/review/delete        {reviewId}
  POST /web/book/addBookmark     {bookId, chapterUid, range, type=1, markText, abstract}
  POST /web/book/removeBookmark  {bookmarkId}
  GET  /web/book/bookmarklist    ?bookId=&syncKey=0
  GET  /web/review/list          ?bookId=&listType=11&listMode=3&syncKey=0&mine=1
  GET  /web/book/underlines      ?bookId=&chapterUid=
  GET  /web/shelf/sync           ?synckey=0

Cookie lifecycle:
  wr_skey has ~48h server-side lifetime, but weread JS silently rotates it
  in the browser (no Set-Cookie response header). Any plain-httpx client dies
  after ~48h. `_ensure_fresh_cookies` opens a headless Chromium when the state
  file is more than 6h old, lets weread JS rotate cookies, and saves them back.
  Result: session stays alive indefinitely with no manual re-login.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import httpx

WEREAD_HOME = "https://weread.qq.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Storage paths (configurable via env)
MCP_MEMORY_DIR = Path(os.environ.get("MCP_MEMORY_DIR", os.path.expanduser("~/.mcp-memory")))
DEFAULT_STATE_PATH = str(MCP_MEMORY_DIR / "weread_state.json")
WEREAD_MODULE_DIR = os.environ.get("WEREAD_MODULE_DIR", str(Path(__file__).parent))

# wr_skey server-side lifetime is ~48h. Don't re-refresh more often than 6h
# (each refresh = ~3-5s Chromium cold start).
_REFRESH_AFTER_HOURS = 6


def _state_age_hours(state_path: str) -> float:
    try:
        mtime = os.path.getmtime(state_path)
        return (time.time() - mtime) / 3600
    except Exception:
        return 999.0


async def _refresh_cookies_via_browser(state_path: str) -> None:
    """Open headless Chromium, visit shelf, let JS rotate wr_skey, save back."""
    if WEREAD_MODULE_DIR not in sys.path:
        sys.path.insert(0, WEREAD_MODULE_DIR)
    from weread_fetch import _new_context
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        ctx = await _new_context(pw, state_path=state_path, headless=True)
        page = await ctx.new_page()
        try:
            await page.goto(f'{WEREAD_HOME}/web/shelf',
                            wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(3)  # let JS finish rotation
            await ctx.storage_state(path=state_path)
        finally:
            await ctx.close()


async def _ensure_fresh_cookies(state_path: str) -> dict[str, str]:
    """Lazy refresh: skip if state file < 6h old; else open browser + save."""
    if _state_age_hours(state_path) > _REFRESH_AFTER_HOURS:
        try:
            await _refresh_cookies_via_browser(state_path)
        except Exception:
            pass
    return _load_cookies(state_path)


def _load_cookies(state_path: str) -> dict[str, str]:
    """Read Playwright storage_state.json → flat cookies dict."""
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    return {c["name"]: c["value"] for c in state.get("cookies", [])}


async def _client(state_path: str) -> httpx.AsyncClient:
    """Build httpx client with auto-refreshed cookies (lazy, 6h cache)."""
    cookies = await _ensure_fresh_cookies(state_path)
    return httpx.AsyncClient(
        cookies=cookies,
        headers={
            "User-Agent": UA,
            "Referer": f"{WEREAD_HOME}/",
            "Origin": WEREAD_HOME,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        timeout=20.0,
    )


# ───────────────────────── reviews (notes / book review) ─────────────────

async def add_review(
    state_path: str,
    book_id: str,
    chapter_uid: int,
    content: str,
    range_str: str | None = None,
    is_private: bool = False,
) -> dict:
    """Add a review/note. `content` is the comment text.
    `range_str="100-110"` anchors to a paragraph; None = chapter/book-level review.
    """
    payload: dict[str, Any] = {
        "bookId": str(book_id),
        "chapterUid": int(chapter_uid),
        "content": content,
        "type": 1,
    }
    if range_str:
        payload["range"] = range_str
    if is_private:
        payload["isPrivate"] = 1
    async with await _client(state_path) as c:
        r = await c.post(f"{WEREAD_HOME}/web/review/add", json=payload)
        try:
            return r.json()
        except Exception:
            return {"error": f"http_{r.status_code}", "body": r.text[:300]}


async def delete_review(state_path: str, review_id: str) -> dict:
    async with await _client(state_path) as c:
        r = await c.post(f"{WEREAD_HOME}/web/review/delete", json={"reviewId": review_id})
        try:
            return r.json()
        except Exception:
            return {"error": f"http_{r.status_code}", "body": r.text[:300]}


async def list_reviews(state_path: str, book_id: str, mine: bool = True) -> dict:
    """List notes for a book. mine=True returns only yours."""
    params = {
        "bookId": str(book_id),
        "listType": 11,
        "listMode": 3,
        "syncKey": 0,
        "mine": 1 if mine else 0,
    }
    async with await _client(state_path) as c:
        r = await c.get(f"{WEREAD_HOME}/web/review/list", params=params)
        try:
            return r.json()
        except Exception:
            return {"error": f"http_{r.status_code}", "body": r.text[:300]}


# ───────────────────────── bookmarks (highlight) ─────────────────────────
# NOTE: We do NOT expose add_bookmark / delete_bookmark as MCP tools by default.
# WeRead validates that `range` corresponds to a real character offset in the
# chapter, with `markText` matching the actual text at that offset. The server
# silently accepts the request (returns 200 + a fake bookmarkId) but stores
# nothing if the range/markText don't match. From the AI's view this looks like
# success; from the user's view nothing appears in the app. That's strictly
# worse than not having the feature. Keep them here as helpers for advanced
# use cases where you can compute exact offsets — re-add the MCP wrappers if
# you have that capability.

async def add_bookmark(
    state_path: str,
    book_id: str,
    chapter_uid: int,
    range_str: str,
    mark_text: str,
    abstract: str = "",
) -> dict:
    """Add a highlight/bookmark. WeRead's internal "bookmark" with markText is
    actually a text highlight.

    Args:
      range_str: "start-end" char offsets, e.g. "100-110"
      mark_text: the text being highlighted (must match the actual chapter text)
      abstract: context summary (optional), defaults to mark_text[:50]
    """
    payload = {
        "bookId": str(book_id),
        "chapterUid": int(chapter_uid),
        "range": range_str,
        "markText": mark_text,
        "abstract": abstract or mark_text[:50],
        "type": 1,
    }
    async with await _client(state_path) as c:
        r = await c.post(f"{WEREAD_HOME}/web/book/addBookmark", json=payload)
        try:
            return r.json()
        except Exception:
            return {"error": f"http_{r.status_code}", "body": r.text[:300]}


async def delete_bookmark(state_path: str, bookmark_id: str) -> dict:
    """bookmarkId format: '{book_id}_{chapterUid}_{range}'"""
    async with await _client(state_path) as c:
        r = await c.post(f"{WEREAD_HOME}/web/book/removeBookmark", json={"bookmarkId": bookmark_id})
        try:
            return r.json()
        except Exception:
            return {"error": f"http_{r.status_code}", "body": r.text[:300]}


async def list_bookmarks(state_path: str, book_id: str) -> dict:
    async with await _client(state_path) as c:
        r = await c.get(f"{WEREAD_HOME}/web/book/bookmarklist",
                        params={"bookId": str(book_id), "syncKey": 0})
        try:
            return r.json()
        except Exception:
            return {"error": f"http_{r.status_code}", "body": r.text[:300]}


# ───────────────────────── underlines (read-only) ─────────────────────────

async def list_underlines(state_path: str, book_id: str, chapter_uid: int) -> dict:
    """The endpoint weread uses for "popular highlights" — same source as
    bookmarklist but a different view."""
    async with await _client(state_path) as c:
        r = await c.get(f"{WEREAD_HOME}/web/book/underlines",
                        params={"bookId": str(book_id), "chapterUid": int(chapter_uid)})
        try:
            return r.json()
        except Exception:
            return {"error": f"http_{r.status_code}", "body": r.text[:300]}


# ───────────────────────── bookshelf ─────────────────────────────────────

async def list_bookshelf(state_path: str) -> dict:
    """Sync the entire bookshelf. Returns ~97KB JSON with all books + progress."""
    async with await _client(state_path) as c:
        r = await c.get(f"{WEREAD_HOME}/web/shelf/sync", params={"synckey": 0})
        try:
            return r.json()
        except Exception:
            return {"error": f"http_{r.status_code}", "body": r.text[:300]}


# ───────────────────────── CLI quick test ────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "shelf"
    state = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_STATE_PATH

    async def run():
        if cmd == "shelf":
            r = await list_bookshelf(state)
            books = r.get("books", [])
            print(f"shelf: {len(books)} books")
            for b in books[:10]:
                print(f"  {b.get('title')} - {b.get('author')} (bookId={b.get('bookId')})")
        elif cmd == "reviews":
            r = await list_reviews(state, sys.argv[3])
            print(json.dumps(r, ensure_ascii=False, indent=2)[:1500])
        elif cmd == "marks":
            r = await list_bookmarks(state, sys.argv[3])
            print(json.dumps(r, ensure_ascii=False, indent=2)[:1500])

    asyncio.run(run())
