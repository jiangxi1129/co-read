"""WeChat Reading (weread.qq.com) chapter fetcher.

Uses the preRenderContent intercept technique from Sec-ant/weread-scraper:
WeRead renders chapter text into a hidden #preRenderContent DOM node, rasterizes
to canvas, then removes the node. We inject a MutationObserver via add_init_script
BEFORE page JS runs to clone innerHTML the moment it appears.

Also handles wr_skey rotation: weread JS silently rotates the cookie in the
browser (no Set-Cookie header), so any plain-httpx client dies after ~48 hours.
Our `_save_state_if_changed` saves the rotated cookies back to storage_state.json
on every successful page load, keeping the session alive indefinitely.

Ref: https://github.com/Sec-ant/weread-scraper
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, BrowserContext, Page

HERE = Path(__file__).parent
INIT_SCRIPT = (HERE / "_init_script.js").read_text(encoding="utf-8")

# Storage paths (configurable via env)
MCP_MEMORY_DIR = Path(os.environ.get("MCP_MEMORY_DIR", os.path.expanduser("~/.mcp-memory")))
DEFAULT_STATE_PATH = str(MCP_MEMORY_DIR / "weread_state.json")
WEREAD_MODULE_DIR = os.environ.get("WEREAD_MODULE_DIR", str(HERE))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
VIEWPORT = {"width": 1280, "height": 900}

SHELF_URL = "https://weread.qq.com/web/shelf"
HOME_URL = "https://weread.qq.com"


# ---------- bookId encoding cache ----------------------------------------
# raw bookId (numeric, from /web/shelf/sync API) → encoded form
# (used in URLs like /web/reader/<encoded>). 24h cache; we re-derive via
# /web/book/info?bookId=<raw> which returns an `encodeId` field.
_URL_MAP_CACHE = str(MCP_MEMORY_DIR / "weread_url_map.json")
_URL_MAP_TTL_SEC = 24 * 3600


def _load_url_map_cache() -> dict[str, str]:
    try:
        if not os.path.exists(_URL_MAP_CACHE):
            return {}
        if time.time() - os.path.getmtime(_URL_MAP_CACHE) > _URL_MAP_TTL_SEC:
            return {}
        return json.loads(Path(_URL_MAP_CACHE).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_url_map_cache(mapping: dict[str, str]) -> None:
    try:
        Path(_URL_MAP_CACHE).parent.mkdir(parents=True, exist_ok=True)
        Path(_URL_MAP_CACHE).write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


async def _save_state_if_changed(ctx, state_path: str) -> None:
    """Save storage_state on every call — captures the silently-rotated wr_skey."""
    try:
        await ctx.storage_state(path=state_path)
    except Exception:
        pass


async def _scrape_shelf_url_map(state_path: str) -> dict[str, str]:
    """Open shelf, scroll to bottom to trigger virtual-list lazy load, then
    scrape all <a href='/web/reader/<encoded>'> links + cover images (cover URL
    contains raw bookId). Returns raw_bookId → encoded mapping for those books."""
    async with async_playwright() as pw:
        ctx = await _new_context(pw, state_path=state_path, headless=True)
        page = await ctx.new_page()
        try:
            await page.goto(SHELF_URL, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)
            prev_count = -1
            for _ in range(8):
                count = await page.evaluate(
                    "() => document.querySelectorAll('a[href*=\"/web/reader/\"]').length"
                )
                if count == prev_count and count > 0:
                    break
                prev_count = count
                await page.evaluate("""() => {
                    window.scrollTo(0, document.body.scrollHeight);
                    const scrollables = document.querySelectorAll('[class*=shelf], [class*=Shelf], [class*=scroll]');
                    for (const el of scrollables) {
                        if (el.scrollHeight > el.clientHeight) {
                            el.scrollTop = el.scrollHeight;
                        }
                    }
                }""")
                await asyncio.sleep(1.5)

            entries = await page.evaluate("""() => {
                const out = [];
                const links = document.querySelectorAll('a[href*="/web/reader/"]');
                for (const a of links) {
                    const m = a.href.match(/\\/web\\/reader\\/([^/?#]+)/);
                    if (!m) continue;
                    const encoded = m[1];
                    const img = a.querySelector('img') ||
                                a.closest('[class*=Card]')?.querySelector('img') ||
                                a.closest('[class*=item]')?.querySelector('img') ||
                                a.closest('div')?.querySelector('img');
                    const cover = img?.src || '';
                    const idMatch = cover.match(/cover\\/\\d+\\/(?:yuewen_|YueWen_|cpplatform_)?([0-9]+)/i);
                    const raw_id = idMatch ? idMatch[1] : null;
                    const title = (a.innerText || '').trim();
                    out.push({encoded, cover, raw_id, title});
                }
                return out;
            }""")
            await _save_state_if_changed(ctx, state_path)
            mapping = {}
            for e in entries:
                if e.get('raw_id'):
                    mapping[e['raw_id']] = e['encoded']
            return mapping
        finally:
            await ctx.close()


async def _resolve_via_book_info_api(state_path: str, raw_id: str) -> str | None:
    """Hit /web/book/info?bookId=<raw> — response contains `encodeId` field.
    ~100ms vs ~5s for a browser open."""
    import sys as _sys
    if WEREAD_MODULE_DIR not in _sys.path:
        _sys.path.insert(0, WEREAD_MODULE_DIR)
    try:
        from weread_write import _ensure_fresh_cookies as _efc
        import httpx as _httpx
        cookies = await _efc(state_path)
        async with _httpx.AsyncClient(
            cookies=cookies,
            headers={
                "User-Agent": UA,
                "Referer": f"{HOME_URL}/",
                "Origin": HOME_URL,
                "Accept": "application/json",
            },
            timeout=15.0,
        ) as c:
            r = await c.get(f"{HOME_URL}/web/book/info", params={"bookId": str(raw_id)})
            d = r.json()
            if 'errCode' in d:
                return None
            return d.get('encodeId')
    except Exception:
        return None


async def _resolve_encoded(state_path: str, book_id: str) -> str:
    """raw bookId → encoded URL form. Passthrough if already encoded.

    Strategy:
      1. cache lookup (24h TTL)
      2. hit /web/book/info API (~100ms)
      3. fallback: shelf scrape (slow but warms many books at once)
    """
    bl = book_id.lower()
    has_letter = any(c.isalpha() for c in bl)
    # Already encoded (URL form: 12+ chars with letters)
    if has_letter and len(bl) >= 12:
        return book_id
    cache = _load_url_map_cache()
    if book_id in cache:
        return cache[book_id]
    encoded = await _resolve_via_book_info_api(state_path, book_id)
    if encoded:
        cache[book_id] = encoded
        _save_url_map_cache(cache)
        return encoded
    fresh = await _scrape_shelf_url_map(state_path)
    if fresh:
        cache.update(fresh)
        _save_url_map_cache(cache)
        if book_id in cache:
            return cache[book_id]
    return book_id


def _encode_book_id(book_id: str) -> str:
    """Sync passthrough — kept for callers that don't have an event loop.
    Real resolution goes through async _resolve_encoded."""
    return book_id


def _encode_chapter_uid(uid: str | int) -> str:
    uid_s = str(uid)
    if uid_s.lstrip("-").isdigit():
        n = int(uid_s)
        if n < 0:
            n = n & 0xFFFFFFFF
        return hex(n)[2:]
    import hashlib
    return hashlib.md5(uid_s.encode()).hexdigest()[:8]


def _reader_url(book_id: str, chapter_uid: str | int | None) -> str:
    enc = _encode_book_id(book_id)
    # Empty chapter_uid: don't append k<...> segment → weread auto-redirects to
    # the user's last reading position or chapter 1.
    if chapter_uid is None or chapter_uid == "" or str(chapter_uid).strip() == "":
        return f"https://weread.qq.com/web/reader/{enc}"
    return f"https://weread.qq.com/web/reader/{enc}k{_encode_chapter_uid(chapter_uid)}"


# ----------------------------- context helper ---------------------------------
async def _new_context(pw, state_path: str | None, headless: bool = True) -> BrowserContext:
    browser = await pw.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    ctx_kwargs: dict[str, Any] = {
        "user_agent": UA,
        "viewport": VIEWPORT,
        "locale": "zh-CN",
        "timezone_id": "Asia/Shanghai",
    }
    if state_path and os.path.exists(state_path):
        ctx_kwargs["storage_state"] = state_path
    ctx = await browser.new_context(**ctx_kwargs)
    await ctx.add_init_script(INIT_SCRIPT)
    await ctx.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    )
    return ctx


# ----------------------------- login / state ----------------------------------
async def login_save_state(state_path: str = DEFAULT_STATE_PATH) -> dict:
    """Open visible Chromium, user scans QR, save storage_state.

    This is the manual-login path. Most users prefer build_weread_state.py which
    takes cookies pasted from devtools — no browser launch needed.
    """
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        ctx = await _new_context(pw, state_path=None, headless=False)
        page = await ctx.new_page()
        await page.goto(HOME_URL, wait_until="domcontentloaded")
        deadline = time.time() + 300
        ok = False
        while time.time() < deadline:
            try:
                await page.goto(SHELF_URL, wait_until="domcontentloaded", timeout=15000)
                content = await page.content()
                if "登录" not in content[:5000] and ("shelf" in content.lower() or "书架" in content):
                    ok = True
                    break
            except Exception:
                pass
            await asyncio.sleep(3)
        if ok:
            await ctx.storage_state(path=state_path)
        await _save_state_if_changed(ctx, state_path)
        await ctx.close()
        if ok:
            return {"success": True, "msg": f"state saved → {state_path}"}
        return {"success": False, "msg": "timeout waiting for login (5 min)"}


async def check_state_valid(state_path: str = DEFAULT_STATE_PATH) -> bool:
    """Test if cookies actually work against the weread API.

    A successful HTML page load does NOT prove session validity — that only
    requires the cookie field to exist client-side. The server-side session can
    be dead. So this function:
      1. Triggers cookie refresh (weread_write._ensure_fresh_cookies opens
         browser to let JS rotate wr_skey)
      2. Hits /web/shelf/sync with the fresh cookies
      3. Returns True only if response contains a real book count
    """
    if not os.path.exists(state_path):
        return False
    try:
        import sys as _sys
        if WEREAD_MODULE_DIR not in _sys.path:
            _sys.path.insert(0, WEREAD_MODULE_DIR)
        from weread_write import _ensure_fresh_cookies
        import httpx as _httpx
        cookies = await _ensure_fresh_cookies(state_path)
        headers = {
            "User-Agent": UA,
            "Referer": f"{HOME_URL}/",
            "Origin": HOME_URL,
            "Accept": "application/json",
        }
        async with _httpx.AsyncClient(cookies=cookies, headers=headers, timeout=15) as c:
            r = await c.get(f"{HOME_URL}/web/shelf/sync?synckey=0")
            try:
                data = r.json()
            except Exception:
                return False
            if 'errCode' in data:
                return False
            return data.get('bookCount', 0) > 0
    except Exception:
        return False


# ----------------------------- TOC --------------------------------------------
async def fetch_book_toc(state_path: str, book_id: str) -> dict:
    """Scrape chapter list from book detail page. book_id can be raw or encoded."""
    encoded = await _resolve_encoded(state_path, book_id)
    url = f"https://weread.qq.com/web/bookDetail/{encoded}"
    async with async_playwright() as pw:
        ctx = await _new_context(pw, state_path=state_path, headless=True)
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)
        try:
            await page.locator("text=目录").first.click(timeout=3000)
            await asyncio.sleep(1)
        except Exception:
            pass
        data = await page.evaluate("""() => {
            try {
                const s = window.__INITIAL_STATE__ || {};
                const r = s.reader || s.book || {};
                return {
                    title: r.bookInfo?.title || document.title,
                    author: r.bookInfo?.author || '',
                    chapters: r.chapterInfos || r.chapters || [],
                };
            } catch(e) { return {error: String(e)}; }
        }""")
        if not data.get("chapters"):
            html = await page.content()
            soup = BeautifulSoup(html, "lxml")
            chapters = []
            for i, a in enumerate(soup.select("[class*=chapter] a, .chapterItem")):
                title = a.get_text(strip=True)
                href = a.get("href", "")
                m = re.search(r"k([0-9a-f]+)", href)
                cuid = m.group(1) if m else str(i)
                chapters.append({"idx": i, "title": title, "chapter_uid": cuid})
            data["chapters"] = chapters
        await _save_state_if_changed(ctx, state_path)
        await ctx.close()
        norm = []
        for i, c in enumerate(data.get("chapters", [])):
            norm.append({
                "idx": c.get("chapterIdx", i),
                "title": c.get("title", f"Chapter {i+1}"),
                "chapter_uid": str(c.get("chapterUid", c.get("chapter_uid", i))),
            })
        return {
            "book_id": book_id,
            "title": data.get("title", ""),
            "author": data.get("author", ""),
            "chapters": norm,
        }


# ----------------------------- chapter fetch ----------------------------------
_CHAP_NUM_RE = re.compile(r'第\s*(\d+)\s*章')


def _extract_chapter_num(html: str) -> int | None:
    """Extract chapter number N from `第 N 章` in captured HTML. None if absent."""
    m = re.search(r'data-wr-id="chapterTitle"[^>]*>([^<]+)<', html)
    if not m:
        return None
    title = m.group(1)
    n = _CHAP_NUM_RE.search(title)
    return int(n.group(1)) if n else None


def _extract_chapter_title(html: str) -> str | None:
    m = re.search(r'data-wr-id="chapterTitle"[^>]*>([^<]+)<', html)
    return m.group(1).strip() if m else None


async def fetch_chapter(state_path: str, book_id: str, chapter_uid: str,
                        max_pages: int = 200) -> dict:
    """Fetch one complete chapter (across multiple sections).

    WeRead splits chapters into "sections" (e.g. `第105章 龙骨十字(1)/(2)/(3)`),
    each fires a preRenderContent insert. Strategy: keep pressing ArrowRight,
    watch chapter-number changes — if the main chapter number changes, we've
    crossed into the next chapter — stop and discard the spillover section.

    `book_id` accepts:
      - raw numeric bookId (e.g. "933334") — auto-resolved to encoded form
      - URL-encoded form (e.g. "be5328e0813ab8bdcg0179fb") — used directly
    """
    encoded = await _resolve_encoded(state_path, book_id)
    url = _reader_url(encoded, chapter_uid)
    captured_pages: list[str] = []
    async with async_playwright() as pw:
        ctx = await _new_context(pw, state_path=state_path, headless=True)
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(3)
        page_title = await page.evaluate(
            "() => document.querySelector('.readerTopBar h1')?.innerText || document.title"
        )

        seen_hashes: set[int] = set()
        chapter_nums_seen: list[int] = []
        section_titles_seen: list[str] = []

        async def drain() -> int:
            """Pull new captures, append to list. Returns count of NEW captures."""
            arr = await page.evaluate(
                "() => { const a = window.__weread_captured || []; window.__weread_captured = []; return a; }"
            )
            new_count = 0
            for entry in arr:
                h = hash(entry["html"])
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                captured_pages.append(entry["html"])
                new_count += 1
                num = _extract_chapter_num(entry["html"])
                t = _extract_chapter_title(entry["html"])
                if num is not None:
                    chapter_nums_seen.append(num)
                if t:
                    section_titles_seen.append(t)
            return new_count

        await drain()

        crossed_chapter = False
        book_ended = False
        idle_count = 0
        # ArrowRight ≈ 1 screen, 1 section = ~6-10 screens, so be generous with idle threshold.
        IDLE_THRESHOLD = 15
        for pg in range(max_pages):
            try:
                await page.keyboard.press("ArrowRight")
            except Exception:
                try:
                    await page.locator(".readerControls_right, .renderTargetContainer").first.click(timeout=2000)
                except Exception:
                    break
            await asyncio.sleep(random.uniform(0.7, 1.4))
            new = await drain()
            if new == 0:
                idle_count += 1
                # Every 8 idle keypresses, also try PageDown as fallback
                if idle_count % 8 == 0:
                    try:
                        await page.keyboard.press("PageDown")
                        await asyncio.sleep(0.8)
                        new = await drain()
                        if new > 0:
                            idle_count = 0
                    except Exception:
                        pass
                # "End of book" banner — stop
                ending = await page.evaluate(
                    "() => !!document.querySelector('.readerFooter_ending_finish, .readerFooter_ending')"
                )
                if ending:
                    book_ended = True
                    break
                if idle_count >= IDLE_THRESHOLD:
                    break
                continue
            idle_count = 0

            # Cross-chapter detection: main chapter number changed
            unique_nums = sorted(set(n for n in chapter_nums_seen if n is not None))
            if len(unique_nums) > 1:
                first_chap = unique_nums[0]
                cut_idx = None
                for i, n in enumerate(chapter_nums_seen):
                    if n != first_chap:
                        cut_idx = i
                        break
                if cut_idx is not None:
                    captured_pages = captured_pages[:cut_idx]
                    section_titles_seen = section_titles_seen[:cut_idx]
                crossed_chapter = True
                break

        await _save_state_if_changed(ctx, state_path)
        await ctx.close()

    full_html = "\n".join(captured_pages)
    soup = BeautifulSoup(full_html, "lxml")
    # WeRead wraps every Chinese character in its own <span> — can't use \n as
    # separator (gives one char per line). Instead: append \n\n after each <p>,
    # convert <br> to \n, add separators around chapter titles, then get_text("").
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for p_tag in soup.find_all("p"):
        p_tag.append("\n\n")
    for title_tag in soup.find_all(attrs={"data-wr-id": "chapterTitle"}):
        title_tag.insert_before("\n\n")
        title_tag.append("\n\n")
    text = soup.get_text("", strip=False)
    text = re.sub(r"[ \t]+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chapter_title = section_titles_seen[0] if section_titles_seen else page_title

    return {
        "book_id": book_id,
        "chapter_uid": chapter_uid,
        "title": chapter_title,
        "page_title": page_title,
        "text": text,
        "section_count": len(captured_pages),
        "section_titles": section_titles_seen,
        "crossed_chapter": crossed_chapter,
        "book_ended": book_ended,
        "captured_at": int(time.time()),
    }


# CLI smoke test
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    state = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else DEFAULT_STATE_PATH
    print(f"[weread_fetch] cmd={cmd}  state={state}", flush=True)

    if cmd == "login":
        print(asyncio.run(login_save_state(state)))
    elif cmd == "check":
        print(asyncio.run(check_state_valid(state)))
    elif cmd == "toc":
        print(json.dumps(asyncio.run(fetch_book_toc(state, sys.argv[3])), ensure_ascii=False, indent=2))
    elif cmd == "ch":
        print(json.dumps(asyncio.run(fetch_chapter(state, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")), ensure_ascii=False, indent=2))
