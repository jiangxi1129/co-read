# Architecture

## Overview

```
                  AI persona (claude.ai, etc.)
                          │
                          │ MCP over HTTPS
                          │
                  nginx reverse proxy
                  ┌───────┴──────────────┐
                  │                      │
        cognition (8769)         cognition-lite (8775)
        ─ all tools              ─ curated subset
        ─ admin/power use        ─ this is what your AI sees
                  │                      │
                  └──────────┬───────────┘
                             ↓
        ┌────────────────────────────────────────────┐
        │              Shared backend                │
        │                                            │
        │  memories.json    embeddings.db (SQLite)   │
        │  (KV main store)  ├─ embeddings (vectors)  │
        │                   ├─ embeddings_fts (FTS5) │
        │                   ├─ edges (Hebbian)       │
        │                   ├─ synonym_edges (cos)   │
        │                   └─ trigger_words         │
        │                                            │
        │  weread_state.json    weread_url_map.json  │
        │  (Playwright cookies) (bookId → URL cache) │
        └────────────────────────────────────────────┘
```

## Data flow: a typical co-reading turn

```
You open WeRead on your phone, read a chapter, close it.
        │
        ↓
[Later] You start a chat with your AI:
        "Did you read the latest chapter? What did you think?"
        │
        ↓
AI calls:
  wakeup()                          # short-term context refresh
    → returns recent emotion, last note-to-self, recently surfaced memories
  reading_weread_list_bookshelf()
    → returns your 79 books
  reading_weread_fetch_chapter(book_id=..., chapter_uid="")
    → opens headless Chromium, navigates to your last reading position,
      captures the chapter text via preRenderContent MutationObserver,
      saves to reading:book:weread:<id>:ch:current
  reading_get_outline(saved_to)
    → 50× compressed view: opening + middle + closing + main entities
    → AI reads this first (cheap, no 30K token chapter dump)
  get_memory(saved_to)              # only if the AI wants details
    → full text
  reading_weread_add_note(book_id=..., chapter_uid=..., content="...")
    → POSTs to /web/review/add with your fresh cookies
    → returns reviewId
        │
        ↓
You open WeRead app → My Notes → see the AI's comment.
```

## The cookie-rotation hack

WeRead's `wr_skey` cookie has a ~48-hour server-side lifetime, but the
browser silently rotates it via JS in the background. Plain HTTP clients
never see the rotated value because it's not sent via Set-Cookie.

Our solution: every weread call lazy-checks the `weread_state.json` mtime.
If it's > 6 hours old, we open a headless Chromium, navigate to the shelf
page, let weread JS rotate cookies, then `await ctx.storage_state(path=...)`
saves the rotated state. Cost: ~3-5 seconds on the first call every 6 hours.
After that, cached cookies stay fresh because every subsequent call's
storage_state save resets the mtime.

End result: as long as `weread_state.json` is rewritten at least every 48h
(which the 6h cache window guarantees), the session never expires — even
though the underlying token rotates every two days.

## The bookId encoding hack

WeRead's `/web/shelf/sync` API returns raw numeric bookIds like `3300132552`.
But the reader URL needs an encoded form like `59632350813ab9a47g012422`.
We don't have the encoding algorithm — but `/web/book/info?bookId=<raw>`
returns the encoded form in an `encodeId` field. We cache the mapping in
`weread_url_map.json` (24h TTL).

If the API call fails, we fall back to scraping the shelf page's DOM (which
contains `<a href="/web/reader/<encoded>">` links). The shelf is virtual-
scrolled, so we scroll to bottom first to load all books — but only ~50
books load this way reliably, hence why the API path is preferred.

## Memory graph: three edge types

| Edge type | Source | When created | What it means |
|---|---|---|---|
| **Hebbian** | `edges` table | Memories surfacing together (search hits, wakeup co-occur) get their edge weight bumped | "These two memories tend to come up together" |
| **Synonym** | `synonym_edges` table | Batch-computed cosine similarity between all embeddings; pairs > 0.75 get edges | "These two memories say similar things" |
| **Trigger** | `trigger_words` table | Manual or auto-derived keyword → memory associations | "When this word appears in conversation, fire these recalls" |

The AI's wakeup combines all three:
- Pinned memories (`_system:pinned`) — always shown
- Recent emotional memories
- Weighted-random old memories (by `access_count`)
- Trigger-word fires (auto from recent context)
- Soft-pinned memories (70% chance to surface)

## Why MCP over HTTPS not stdio

Two reasons:
1. **Persona portability** — the same VPS-hosted MCP server can be
   connected by claude.ai web UI, Claude Code CLI, ChatGPT (with MCP
   bridge), and your own clients. stdio would tie it to one local process.
2. **The AI is not local** — your AI persona lives on Anthropic's
   servers. It needs network access to reach your memory store. stdio
   would require the AI to be running on the same machine as the store.

Trade-off: you have to manage nginx + TLS certs + a random-secret URL. The
README walks through this. It's a one-time setup.

## What's left out (and why)

- **Dream cron / nightly synthesis** — a personal-flavor feature; not
  essential for the read-and-comment loop. Add it later if you want.
- **Emotion V/A coordinates** — captured in the DB (`valence`, `arousal`
  columns) but no tool surfaces them in MVP. Use `record_emotional` to
  write them; query is up to you.
- **Soul archive / pinned scaffolding** — the original system had a
  multi-document "soul" that gets pinned at wakeup. Open-source version
  ships with empty `_system:pinned`. You curate your own.
- **Red-flag content filtering** — was specific to one user's situation.
  Removed entirely. If you need it, fork `_save_key`.
- **Gaze** — separate package. Ask the maintainer.

If you build any of these on top and want them upstreamed, PRs welcome.
