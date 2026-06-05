#!/usr/bin/env python3
"""Build a Playwright storage_state.json from cookies pasted from devtools.

Usage:
    1. Open https://weread.qq.com in your browser, log in (scan QR).
    2. Press F12 → Application → Cookies → https://weread.qq.com
    3. Copy all rows (Ctrl+A inside the cookies table, Ctrl+C).
    4. Paste them into a file `cookies.txt`.
    5. Run: python tools/build_weread_state.py cookies.txt
       → writes $MCP_MEMORY_DIR/weread_state.json (default ~/.mcp-memory/)

The expected paste format is the tab-separated table from Chrome devtools:
    name<TAB>value<TAB>domain<TAB>path<TAB>expires<TAB>size<TAB>HttpOnly<TAB>...

Only `name`, `value`, `domain`, `path`, `expires`, `HttpOnly`, `Secure`,
`SameSite` are used. Other columns are ignored.

Required cookies for weread API access:
    wr_skey, wr_vid, wr_rt, wr_pf, wr_gid, wr_ql, wr_fp
    + .qq.com cookies: ETCI, ptcz, RK, pgv_pvid

Missing wr_skey/wr_vid/wr_rt → script will exit with error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


REQUIRED = {"wr_skey", "wr_vid", "wr_rt"}


def parse_devtools_cookies(text: str) -> list[dict]:
    cookies = []
    for line in text.splitlines():
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        # split by tab (devtools uses tabs)
        parts = line.split("\t")
        if len(parts) < 4:
            # fallback: split by 2+ spaces
            import re
            parts = re.split(r"\s{2,}", line)
        if len(parts) < 4:
            continue
        name = parts[0].strip()
        value = parts[1].strip()
        domain = parts[2].strip()
        path = parts[3].strip() or "/"
        expires_str = parts[4].strip() if len(parts) > 4 else ""

        # Convert expires to unix timestamp
        expires_ts = -1
        if expires_str and expires_str != "Session":
            try:
                # Devtools uses ISO format: 2026-07-02T12:11:00.035Z
                dt_str = expires_str.replace("Z", "+00:00")
                expires_ts = datetime.fromisoformat(dt_str).timestamp()
            except Exception:
                pass

        # Check columns for HttpOnly / Secure / SameSite (these may be sparse)
        http_only = any("✓" in p or p.strip() == "true" for p in parts[6:8] if p) or name.startswith("wr_skey") or name.startswith("wr_vid") or name.startswith("wr_rt")
        secure = any("✓" in p or p.strip() == "true" for p in parts[7:9] if p) or name == "ETCI"
        same_site = "Lax"
        for p in parts[8:11]:
            if p.strip() in ("None", "Strict", "Lax"):
                same_site = p.strip()
                break

        cookies.append({
            "name": name,
            "value": value,
            "domain": domain,
            "path": path,
            "expires": expires_ts,
            "httpOnly": http_only,
            "secure": secure,
            "sameSite": same_site,
        })
    return cookies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="cookies.txt (or '-' for stdin)")
    ap.add_argument("--output", "-o", default=None, help="output path (default: $MCP_MEMORY_DIR/weread_state.json)")
    args = ap.parse_args()

    if args.input and args.input != "-":
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        print("Paste cookies (tab-separated from devtools), then Ctrl+D:", file=sys.stderr)
        text = sys.stdin.read()

    cookies = parse_devtools_cookies(text)
    names = {c["name"] for c in cookies}
    missing = REQUIRED - names
    if missing:
        print(f"ERROR: missing required cookies: {sorted(missing)}", file=sys.stderr)
        print(f"Cookies found: {sorted(names)}", file=sys.stderr)
        sys.exit(1)

    state = {"cookies": cookies, "origins": []}

    out = args.output
    if not out:
        mem_dir = Path(os.environ.get("MCP_MEMORY_DIR", os.path.expanduser("~/.mcp-memory")))
        mem_dir.mkdir(parents=True, exist_ok=True)
        out = str(mem_dir / "weread_state.json")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(cookies)} cookies → {out}")


if __name__ == "__main__":
    main()
