#!/usr/bin/env python3
"""Initialize an empty MCP memory directory.

Creates the directory structure + an empty memories.json + initializes the
SQLite schema. Idempotent — safe to re-run.

Usage:
    python tools/init_memory.py
    # Or with custom path:
    MCP_MEMORY_DIR=/var/lib/mcp python tools/init_memory.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path


def main():
    mem_dir = Path(os.environ.get('MCP_MEMORY_DIR', os.path.expanduser('~/.mcp-memory')))
    mem_dir.mkdir(parents=True, exist_ok=True)
    print(f"memory dir: {mem_dir}")

    # memories.json
    mem_file = mem_dir / 'memories.json'
    if not mem_file.exists():
        mem_file.write_text('{}', encoding='utf-8')
        print(f"  created empty {mem_file}")
    else:
        print(f"  exists: {mem_file}")

    # backups dir
    (mem_dir / 'backups').mkdir(exist_ok=True)
    print(f"  ready: {mem_dir / 'backups'}")

    # embeddings.db — let vector_engine create it on first import
    # but pre-create the file so permissions are right
    db_file = mem_dir / 'embeddings.db'
    if not db_file.exists():
        with sqlite3.connect(db_file) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT OR IGNORE INTO _meta VALUES ('schema_version', '1')")
            conn.commit()
        print(f"  created skeleton {db_file}")
    else:
        print(f"  exists: {db_file}")

    print("\nDone. Now you can:")
    print("  1. Paste weread cookies → python tools/build_weread_state.py cookies.txt")
    print("  2. Start the server     → pm2 start cognition_lite_server.py --interpreter venv/bin/python")
    print("  3. Hook up your AI to https://YOUR_DOMAIN/<your-secret>/mcp")


if __name__ == '__main__':
    main()
