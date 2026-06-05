#!/usr/bin/env python3
"""Weekly weread cookies health check.

Crontab:
    0 3 * * 1  /path/to/venv/bin/python3 /path/to/crons/check_weread_cookies.py

(Every Monday at 03:00 — avoid your other cron schedules.)

If cookies are still valid (the fetch can refresh wr_skey on its own), do
nothing. If they're truly dead (e.g. account-level logout), write a high-priority
todo entry to the memory store and update the next-window note so your AI sees
it on next startup.

Setup:
    Set MCP_MEMORY_DIR + WEREAD_MODULE_DIR in the cron environment, or edit
    the constants at the top of this file.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path


def log(msg: str, log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'[{datetime.now().isoformat(timespec="seconds")}] {msg}\n')


def main():
    MCP_MEMORY_DIR = Path(os.environ.get('MCP_MEMORY_DIR', os.path.expanduser('~/.mcp-memory')))
    WEREAD_MODULE_DIR = os.environ.get('WEREAD_MODULE_DIR', '')
    LOG_PATH = str(MCP_MEMORY_DIR / 'weread_check.log')
    STATE = str(MCP_MEMORY_DIR / 'weread_state.json')

    # Make modules importable
    if WEREAD_MODULE_DIR:
        sys.path.insert(0, WEREAD_MODULE_DIR)
    # Also add parent of this script (assumes layout: repo/crons/this.py and repo/weread/)
    here = Path(__file__).parent.parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here / 'weread'))

    try:
        from weread_fetch import check_state_valid
        valid = asyncio.run(check_state_valid(STATE))
        log(f'check_state_valid → {valid}', LOG_PATH)

        if not valid:
            # Try to write a todo through the cognition memory layer
            try:
                # cognition_server's _save_key works against memories.json
                import cognition_server as c
                save = getattr(c._save_key, 'fn', c._save_key)
                today = datetime.now().strftime('%Y-%m-%d')
                todo_key = f'todo:cookies:weread_expired_{today}'
                todo_val = f"""[priority] high
[created] {datetime.now().isoformat(timespec='seconds')} (auto-detected by weekly cron)
[status] pending

WeRead cookies have expired. Reading flow is broken until refreshed.

How to fix:
  1. In your browser, log into weread.qq.com
  2. F12 → Application → Cookies → copy all rows
  3. Run: python tools/build_weread_state.py cookies.txt
     (this writes a fresh storage_state.json)

Old cookies (for reference): {STATE}.bak
"""
                save(todo_key, todo_val)
                log(f'wrote {todo_key}', LOG_PATH)

                # Also stick a heads-up in next_window_note so the AI sees it
                try:
                    note_key = '_system:next_window_note'
                    note_val = (f'WeRead cookies expired on {today}. Reading flow is down. '
                                f'See {todo_key} for fix steps.')
                    save(note_key, note_val)
                    log('wrote next_window_note', LOG_PATH)
                except Exception as e:
                    log(f'next_window_note failed: {e}', LOG_PATH)
            except Exception as e:
                log(f'could not write todo: {e}', LOG_PATH)
        else:
            log('cookies still valid, no action', LOG_PATH)
    except Exception as e:
        import traceback
        log(f'ERROR: {type(e).__name__}: {e}', LOG_PATH)
        log(traceback.format_exc(), LOG_PATH)


if __name__ == '__main__':
    main()
