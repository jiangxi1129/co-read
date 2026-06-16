"""Tests for the bare-key guard + alias normalization in cognition_server.

Runnable two ways:
    python tests/test_key_guard.py     # plain asserts, exits non-zero on failure
    pytest tests/test_key_guard.py     # if pytest is available

The real `mcp` dependency is stubbed so the module imports without it, and
KEY_ALIASES / MCP_MEMORY_DIR are set *before* import (the module reads env at
import time).
"""
import os
import sys
import types
import tempfile
from pathlib import Path

# ── make cognition_server importable in a bare environment ──────────────
# 1) stub the `mcp` package (only FastMCP with a no-op .tool decorator is used)
_fastmcp = types.ModuleType("mcp.server.fastmcp")


class _FakeMCP:
    def __init__(self, *a, **k):
        pass

    def tool(self, *a, **k):
        def deco(fn):
            return fn
        return deco


_fastmcp.FastMCP = _FakeMCP
sys.modules.setdefault("mcp", types.ModuleType("mcp"))
sys.modules.setdefault("mcp.server", types.ModuleType("mcp.server"))
sys.modules["mcp.server.fastmcp"] = _fastmcp

# 2) env must be set before import (module reads it at import time)
os.environ["MCP_MEMORY_DIR"] = tempfile.mkdtemp(prefix="cotest_")
os.environ["KEY_ALIASES"] = (
    "arcadia:>project:桃源:arcadia_,"
    "cc:todo:>todo:江栈:,"
    "todo:cc:>todo:江栈:,"
    "todo:栈:>todo:江栈:"
)
os.environ["ENFORCE_KEY_PREFIX"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cognition_server as cs  # noqa: E402


# ── alias normalization ─────────────────────────────────────────────────
def test_normalize_folds_alias_prefixes():
    assert cs._normalize_key("arcadia:whitepaper2.5") == ("project:桃源:arcadia_whitepaper2.5", "arcadia:")
    assert cs._normalize_key("cc:todo:disclosure") == ("todo:江栈:disclosure", "cc:todo:")
    assert cs._normalize_key("todo:cc:weread") == ("todo:江栈:weread", "todo:cc:")
    assert cs._normalize_key("todo:栈:gaze") == ("todo:江栈:gaze", "todo:栈:")


def test_normalize_leaves_canonical_keys_untouched():
    assert cs._normalize_key("episodic:2026-06-01T10:00:x") == ("episodic:2026-06-01T10:00:x", None)
    assert cs._normalize_key("todo:江栈:already_ok") == ("todo:江栈:already_ok", None)
    assert cs._normalize_key("息息:profile") == ("息息:profile", None)


# ── bare-key guard ──────────────────────────────────────────────────────
def test_guard_rejects_bare_and_naked_keys():
    assert cs._validate_key("2026-06-01T16:48:topic") is not None   # bare timestamp
    assert cs._validate_key("2026-06-08T09:54") is not None         # pure timestamp
    assert cs._validate_key("noprefixkey") is not None              # no colon
    assert cs._validate_key("") is not None                         # empty


def test_guard_accepts_proper_and_system_keys():
    assert cs._validate_key("episodic:2026-06-01:x") is None
    assert cs._validate_key("semantic:foo") is None
    assert cs._validate_key("息息:profile") is None
    assert cs._validate_key("_meta:trail:x") is None                # system key exempt
    assert cs._validate_key("_system:pinned") is None


# ── the two stages compose: alias first, then guard ─────────────────────
def test_alias_then_guard_composition():
    # a drifting alias key normalizes to a valid canonical key (guard passes)
    norm, alias = cs._normalize_key("cc:todo:x")
    assert alias == "cc:todo:"
    assert cs._validate_key(norm) is None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
