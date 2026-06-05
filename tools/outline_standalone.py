#!/usr/bin/env python3
"""Rule-based chapter outline extraction. Runs standalone on any text file.

Compresses a 30K-char chapter into ~1K-char outline:
  - opening: first sentence of first 3 paragraphs
  - middle: first sentence of ~6 evenly-spaced middle paragraphs
  - closing: first sentence of last 2 paragraphs
  - main_entities: jieba.posseg → top 5 by frequency for nr/nrt/nz/nt tags
  - dialogue_density: speech markers + quote marks per 1K chars
  - head/tail excerpt: first/last 200 raw characters

No LLM, no network — pure regex + jieba. Zero privacy risk.

Usage:
    python tools/outline_standalone.py path/to/chapter.txt
    cat chapter.txt | python tools/outline_standalone.py -
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_FIRST_SENT_PAT_TPL = r'^([^。！？\.\!\?\n]{4,%d}[。！？\.\!\?])'
_CN_RANGE_LO = '一'
_CN_RANGE_HI = '鿿'


def _first_sentence(p: str, max_chars: int = 80) -> str:
    pattern = _FIRST_SENT_PAT_TPL % max_chars
    m = re.match(pattern, p)
    if m:
        return m.group(1).strip()
    snippet = p[:max_chars].strip()
    return snippet + ('…' if len(p) > max_chars else '')


def _is_junk(p: str) -> bool:
    if len(p) < 8:
        return True
    if p.startswith(('##', '[来源]', '【', '<', 'function ', 'var ', '/*', '//')):
        return True
    # JS/CSS keyword pattern
    if re.search(r'\b(type|data|url|method|async|var|let|const)\s*:', p):
        return True
    if 'log_data' in p or 'getElementById' in p or '.ajax' in p:
        return True
    # Code-character density
    code_chars = sum(p.count(c) for c in '{}=();')
    if code_chars / max(len(p), 1) > 0.05:
        return True
    # Low Chinese ratio over a long paragraph → probably code/URL
    if len(p) > 12:
        cn_chars = sum(1 for c in p if _CN_RANGE_LO <= c <= _CN_RANGE_HI)
        if cn_chars / len(p) < 0.3:
            return True
    return False


def get_outline(text: str, source_key: str = "") -> dict:
    if not text or len(text.strip()) < 20:
        return {'error': 'content empty or too short', 'total_chars': len(text)}

    # Try to extract title if input is JSON-wrapped (e.g. weread fetch output)
    title = ""
    if text.lstrip().startswith('{'):
        try:
            obj = json.loads(text)
            inner = obj.get('text') or obj.get('content') or ''
            if inner:
                title = obj.get('title') or obj.get('chapter_title') or ''
                text = inner
        except Exception:
            pass

    # Paragraph split: prefer \n\n; fall back to \n if too few or any too long.
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) < 20 or any(len(p) > 1500 for p in paragraphs):
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    # Markdown header → title fallback
    if not title:
        for p in paragraphs[:3]:
            m = re.match(r'^##\s*(.+?)(?:\s*[·•]\s*(.+))?$', p)
            if m:
                title = m.group(0).lstrip('# ').strip()
                break

    paragraphs = [p for p in paragraphs if not _is_junk(p)]
    n_paras = len(paragraphs)

    if not title and paragraphs and len(paragraphs[0]) < 30:
        title = paragraphs[0]

    opening = [_first_sentence(p) for p in paragraphs[:3]]
    closing = [_first_sentence(p) for p in paragraphs[-2:]] if n_paras > 5 else []

    middle = []
    if n_paras > 5:
        candidates = list(range(3, n_paras - 2))
        if candidates:
            sample_n = min(6, len(candidates))
            step = max(1, len(candidates) // sample_n)
            for i in candidates[::step][:sample_n]:
                middle.append(_first_sentence(paragraphs[i]))

    # Entities via jieba.posseg
    entities = []
    try:
        import jieba.posseg as pseg
        from collections import Counter
        freq: Counter = Counter()
        for w, flag in pseg.cut(text[:30000]):
            w = w.strip()
            if len(w) >= 2 and flag in ('nr', 'nrt', 'nz', 'nt'):
                freq[w] += 1
        entities = [{'name': n, 'count': c} for n, c in freq.most_common(5)]
    except ImportError:
        pass

    # Dialogue density
    quote_count = sum(text.count(q) for q in ('"', '"', '「', '『', '"'))
    speak_count = sum(text.count(s) for s in ('说道', '说：', '回答', '问道', '道：', '叫道', '笑道', '答道'))
    total_chars = len(text)
    dialogue_density = round((quote_count + speak_count) / max(total_chars / 1000, 1), 1)

    summary_chars = sum(len(s) for s in opening + middle + closing) + 400
    compression_ratio = round(total_chars / max(summary_chars, 1), 1)

    return {
        'source': source_key,
        'title': title or '(untitled)',
        'total_chars': total_chars,
        'paragraph_count': n_paras,
        'opening_sentences': opening,
        'middle_sentences': middle,
        'closing_sentences': closing,
        'main_entities': entities,
        'dialogue_density_per_1k_chars': dialogue_density,
        'head_excerpt': text[:200],
        'tail_excerpt': text[-200:] if total_chars > 400 else '',
        'compression_ratio': compression_ratio,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default="-", help="chapter text file (or '-' for stdin)")
    ap.add_argument("--json", action="store_true", help="output raw JSON instead of pretty print")
    args = ap.parse_args()

    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding="utf-8")

    outline = get_outline(text, source_key=args.input)
    if args.json:
        print(json.dumps(outline, ensure_ascii=False, indent=2))
        return

    # Pretty print
    print(f"title: {outline.get('title')}")
    print(f"total: {outline.get('total_chars')} chars, {outline.get('paragraph_count')} paragraphs")
    print(f"compression: {outline.get('compression_ratio')}x")
    print(f"dialogue density: {outline.get('dialogue_density_per_1k_chars')} markers/1K chars")
    print(f"main entities: {[e['name'] for e in outline.get('main_entities', [])]}")
    print()
    print("--- opening ---")
    for s in outline.get('opening_sentences', []):
        print(f"  · {s}")
    print("--- middle ---")
    for s in outline.get('middle_sentences', []):
        print(f"  · {s}")
    print("--- closing ---")
    for s in outline.get('closing_sentences', []):
        print(f"  · {s}")


if __name__ == "__main__":
    main()
