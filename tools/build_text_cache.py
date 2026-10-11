"""Build a {source text: translation} cache from an old (v2) checkpoint.

Useful to re-use translations that were already paid for when the extraction
strategy changes (e.g. moving from the plain pipeline to the layout pipeline).

Usage:
    python tools/build_text_cache.py SAI_fa.docx.checkpoint.json cache.json [--pdf SAI.pdf]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_translator import extract_pdf_targets


def norm(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("¶", "'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--pdf", help="Original PDF, needed to recover source texts of v2 checkpoints")
    args = parser.parse_args()

    state = json.loads(Path(args.checkpoint).read_text(encoding="utf-8"))
    translations = state.get("translations", {})
    sources: dict[str, str] = {}
    if args.pdf:
        for target in extract_pdf_targets(Path(args.pdf)):
            sources[target["id"]] = target["text"]
    cache: dict[str, str] = {}
    missing = 0
    for item_id, translation in translations.items():
        source = sources.get(item_id)
        if not source:
            missing += 1
            continue
        cache[norm(source)] = translation
    Path(args.output).write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(cache)} pairs ({missing} ids without source text) to {args.output}")


if __name__ == "__main__":
    main()
