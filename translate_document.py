from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from document_translator import extract_targets, translate_document
from translator import QuotaExhaustedError, RetryExhaustedError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("document-translator")


def main():
    parser = argparse.ArgumentParser(description="Translate PDF/DOCX text into a plain Persian Word document.")
    parser.add_argument("input", help="Input PDF or DOCX")
    parser.add_argument("-o", "--output", help="Output DOCX path")
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("--provider", help="Provider key from config.json")
    parser.add_argument("--resume", action="store_true", help="Resume using the saved translation checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Extract/count paragraphs without calling the API")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")
    output_path = Path(args.output).resolve() if args.output else input_path.with_name(input_path.stem + "_fa.docx")
    cfg_path = Path(args.config)
    if not args.dry_run and not cfg_path.exists():
        parser.error(f"Config file not found: {cfg_path}. Copy config.example.json to config.json first.")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    provider_name = args.provider or cfg.get("provider", "winkapi")

    if args.dry_run:
        targets = extract_targets(input_path)
        log.info("Dry run successful: %d paragraphs extracted; no API calls made.", len(targets))
        for item in targets[:8]:
            log.info("%s", item["text"][:180])
        return

    try:
        translate_document(input_path, output_path, cfg, provider_name, resume=args.resume)
    except QuotaExhaustedError as exc:
        log.error("Provider quota exhausted; completed translations are checkpointed. %s", exc)
        if exc.retry_after_seconds:
            log.error("Provider retry delay: about %.1f hours", exc.retry_after_seconds / 3600)
        sys.exit(3)
    except RetryExhaustedError as exc:
        log.error("Paused safely after transient API failures. Use --resume to continue. %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
