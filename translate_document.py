from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import fitz  # PyMuPDF

from document_translator import (
    build_layout_docx,
    parse_page_range,
    translate_document,
    translate_document_plain,
)
from docx_layout import collect_units
from docx import Document
from translator import QuotaExhaustedError, RetryExhaustedError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("document-translator")


def main():
    parser = argparse.ArgumentParser(
        description="Translate PDF/DOCX into a layout-preserving Persian Word document "
                    "(figures copied as images, tables translated in place).")
    parser.add_argument("input", help="Input PDF or DOCX")
    parser.add_argument("-o", "--output", help="Output DOCX path")
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("--provider", help="Provider key from config.json")
    parser.add_argument("--resume", action="store_true", help="Resume using the saved translation checkpoint")
    parser.add_argument("--pages", help="Only convert/translate this page range, e.g. 1-10")
    parser.add_argument("--plain", action="store_true",
                        help="Legacy mode: plain Word text without source layout/figures/tables")
    parser.add_argument("--dry-run", action="store_true", help="Build the layout skeleton and list paragraphs without calling the API")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")
    output_path = Path(args.output).resolve() if args.output else input_path.with_name(input_path.stem + "_fa.docx")
    cfg_path = Path(args.config)
    if not args.dry_run and not args.plain and not cfg_path.exists():
        parser.error(f"Config file not found: {cfg_path}. Copy config.example.json to config.json first.")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    provider_name = args.provider or cfg.get("provider", "winkapi")

    pages = None
    if args.pages:
        if input_path.suffix.lower() != ".pdf":
            parser.error("--pages is only supported for PDF input.")
        with fitz.open(str(input_path)) as pdf:
            pages = parse_page_range(args.pages, len(pdf))

    if args.dry_run:
        layout_path = output_path.with_suffix(".layout.docx")
        build_layout_docx(input_path, cfg, layout_path, pages=pages)
        units = collect_units(Document(str(layout_path)))
        figures = "n/a"
        log.info("Dry run successful: %d translatable paragraph(s)/cell(s); skeleton: %s",
                 len(units), layout_path)
        for unit in units[:12]:
            log.info("%s [%s] %s", unit.id, unit.kind, unit.text[:160])
        log.info("Review the skeleton file in Word to check figures/tables, then delete it or translate.")
        return

    try:
        if args.plain:
            if not cfg_path.exists():
                parser.error(f"Config file not found: {cfg_path}. Copy config.example.json to config.json first.")
            translate_document_plain(input_path, output_path, cfg, provider_name, resume=args.resume)
        else:
            translate_document(input_path, output_path, cfg, provider_name,
                               resume=args.resume, pages=pages)
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
