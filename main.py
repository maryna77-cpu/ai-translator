from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

from checkpoint import load as load_checkpoint
from checkpoint import save as save_checkpoint
from providers import build_provider
from srt import initialize_output, parse_srt, recover_output
from translator import QuotaExhaustedError, RetryExhaustedError, process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("subtitle-translator")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Context-aware English -> Persian SRT translator"
    )
    parser.add_argument("input", help="Input SRT file")
    parser.add_argument("-o", "--output", help="Output SRT file")
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("--provider", help="Provider name from config.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Parse/check configuration only")
    args = parser.parse_args()

    cfg = load_json(args.config)
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(
        args.output or str(Path(input_path).with_suffix("")) + "_fa.srt"
    )
    checkpoint_path = output_path + ".checkpoint.json"

    subtitles = parse_srt(input_path)
    input_hash = file_sha256(input_path)
    provider_name = args.provider or cfg["provider"]

    log.info("Input: %s", input_path)
    log.info("Subtitles: %d", len(subtitles))
    log.info("Provider: %s", provider_name)
    log.info("Output: %s", output_path)

    provider = build_provider(cfg, provider_name)

    glossary_path = cfg.get("glossary_file", "glossary.json")
    glossary = load_json(glossary_path) if os.path.exists(glossary_path) else {}

    if args.dry_run:
        log.info("Dry run successful.")
        return

    checkpoint = load_checkpoint(checkpoint_path)
    existing_numbers = recover_output(output_path)

    if checkpoint:
        old_hash = checkpoint.get("input_sha256")
        if old_hash and old_hash != input_hash:
            raise RuntimeError(
                "The input SRT has changed since the checkpoint was created. "
                "Use a new output filename or remove the old checkpoint/output."
            )

    if not checkpoint and not existing_numbers:
        initialize_output(output_path)

    completed = set(checkpoint.get("completed", [])) | existing_numbers
    checkpoint_state = {
        "version": 3,
        "status": "ready",
        "input": input_path,
        "input_sha256": input_hash,
        "output": output_path,
        "provider": provider_name,
        "completed": sorted(completed),
        "completed_count": len(completed),
    }
    save_checkpoint(checkpoint_path, checkpoint_state)

    log.info("Already completed: %d/%d", len(completed), len(subtitles))
    try:
        process(
            provider,
            subtitles,
            completed,
            output_path,
            checkpoint_path,
            checkpoint_state,
            glossary,
            cfg,
        )
    except QuotaExhaustedError as exc:
        log.error("Provider quota is exhausted; the failed batch was not committed.")
        if exc.retry_after_seconds:
            hours = exc.retry_after_seconds / 3600
            log.error("Provider requested retry after about %.1f hours.", hours)
        log.error("Run the same command with --resume after the provider quota resets or after changing provider/model.")
        sys.exit(3)
    except RetryExhaustedError:
        log.error("Safe pause: no subtitle from the failed batch was committed after the last successful checkpoint.")
        log.error("Run the same command with --resume when the provider becomes available.")
        sys.exit(2)

    log.info("Translation completed: %s", output_path)
    log.info("After verifying the SRT, you can archive/delete the checkpoint and memory files.")


if __name__ == "__main__":
    main()
