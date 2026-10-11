from __future__ import annotations

import logging
import random
import time

from checkpoint import save
from prompt import build_review_prompt, build_translation_prompt
from providers import BaseProvider, ProviderError
from srt import Subtitle, append_subtitle
from translation_memory import TranslationMemory
from validator import parse_and_validate

log = logging.getLogger("subtitle-translator")


def estimate_tokens(text: str, chars_per_token: float, safety_factor: float) -> int:
    return int((len(text) / chars_per_token) * safety_factor) + 80


def looks_like_context_error(exc: Exception) -> bool:
    text = str(exc).lower()
    signals = (
        "context length",
        "context window",
        "too many tokens",
        "maximum context",
        "max context",
        "prompt is too long",
        "request too large",
        "payload too large",
        "token limit",
    )
    return any(s in text for s in signals)


def should_split_on_error(exc: Exception) -> bool:
    if looks_like_context_error(exc):
        return True
    # Bad JSON / missing items are common with overloaded local models.
    text = str(exc).lower()
    return any(s in text for s in ("subtitle mismatch", "json", "empty translation"))


def build_batches(subtitles: list[Subtitle], completed: set[int], cfg: dict):
    t = cfg["translation"]
    max_items = int(t.get("batch_max_items", 20))
    max_chars = int(t.get("batch_max_chars", 12000))
    context_window = int(t.get("context_window_tokens", 16384))
    max_output = int(t.get("max_output_tokens", 5000))
    chars_per_token = float(t.get("chars_per_token_estimate", 3.5))
    safety = float(t.get("token_safety_factor", 1.25))
    safe_input = max(400, context_window - max_output - 900)

    index = 0
    n = len(subtitles)
    while index < n:
        while index < n and subtitles[index].number in completed:
            index += 1
        if index >= n:
            return

        batch = []
        chars = 0
        j = index
        while j < n and subtitles[j].number not in completed:
            candidate = subtitles[j]
            combined = "\n".join(s.text for s in batch + [candidate])
            if batch and (
                len(batch) >= max_items
                or chars + len(candidate.text) > max_chars
                or estimate_tokens(combined, chars_per_token, safety) > safe_input
            ):
                break
            batch.append(candidate)
            chars += len(candidate.text)
            j += 1
            if len(batch) == 1 and estimate_tokens(candidate.text, chars_per_token, safety) > safe_input:
                break

        if not batch:
            raise RuntimeError("Unable to create a safe translation batch.")
        yield batch
        index = j


def context_for(subtitles: list[Subtitle], batch: list[Subtitle], before_n: int, after_n: int):
    pos = {s.number: i for i, s in enumerate(subtitles)}
    first = pos[batch[0].number]
    last = pos[batch[-1].number]
    previous = subtitles[max(0, first - before_n):first]
    following = subtitles[last + 1:last + 1 + after_n]
    return previous, following


def flatten_glossary(glossary: dict) -> dict[str, str]:
    """Flatten category-based glossary JSON into term -> preferred Persian translation."""
    result: dict[str, str] = {}
    categories = glossary.get("categories", glossary) if isinstance(glossary, dict) else {}
    for category_name, category_terms in categories.items():
        if category_name in {"metadata", "do_not_translate"}:
            continue
        if isinstance(category_terms, dict):
            for term, translation in category_terms.items():
                if isinstance(term, str) and isinstance(translation, str):
                    result[term] = translation
    return result


def select_relevant_glossary(glossary: dict, texts: list[str], max_terms: int = 80) -> dict:
    """Select only glossary entries relevant to the current batch/context."""
    terms = flatten_glossary(glossary)
    corpus = "\n".join(texts).lower()
    matched = []
    for term, translation in terms.items():
        if term.lower() in corpus:
            matched.append((len(term), term, translation))
    matched.sort(key=lambda x: (-x[0], x[1].lower()))
    selected = {term: translation for _, term, translation in matched[:max_terms]}

    protected = []
    for term in glossary.get("do_not_translate", []):
        if isinstance(term, str) and term.lower() in corpus:
            protected.append(term)

    return {
        "terms": selected,
        "protected_terms": protected,
    }


class RetryExhaustedError(RuntimeError):
    """Raised when a transient API error persists after all configured retries."""

    def __init__(self, message: str, *, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


class QuotaExhaustedError(RuntimeError):
    """Raised when the provider reports a non-retryable account/project quota."""

    def __init__(self, message: str, *, last_error: Exception | None = None, retry_after_seconds: int | None = None):
        super().__init__(message)
        self.last_error = last_error
        self.retry_after_seconds = retry_after_seconds


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            return int(status)
        except (TypeError, ValueError):
            pass
    return None


def _error_text(exc: Exception) -> str:
    return str(exc).lower()


def is_daily_quota_error(exc: Exception) -> bool:
    text = _error_text(exc)
    # Gemini native API / WinkAPI / OpenAI-compatible variants.
    return any(signal in text for signal in (
        "generate_content_free_tier_requests",
        "generaterequestsperdayperproject-freetier",
        "generate requests per day",
        "daily quota",
        "quota exceeded for metric",
        "you exceeded your current quota",
        "insufficient_quota",
    ))


def is_rate_limit_error(exc: Exception) -> bool:
    text = _error_text(exc)
    code = str(getattr(exc, "error_code", "") or "").lower()
    return any(signal in text or signal in code for signal in (
        "rate_limit_exceeded",
        "rate limit",
        "too many requests",
        "temporarily rate limited",
    ))


def is_transient_api_error(exc: Exception) -> bool:
    """Return True only for errors that are normally safe to retry."""
    if is_daily_quota_error(exc):
        return False

    status = _status_code(exc)
    if status in {408, 409, 500, 502, 503, 504}:
        return True
    if status == 429:
        return is_rate_limit_error(exc)

    text = _error_text(exc)
    return any(signal in text for signal in (
        "503 unavailable",
        "service unavailable",
        "upstream_unavailable",
        "upstream_service_unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
    ))


def _retry_after_seconds(exc: Exception) -> int | None:
    # Prefer Retry-After headers when exposed by an OpenAI-compatible exception.
    headers = getattr(exc, "headers", None)
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value:
            try:
                return max(0, int(float(value)))
            except (TypeError, ValueError):
                pass

    # Gemini may expose: "retry in 40643s" or "retryDelay ... 40643s".
    import re
    text = str(exc)
    matches = re.findall(r"(?:retry(?: in|after|delay)[^0-9]{0,30})(\d+(?:\.\d+)?)\s*s", text, flags=re.I)
    if matches:
        return max(0, int(float(matches[0])))
    return None


def api_with_retries(provider, system_prompt, user_prompt, cfg):
    t = cfg["translation"]
    retries = max(0, int(t.get("retry_count", 5)))
    base = max(0.1, float(t.get("retry_base_seconds", 10)))
    max_delay = max(base, float(t.get("retry_max_seconds", 180)))
    jitter = max(0.0, float(t.get("retry_jitter_seconds", 3)))
    temperature = float(t.get("temperature", 0.2))
    max_output = int(t.get("max_output_tokens", 5000))
    last: Exception | None = None

    for attempt in range(retries + 1):
        try:
            return provider.generate(
                system_prompt,
                user_prompt,
                temperature=temperature,
                max_output_tokens=max_output,
            )
        except Exception as exc:
            last = exc

            if is_daily_quota_error(exc):
                retry_after = _retry_after_seconds(exc)
                raise QuotaExhaustedError(
                    "Provider daily/project quota is exhausted. No retry was attempted.",
                    last_error=exc,
                    retry_after_seconds=retry_after,
                ) from exc

            if not is_transient_api_error(exc):
                raise

            if attempt >= retries:
                break

            provider_retry_after = _retry_after_seconds(exc) if is_rate_limit_error(exc) else None
            if provider_retry_after is not None:
                delay = min(max_delay, max(base, provider_retry_after))
            else:
                delay = min(max_delay, base * (2 ** attempt))
                if jitter:
                    delay += random.uniform(0, jitter)

            status = _status_code(exc)
            status_text = f"HTTP {status}" if status else "transient API error"
            log.warning(
                "Transient API error (%s): %s | retry %d/%d in %.1fs",
                status_text,
                exc,
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)

    raise RetryExhaustedError(
        f"Transient API error persisted after {retries} retries. "
        "The current batch was not written; resume can safely continue from the last checkpoint.",
        last_error=last,
    ) from last

def translate_once(provider: BaseProvider, batch, previous, following, glossary, style, cfg):
    source = [{"number": s.number, "timestamp": s.timestamp, "text": s.text} for s in batch]
    prev = [{"number": s.number, "text": s.text} for s in previous]
    foll = [{"number": s.number, "text": s.text} for s in following]
    relevant_glossary = select_relevant_glossary(
        glossary,
        [x["text"] for x in source] + [x["text"] for x in prev] + [x["text"] for x in foll],
        int(cfg["translation"].get("glossary_max_terms_per_batch", 80)),
    )
    prompt = build_translation_prompt(source, prev, foll, relevant_glossary, style)
    result = api_with_retries(provider, __import__("prompt").SYSTEM_PROMPT, prompt, cfg)
    translations = parse_and_validate(result.content, [s.number for s in batch])
    return source, prev, foll, translations


def review_once(provider, source, translations, previous, following, glossary, style, cfg):
    relevant_glossary = select_relevant_glossary(
        glossary,
        [x["text"] for x in source] +
        [x["translation"] for x in translations] +
        [x["text"] for x in previous] +
        [x["text"] for x in following],
        int(cfg["translation"].get("glossary_max_terms_per_batch", 80)),
    )
    prompt = build_review_prompt(source, translations, previous, following, relevant_glossary, style)
    result = api_with_retries(provider, __import__("prompt").SYSTEM_PROMPT, prompt, cfg)
    return parse_and_validate(result.content, [x["number"] for x in source])


def translate_batch_resilient(provider, batch, subtitles, glossary, cfg):
    t = cfg["translation"]
    previous, following = context_for(
        subtitles,
        batch,
        int(t.get("context_before", 3)),
        int(t.get("context_after", 3)),
    )
    style = t.get("style", "Natural, technically precise Persian")

    try:
        source, prev, foll, translations = translate_once(
            provider, batch, previous, following, glossary, style, cfg
        )
    except Exception as exc:
        if len(batch) > 1 and should_split_on_error(exc):
            midpoint = max(1, len(batch) // 2)
            log.warning("Splitting batch %d-%d after model/output failure.", batch[0].number, batch[-1].number)
            left = translate_batch_resilient(provider, batch[:midpoint], subtitles, glossary, cfg)
            right = translate_batch_resilient(provider, batch[midpoint:], subtitles, glossary, cfg)
            return left + right
        raise

    review_cfg = cfg.get("review", {})
    if review_cfg.get("enabled", False):
        review_retries = int(review_cfg.get("retry_count", 1))
        original = translations
        for attempt in range(review_retries + 1):
            try:
                translations = review_once(
                    provider, source, translations, prev, foll, glossary, style, cfg
                )
                break
            except Exception as exc:
                translations = original
                if attempt == review_retries:
                    log.warning("Review failed; keeping first-pass translation: %s", exc)
                else:
                    time.sleep(1.5 * (attempt + 1))

    return translations


def process(
    provider,
    subtitles,
    completed,
    output_path,
    checkpoint_path,
    checkpoint_state,
    glossary,
    cfg,
):
    tm_cfg = cfg.get("translation_memory", {})
    tm = TranslationMemory(tm_cfg.get("path", output_path + ".memory.sqlite")) if tm_cfg.get("enabled", False) else None

    try:
        done = len(completed)
        total = len(subtitles)
        for batch in build_batches(subtitles, completed, cfg):
            log.info(
                "Batch %d-%d (%d subtitles), progress %d/%d",
                batch[0].number, batch[-1].number, len(batch), done, total,
            )

            checkpoint_state["status"] = "translating"
            checkpoint_state["current_batch"] = [batch[0].number, batch[-1].number]
            save(checkpoint_path, checkpoint_state)

            translations = None
            if tm:
                tcfg = cfg["translation"]
                previous, following = context_for(
                    subtitles, batch,
                    int(tcfg.get("context_before", 3)),
                    int(tcfg.get("context_after", 3)),
                )
                key_payload = {
                    "version": 2,
                    "provider": getattr(provider, "name", provider.__class__.__name__),
                    "model": getattr(provider, "model", None),
                    "batch": [{"number": s.number, "text": s.text} for s in batch],
                    "previous": [{"number": s.number, "text": s.text} for s in previous],
                    "following": [{"number": s.number, "text": s.text} for s in following],
                    "glossary": glossary,
                    "style": tcfg.get("style", "Natural, technically precise Persian"),
                }
                translations = tm.get(key_payload)

            if translations is None:
                try:
                    translations = translate_batch_resilient(provider, batch, subtitles, glossary, cfg)
                except QuotaExhaustedError as exc:
                    checkpoint_state["status"] = "paused_quota"
                    checkpoint_state["last_error"] = str(exc)
                    checkpoint_state["last_error_detail"] = str(exc.last_error) if exc.last_error else None
                    checkpoint_state["retry_after_seconds"] = exc.retry_after_seconds
                    checkpoint_state["current_batch"] = [batch[0].number, batch[-1].number]
                    save(checkpoint_path, checkpoint_state)
                    log.error(
                        "Translation paused: provider quota is exhausted. "
                        "No retry was attempted. Last error: %s",
                        exc.last_error or exc,
                    )
                    raise
                except RetryExhaustedError as exc:
                    checkpoint_state["status"] = "paused"
                    checkpoint_state["last_error"] = str(exc)
                    checkpoint_state["last_error_detail"] = str(exc.last_error) if exc.last_error else None
                    checkpoint_state["current_batch"] = [batch[0].number, batch[-1].number]
                    save(checkpoint_path, checkpoint_state)
                    log.error(
                        "Translation paused safely after repeated transient API failures. "
                        "Completed subtitles remain intact. Run the same command with --resume to continue. "
                        "Last error: %s",
                        exc.last_error or exc,
                    )
                    raise
                if tm:
                    tm.put(key_payload, translations)
            else:
                log.info("Translation-memory hit for batch %d-%d", batch[0].number, batch[-1].number)

            source_by_number = {s.number: s for s in batch}
            for item in translations:
                source = source_by_number[item["number"]]
                append_subtitle(
                    output_path,
                    Subtitle(source.number, source.timestamp, item["translation"]),
                )
                completed.add(source.number)
                done += 1

                checkpoint_state["completed"] = sorted(completed)
                checkpoint_state["completed_count"] = done
                checkpoint_state["status"] = "translating"
                save(checkpoint_path, checkpoint_state)

                # Each subtitle is durable before moving to the next one.
                log.debug("Saved subtitle %d (%d/%d)", source.number, done, total)

        checkpoint_state["status"] = "completed"
        checkpoint_state.pop("current_batch", None)
        checkpoint_state.pop("last_error", None)
        checkpoint_state.pop("last_error_detail", None)
        save(checkpoint_path, checkpoint_state)
    finally:
        if tm:
            tm.close()
