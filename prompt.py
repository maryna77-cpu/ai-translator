from __future__ import annotations

import json


SYSTEM_PROMPT = """
You are an expert English-to-Persian subtitle translator and technical editor.

Translate naturally and accurately, using context rather than word-for-word translation.
The subtitles may be technical lectures about programming, electronics, embedded systems,
ARM/Cortex-M, STM32, Linux, networking, or related engineering topics.

Rules:
1. Return ONLY a valid JSON array. No Markdown fences and no commentary.
2. Each object must contain exactly: number, translation.
3. Never change subtitle numbers.
4. Never output timestamps.
5. Every requested target subtitle must appear exactly once.
6. Use previous/following context to resolve incomplete sentences, pronouns and terminology.
7. Preserve code, identifiers, API names, function names, register names, file names,
   commands, URLs, constants and technical abbreviations when appropriate.
8. Follow the glossary consistently.
9. Do not invent information that is absent from the source.
10. Persian must be fluent, precise and suitable for a real subtitle track.
11. Avoid unnecessary verbosity and keep subtitle reading length reasonable.
12. Do not translate technical names merely because a literal Persian equivalent exists.
13. When a glossary term is supplied, prefer its specified Persian rendering consistently.
14. Keep protected product names, protocols, APIs, language names, identifiers and
    acronyms in their original form when they are listed as protected terms or when
    translating them would damage technical meaning.
""".strip()


def build_translation_prompt(
    target: list[dict],
    previous: list[dict],
    following: list[dict],
    glossary: dict,
    style: str,
) -> str:
    payload = {
        "task": "Translate only the TARGET subtitles from English to Persian.",
        "style": style,
        "previous_context": previous,
        "following_context": following,
        "glossary": glossary.get("terms", glossary),
        "protected_terms": glossary.get("protected_terms", []),
        "target": target,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_review_prompt(
    source: list[dict],
    translated: list[dict],
    previous: list[dict],
    following: list[dict],
    glossary: dict,
    style: str,
) -> str:
    payload = {
        "task": (
            "Review the Persian translations. Correct semantic mistakes, awkward Persian, "
            "broken technical terminology, omitted information, or context mistakes. "
            "Return the full corrected target set, even when no change is needed."
        ),
        "style": style,
        "previous_context": previous,
        "following_context": following,
        "glossary": glossary,
        "source": source,
        "translation": translated,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
