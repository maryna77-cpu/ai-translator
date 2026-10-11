from __future__ import annotations

import json
import re


def _clean_model_json(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_and_validate(raw: str, expected_numbers: list[int]) -> list[dict]:
    data = json.loads(_clean_model_json(raw))
    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON array.")

    expected = list(expected_numbers)
    got: dict[int, str] = {}

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Model returned an invalid item.")
        if set(item.keys()) != {"number", "translation"}:
            raise ValueError("Each output item must contain only number and translation.")
        number = int(item["number"])
        translation = str(item["translation"]).strip()
        if not translation:
            raise ValueError(f"Empty translation for subtitle {number}.")
        if number in got:
            raise ValueError(f"Duplicate subtitle {number}.")
        got[number] = translation

    if set(got) != set(expected) or len(got) != len(expected):
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise ValueError(f"Subtitle mismatch. missing={missing}, extra={extra}")

    return [{"number": n, "translation": got[n]} for n in expected]
