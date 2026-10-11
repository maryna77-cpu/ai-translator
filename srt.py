from __future__ import annotations

from dataclasses import dataclass
import os
import re


@dataclass(frozen=True)
class Subtitle:
    number: int
    timestamp: str
    text: str


BLOCK_RE = re.compile(
    r"(?ms)^\s*(\d+)\s*\n([^\n]+-->[^\n]+)\s*\n(.*?)(?=\n\s*\n|\Z)"
)


def parse_srt(path: str) -> list[Subtitle]:
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    result: list[Subtitle] = []
    for match in BLOCK_RE.finditer(raw):
        result.append(
            Subtitle(
                number=int(match.group(1)),
                timestamp=match.group(2).strip(),
                text=match.group(3).strip(),
            )
        )
    if not result:
        raise ValueError("No valid SRT subtitle blocks found.")
    return result


def render(sub: Subtitle) -> str:
    return f"{sub.number}\n{sub.timestamp}\n{sub.text}\n\n"


def _is_complete_block(block: str) -> bool:
    lines = block.strip("\n").split("\n")
    if len(lines) < 3:
        return False
    if not re.fullmatch(r"\d+", lines[0].strip()):
        return False
    return "-->" in lines[1]


def recover_output(path: str) -> set[int]:
    """Remove an incomplete final SRT block after a crash and return completed IDs."""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().replace("\r\n", "\n").replace("\r", "\n")

    chunks = raw.split("\n\n")
    complete = [c for c in chunks if _is_complete_block(c)]
    numbers = set()
    for chunk in complete:
        number = int(chunk.split("\n", 1)[0].strip())
        numbers.add(number)

    normalized = "\n\n".join(c.strip("\n") for c in complete)
    if normalized:
        normalized += "\n\n"

    if normalized != raw:
        tmp = path + ".recover.tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(normalized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    return numbers


def initialize_output(path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n"):
        pass


def append_subtitle(path: str, subtitle: Subtitle) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(render(subtitle))
        f.flush()
        os.fsync(f.fileno())
