from __future__ import annotations

import re
from dataclasses import dataclass

from .article import split_text


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int


@dataclass(frozen=True)
class Cue:
    """Original article offsets and positions in the unsped audio timeline."""

    start: int
    end: int
    time: float
    end_time: float | None = None


def narration_spans(text: str, limit: int = 1800) -> list[TextSpan]:
    """Use paragraphs as reading units; split long ones near sentence boundaries.

    Keep offsets in the original text, including repeated passages and Unicode.
    The same units drive synthesis, highlighting, and clicking.
    """
    result = []
    cursor = 0
    for paragraph in re.split(r"\n\s*\n", text):
        for chunk in split_text(paragraph, limit):
            start = text.index(chunk, cursor)
            result.append(TextSpan(start, start + len(chunk)))
            cursor = start + len(chunk)
    return result
