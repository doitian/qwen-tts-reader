from __future__ import annotations

import re

# Seconds per character before any audio is known; CJK speech covers far fewer characters.
LATIN_RATE = 0.065
CJK_RATE = 0.24
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


class Timeline:
    """Whole-article time, from actual durations where known and estimates elsewhere.

    The playlist covers units from `anchor` onward. Its start time is fixed when it is
    anchored, so positions inside it don't jump as estimates for other units improve.
    Earlier units are measured backwards from it, keeping times continuous around the
    playlist start, where seeking happens.
    """

    def __init__(self, lengths: list[int], known: dict[int, float] | None = None, text: str = ""):
        self.lengths = lengths
        self.known = dict(known or {})
        cjk = len(CJK.findall(text))
        self.default_rate = CJK_RATE if text and cjk > len(text) / 3 else LATIN_RATE
        self.anchor = 0
        self.anchor_time = 0.0

    def rate(self) -> float:
        characters = sum(self.lengths[unit] for unit in self.known)
        if not characters:
            return self.default_rate
        return sum(self.known.values()) / characters

    def duration(self, unit: int) -> float:
        known = self.known.get(unit)
        return known if known is not None else self.lengths[unit] * self.rate()

    def set_anchor(self, unit: int) -> None:
        # Measured from the current anchor, so times stay continuous across restarts.
        self.anchor_time = self.start(unit)
        self.anchor = unit

    def start(self, unit: int) -> float:
        if unit >= self.anchor:
            return self.anchor_time + sum(self.duration(i) for i in range(self.anchor, unit))
        if unit == 0:
            return 0.0
        return max(0.0, self.anchor_time - sum(self.duration(i) for i in range(unit, self.anchor)))

    def total(self) -> float:
        return self.start(len(self.lengths))

    def exact(self, end: int | None = None) -> bool:
        """True if every unit before `end` (default: all) has a known duration."""
        return all(unit in self.known for unit in range(len(self.lengths) if end is None else end))

    def locate(self, time: float) -> tuple[int, float]:
        """The unit playing at `time` and the offset into it."""
        if not self.lengths:
            return 0, 0.0
        if time < self.anchor_time and self.anchor:
            unit = self.anchor - 1
            while unit > 0 and self.start(unit) > time:
                unit -= 1
            return unit, max(0.0, time - self.start(unit))
        unit, start = self.anchor, self.anchor_time
        while unit < len(self.lengths) - 1 and start + self.duration(unit) <= time:
            start += self.duration(unit)
            unit += 1
        return unit, max(0.0, time - start)
