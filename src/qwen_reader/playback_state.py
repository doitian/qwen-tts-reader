from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, astuple, dataclass
from pathlib import Path

from .article import normalize_url


@dataclass(frozen=True)
class Bookmark:
    url: str
    narration: str
    position: float
    speed: float = 1.0
    paused: bool = False
    completed: bool = False
    model: str = ""
    voice: str = ""
    configuration: str = ""
    # The reading unit and offset into it; position alone may be an estimate. -1: older file.
    unit: int = -1
    unit_offset: float = 0.0


@dataclass(frozen=True)
class ModelChoice:
    """The model and voice last applied in the TUI, valid while startup config is unchanged."""

    model: str
    voice: str
    configuration: str


class PlaybackState:
    """Atomic, private progress file; credentials and article text aren't stored."""

    def __init__(self, path: Path):
        self.path = path
        self.last_url = ""
        self.last_narration = ""
        self.bookmarks: dict[str, Bookmark] = {}
        self.choice: ModelChoice | None = None
        try:
            data = json.loads(path.read_text())
            if data.get("version") != 1:
                return
            for entry in data.get("bookmarks", []):
                try:
                    bookmark = Bookmark(**entry)
                    if (
                        normalize_url(bookmark.url) != bookmark.url
                        or not isinstance(bookmark.narration, str)
                        or not bookmark.narration
                        or not math.isfinite(bookmark.position)
                        or bookmark.position < 0
                        or not math.isfinite(bookmark.speed)
                        or not 0.5 <= bookmark.speed <= 3
                        or type(bookmark.paused) is not bool
                        or type(bookmark.completed) is not bool
                        or type(bookmark.unit) is not int
                        or bookmark.unit < -1
                        or not isinstance(bookmark.unit_offset, int | float)
                        or not math.isfinite(bookmark.unit_offset)
                        or bookmark.unit_offset < 0
                        or not all(
                            isinstance(value, str)
                            for value in (bookmark.model, bookmark.voice, bookmark.configuration)
                        )
                    ):
                        continue
                except (TypeError, ValueError, AttributeError):
                    continue
                self.bookmarks[self.key(bookmark.url, bookmark.narration)] = bookmark
            choice = data.get("choice")
            if isinstance(choice, dict):
                try:
                    value = ModelChoice(**choice)
                    if all(isinstance(field, str) for field in astuple(value)):
                        self.choice = value
                except TypeError:
                    pass
            last_url = data.get("last_url", "")
            if any(bookmark.url == last_url for bookmark in self.bookmarks.values()):
                self.last_url = last_url
                self.last_narration = str(data.get("last_narration", ""))
        except (OSError, ValueError, TypeError, AttributeError):
            pass

    @staticmethod
    def key(url: str, narration: str) -> str:
        return hashlib.sha256(f"{url}\0{narration}".encode()).hexdigest()

    def get(self, url: str, narration: str) -> Bookmark | None:
        return self.bookmarks.get(self.key(url, narration))

    def save(self, bookmark: Bookmark) -> None:
        self.bookmarks[self.key(bookmark.url, bookmark.narration)] = bookmark
        self.last_url = bookmark.url
        self.last_narration = bookmark.narration
        self.write()

    def save_choice(self, choice: ModelChoice) -> None:
        self.choice = choice
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, prefix=".playback-", suffix=".tmp", delete=False
            ) as output:
                temporary = Path(output.name)
                json.dump(
                    {
                        "version": 1,
                        "last_url": self.last_url,
                        "last_narration": self.last_narration,
                        "bookmarks": [asdict(value) for value in self.bookmarks.values()],
                        "choice": asdict(self.choice) if self.choice else None,
                    },
                    output,
                )
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
