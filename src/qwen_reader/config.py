from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

MODEL = "qwen-audio-3.1-tts-flash"
DEFAULT_VOICE = "longanlingxin_v3.1"
MODEL_VOICES = {
    MODEL: DEFAULT_VOICE,
    "qwen-audio-3.0-tts-plus": "longanlingxin",
    "qwen-audio-3.0-tts-flash": "longanhuan_v3.6",
}


def default_save_dir() -> Path:
    downloads = Path.home() / "Downloads"
    return downloads if downloads.is_dir() else Path.cwd()


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    endpoint: str = ""
    model: str = MODEL
    voice: str = ""
    cache_dir: Path = Path.home() / ".cache" / "qwen-tts-reader"
    defuddle_key: str = field(default="", repr=False)
    save_dir: Path = field(default_factory=lambda: default_save_dir())
    chunk_chars: int = 1800

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            api_key=os.getenv("QWEN_TTS_API_KEY", "").strip(),
            endpoint=os.getenv("QWEN_TTS_ENDPOINT", "").strip(),
            model=os.getenv("QWEN_TTS_MODEL", MODEL).strip() or MODEL,
            voice=os.getenv("QWEN_TTS_VOICE", "").strip(),
            cache_dir=Path(
                os.getenv("QWEN_READER_CACHE_DIR")
                or Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "qwen-tts-reader"
            ).expanduser(),
            defuddle_key=os.getenv("DEFUDDLE_API_KEY", "").strip(),
            save_dir=Path(os.getenv("QWEN_READER_SAVE_DIR") or default_save_dir()).expanduser(),
        )

    def validate_tts(self) -> None:
        if self.model not in MODEL_VOICES:
            raise ValueError(
                f"Unknown model {self.model!r}. Choose one of: {', '.join(MODEL_VOICES)}"
            )
        missing = [
            name
            for name, value in (
                ("QWEN_TTS_ENDPOINT", self.endpoint),
                ("QWEN_TTS_API_KEY", self.api_key),
            )
            if not value.strip()
        ]
        if missing:
            raise ValueError(
                f"Set {' and '.join(missing)} in .env or your environment, then restart."
            )
        try:
            url = urlsplit(self.endpoint)
            _ = url.port
        except ValueError as exc:
            raise ValueError("QWEN_TTS_ENDPOINT must be a valid WSS or HTTPS URL.") from exc
        if url.scheme not in ("wss", "https") or not url.hostname or url.username or url.password:
            raise ValueError("QWEN_TTS_ENDPOINT must be a WSS or HTTPS URL without credentials.")

    @property
    def voice_id(self) -> str:
        return self.voice or MODEL_VOICES.get(self.model, "")
