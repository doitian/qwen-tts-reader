from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from .app import ReaderApp
from .config import MODEL, MODEL_VOICES, Settings
from .player import MpvPlayer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read web articles aloud with selectable Qwen Audio TTS models."
    )
    parser.add_argument(
        "url", nargs="?", default="", help="Article URL (defaults to the last article)"
    )
    parser.add_argument("--model", choices=MODEL_VOICES, help=f"TTS model (default: {MODEL})")
    parser.add_argument("--voice", help="Voice override (otherwise use the model's default)")
    parser.add_argument(
        "--log",
        type=Path,
        help="Write a playback timing log here, and mpv's own log beside it (or QWEN_READER_LOG)",
    )
    args = parser.parse_args()
    load_dotenv()
    log_file = args.log or (Path(value) if (value := os.getenv("QWEN_READER_LOG")) else None)
    player = None
    if log_file:
        logging.basicConfig(
            filename=log_file,
            filemode="w",
            encoding="utf-8",
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        )
        logging.getLogger("qwen_reader").setLevel(logging.DEBUG)
        player = MpvPlayer(log_file=log_file.with_suffix(".mpv.log"))
    try:
        settings = Settings.from_env()
        if args.model:
            settings = replace(settings, model=args.model, voice="")
        if args.voice:
            settings = replace(settings, voice=args.voice)
    except ValueError as exc:
        parser.error(str(exc))
    ReaderApp(
        settings, args.url, player=player, restore_choices=not (args.model or args.voice)
    ).run()


if __name__ == "__main__":
    main()
