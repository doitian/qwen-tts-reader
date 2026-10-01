from __future__ import annotations

import argparse
from dataclasses import replace

from dotenv import load_dotenv

from .app import ReaderApp
from .config import MODEL, MODEL_VOICES, Settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read web articles aloud with selectable Qwen Audio TTS models."
    )
    parser.add_argument(
        "url", nargs="?", default="", help="Article URL (defaults to the last article)"
    )
    parser.add_argument("--model", choices=MODEL_VOICES, help=f"TTS model (default: {MODEL})")
    parser.add_argument("--voice", help="Voice override (otherwise use the model's default)")
    args = parser.parse_args()
    load_dotenv()
    try:
        settings = Settings.from_env()
        if args.model:
            settings = replace(settings, model=args.model, voice="")
        if args.voice:
            settings = replace(settings, voice=args.voice)
    except ValueError as exc:
        parser.error(str(exc))
    ReaderApp(settings, args.url, restore_choices=not (args.model or args.voice)).run()


if __name__ == "__main__":
    main()
