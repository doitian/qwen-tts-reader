from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from .app import ReaderApp
from .article import STDIN
from .config import MODEL, MODEL_VOICES, Settings
from .player import MpvPlayer
from .synthesis import purge_cache


def read_stdin() -> str:
    """Read piped Markdown, then give the TUI the terminal as its keyboard."""
    data = sys.stdin.buffer.read()
    if not sys.stdin.isatty():
        # Textual reads keys from file descriptor 0; on Windows dup2 also sets the std handle.
        terminal = os.open("CONIN$" if os.name == "nt" else "/dev/tty", os.O_RDWR)
        os.dup2(terminal, 0)
        os.close(terminal)
    return data.decode("utf-8-sig")


def purge(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="qwen-reader purge-cache",
        description="Delete cached speech. Bookmarks and the saved model choice are kept.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Also delete bookmarks and the saved model choice"
    )
    args = parser.parse_args(argv)
    load_dotenv()
    cache_dir = Settings.from_env().cache_dir
    removed, freed, busy = purge_cache(cache_dir, state=args.all)
    print(f"Removed {removed} files ({freed / 1_000_000:.1f} MB) from {cache_dir}.")
    if busy:
        print(f"Kept {len(busy)} files in use; close the reader and run this again.")


def main() -> None:
    # The URL is an optional positional, which argparse subcommands cannot share.
    if sys.argv[1:2] == ["purge-cache"]:
        purge(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(
        description="Read web articles or Markdown aloud with selectable Qwen Audio TTS models.",
        epilog="Run `qwen-reader purge-cache` to delete cached speech (--help for options).",
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="",
        help="Article URL, Markdown file, or - for standard input (defaults to the last article)",
    )
    parser.add_argument("--model", choices=MODEL_VOICES, help=f"TTS model (default: {MODEL})")
    parser.add_argument("--voice", help="Voice override (otherwise use the model's default)")
    parser.add_argument(
        "--log",
        type=Path,
        help="Write a playback timing log here, and mpv's own log beside it (or QWEN_READER_LOG)",
    )
    args = parser.parse_args()
    stdin = None
    if args.url.strip() == STDIN:
        try:
            stdin = read_stdin()
        except UnicodeDecodeError:
            parser.error("standard input isn't UTF-8 text")
        except OSError as exc:
            parser.error(f"no terminal for keyboard input after reading standard input: {exc}")
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
        settings,
        args.url,
        player=player,
        restore_choices=not (args.model or args.voice),
        stdin=stdin,
    ).run()


if __name__ == "__main__":
    main()
