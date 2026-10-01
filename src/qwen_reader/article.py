from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from typing import NamedTuple
from urllib.parse import urldefrag, urlsplit

import httpx
import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token


@dataclass(frozen=True)
class Article:
    url: str
    title: str
    author: str
    markdown: str
    text: str


def normalize_url(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Paste an article URL first.")
    if "://" not in value:
        value = "https://" + value
    try:
        url = urlsplit(value)
        _ = url.port
    except ValueError as exc:
        raise ValueError("Enter a valid HTTP or HTTPS article URL.") from exc
    if url.scheme not in ("http", "https") or not url.hostname or re.search(r"\s", value):
        raise ValueError("Enter a valid HTTP or HTTPS article URL.")
    if url.username or url.password:
        raise ValueError("Use a public article URL without embedded credentials.")
    return urldefrag(value)[0]


MARKS = {"strong": "strong", "em": "em", "s": "strike", "link": "link"}
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def markdown_parser() -> MarkdownIt:
    return MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")


class Run(NamedTuple):
    text: str
    marks: frozenset[str]
    href: str = ""


def inline_runs(children: list[Token]) -> list[Run]:
    """The spoken text of one inline block, as runs tagged with their formatting and link.

    Joining the runs gives exactly the block's line in the speech text, so the
    display can show the same characters at the same offsets.
    """
    chars: list[tuple[str, frozenset[str], str]] = []
    marks: set[str] = set()
    links: list[str] = []
    for child in children:
        kind = child.type.removesuffix("_open").removesuffix("_close")
        if kind in MARKS and kind != child.type:
            opening = child.type.endswith("_open")
            (marks.add if opening else marks.discard)(MARKS[kind])
            if kind == "link":
                if opening:
                    links.append(str(child.attrGet("href") or ""))
                elif links:
                    links.pop()
            continue
        href = links[-1] if links else ""
        if child.type == "text":
            piece, current = unescape(child.content), frozenset(marks)
        elif child.type == "code_inline":
            piece, current = unescape(child.content), frozenset(marks | {"code"})
        elif child.type in ("softbreak", "hardbreak"):
            piece, current = " ", frozenset(marks)
        else:
            continue
        chars.extend((char, current, href) for char in piece)
    collapsed: list[tuple[str, frozenset[str], str]] = []
    for char, current, href in chars:
        if char.isspace():
            if collapsed and collapsed[-1][0] == " ":
                continue
            collapsed.append((" ", current, href))
        else:
            collapsed.append((char, current, href))
    while collapsed and collapsed[0][0] == " ":
        collapsed.pop(0)
    while collapsed and collapsed[-1][0] == " ":
        collapsed.pop()
    # Terminal escape/control characters are never useful in article prose.
    runs: list[Run] = []
    for char, current, href in collapsed:
        if CONTROL.match(char):
            continue
        if runs and runs[-1][1:] == (current, href):
            runs[-1] = Run(runs[-1].text + char, current, href)
        else:
            runs.append(Run(char, current, href))
    return runs


def speech_text(markdown: str) -> str:
    """Keep prose and link labels; skip code, images, and formatting syntax."""
    blocks = []
    for token in markdown_parser().parse(markdown):
        if token.type == "inline":
            line = "".join(run.text for run in inline_runs(token.children or []))
            if line:
                blocks.append(line)
    return "\n\n".join(blocks)


def parse_article(url: str, body: str) -> Article:
    metadata = {}
    match = re.match(r"\A\ufeff?---\s*\n(.*?)\n---\s*(?:\n|$)", body, re.DOTALL)
    if match:
        try:
            loaded = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise ValueError("Defuddle returned invalid article metadata.") from exc
        if isinstance(loaded, dict):
            metadata = loaded
        body = body[match.end() :]
    title = str(metadata.get("title") or urlsplit(url).hostname or "Article")
    author = str(metadata.get("author") or "")
    text = speech_text(body)
    if not text:
        raise ValueError("No readable article text found. Try another public article URL.")
    # Defuddle generally removes the title from its body; speak it exactly once.
    if text.split("\n", 1)[0].casefold() != title.casefold():
        text = title + ".\n\n" + text
    return Article(url, title, author, body.strip(), text)


async def fetch_article(client: httpx.AsyncClient, url: str, api_key: str = "") -> Article:
    url = normalize_url(url)
    headers = {"Accept": "text/markdown"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Keep the target query string: Defuddle forwards it to the article URL.
    response = await client.get("https://defuddle.md/" + url, headers=headers, timeout=90)
    if response.status_code == 429:
        raise ValueError("Defuddle request limit reached. Try later or set DEFUDDLE_API_KEY.")
    response.raise_for_status()
    if "text/html" in response.headers.get("content-type", ""):
        raise ValueError("Defuddle returned an HTML page instead of an article. Try another URL.")
    return parse_article(url, response.text)


def split_text(text: str, limit: int = 1800) -> list[str]:
    """Prefer sentence/paragraph boundaries, then words, then hard splits for CJK."""
    if limit < 1:
        raise ValueError("Chunk size must be positive.")
    remaining = text.strip()
    chunks = []
    while len(remaining) > limit:
        window = remaining[:limit]
        boundaries = [m.end() for m in re.finditer(r"\n\n|[。！？]|[.!?](?:\s|$)", window)]
        cut = boundaries[-1] if boundaries and boundaries[-1] >= limit // 3 else 0
        if not cut:
            cut = window.rfind(" ")
        if cut < limit // 3:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks
