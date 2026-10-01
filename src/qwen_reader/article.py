from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import urldefrag, urlsplit

import httpx
import yaml
from markdown_it import MarkdownIt


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


def speech_text(markdown: str) -> str:
    """Keep prose and link labels; skip code, images, and formatting syntax."""
    blocks = []
    parser = MarkdownIt("commonmark", {"html": True})
    for token in parser.parse(markdown):
        if token.type != "inline":
            continue
        parts = []
        for child in token.children or []:
            if child.type in ("text", "code_inline"):
                parts.append(child.content)
            elif child.type in ("softbreak", "hardbreak"):
                parts.append(" ")
        line = re.sub(r"\s+", " ", unescape("".join(parts))).strip()
        # Terminal escape/control characters are never useful in article prose.
        line = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", line)
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
