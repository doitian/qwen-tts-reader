import re

import httpx
import pytest

from qwen_reader.article import (
    STDIN,
    fetch_article,
    load_article,
    normalize_source,
    normalize_url,
    parse_article,
    speech_text,
    split_text,
)


@pytest.mark.parametrize(
    "url", ["", "file:///etc/passwd", "https://", "https://user:pass@host/a", "a b"]
)
def test_rejects_invalid_urls(url):
    with pytest.raises(ValueError):
        normalize_url(url)


def test_url_preserves_article_query():
    assert (
        normalize_url(" example.com/story?a=1&b=2#section ") == "https://example.com/story?a=1&b=2"
    )


def test_markdown_extracts_prose_without_frontmatter_links_or_code_blocks():
    article = parse_article(
        "https://example.com",
        """---
title: "A story"
author: Jane
published: 2026-10-01
---

Hello **world**. Read [this](https://example.com), use `tools`.

![photo](https://example.com/image.png)

```python
print("not narration")
```

## Conclusion

你好，世界。
""",
    )
    assert article.title == "A story"
    assert article.author == "Jane"
    assert (
        article.text
        == "A story.\n\nHello world. Read this, use tools.\n\nConclusion\n\n你好，世界。"
    )


def test_title_is_not_duplicated():
    article = parse_article("https://example.com", "---\ntitle: Test\n---\n# Test\n\nBody.")
    assert article.text == "Test\n\nBody."


def test_empty_article_is_reported():
    with pytest.raises(ValueError, match="No readable"):
        parse_article("https://example.com", "---\ntitle: Empty\n---\n![image](image.png)")


@pytest.mark.parametrize(
    "text",
    [
        "Hello world. " * 500,
        "你好世界。这是一个很长的故事。" * 400,
        "x" * 10000,
        "A short article.",
        "第一段\n\nSecond paragraph, with words!\n\n" * 300,
    ],
    # Windows caps environment variables, including PYTEST_CURRENT_TEST, at 32767 characters.
    ids=["english", "chinese", "unbroken", "short", "paragraphs"],
)
def test_chunks_preserve_all_nonwhitespace_and_respect_limit(text):
    chunks = split_text(text, 100)
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    assert re.sub(r"\s", "", "".join(chunks)) == re.sub(r"\s", "", text)


def test_sentence_boundary_preferred():
    assert split_text("First sentence. Second sentence. Third sentence.", 35) == [
        "First sentence. Second sentence.",
        "Third sentence.",
    ]


def test_html_tags_are_not_spoken():
    assert speech_text("Hi <strong>there</strong> &amp; goodbye.") == "Hi there & goodbye."


async def test_defuddle_receives_target_query_and_its_own_key():
    def handler(request):
        assert str(request.url) == "https://defuddle.md/https://example.com/post?id=2&lang=en"
        assert request.headers["Authorization"] == "Bearer defuddle-key"
        return httpx.Response(200, text="---\ntitle: Hello\n---\nAn article.")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        article = await fetch_article(
            client, "https://example.com/post?id=2&lang=en", "defuddle-key"
        )
    assert article.title == "Hello"


@pytest.mark.parametrize(
    "status,headers,body,expected",
    [
        (429, {}, "", "limit reached"),
        (200, {"content-type": "text/html"}, "<html>Error</html>", "HTML page"),
    ],
)
async def test_defuddle_failures_are_actionable(status, headers, body, expected):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, headers=headers, text=body))
    ) as client:
        with pytest.raises(ValueError, match=expected):
            await fetch_article(client, "https://example.com")


def test_sources_are_urls_local_files_or_stdin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("Notes.")
    (tmp_path / "example.com").write_text("A file that looks like a host.")
    assert normalize_source(" - ") == STDIN
    assert normalize_source("notes.md") == str(tmp_path.resolve() / "notes.md")
    assert normalize_source("example.com") == str(tmp_path.resolve() / "example.com")
    assert normalize_source("example.org/story") == "https://example.org/story"
    # Paths are recognized before the file exists, so a typo isn't sent to Defuddle.
    assert normalize_source("./missing.md") == str(tmp_path.resolve() / "missing.md")
    assert normalize_source(str(tmp_path / "missing.md")) == str(tmp_path.resolve() / "missing.md")
    for source in ["-", "notes.md", "./missing.md", "example.org/story"]:
        assert normalize_source(normalize_source(source)) == normalize_source(source)
    with pytest.raises(ValueError):
        normalize_source("file:///etc/passwd")


async def test_local_markdown_and_stdin_skip_defuddle(tmp_path):
    def handler(request):
        pytest.fail("Local Markdown must not be sent anywhere")

    heading = tmp_path / "heading.md"
    heading.write_bytes("﻿# My **notes**\r\n\r\nFirst point.\r\n".encode())
    plain = tmp_path / "plain.md"
    plain.write_text("Just text.", encoding="utf-8")
    titled = tmp_path / "titled.md"
    titled.write_text("---\ntitle: Front matter\nauthor: Me\n---\nBody.", encoding="utf-8")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        article = await load_article(client, str(heading))
        assert (article.title, article.text) == ("My notes", "My notes\n\nFirst point.")
        # A file name or "Standard input" is shown, never spoken.
        article = await load_article(client, str(plain))
        assert (article.title, article.text) == ("plain", "Just text.")
        article = await load_article(client, str(titled))
        assert (article.title, article.author) == ("Front matter", "Me")
        assert article.text == "Front matter.\n\nBody."
        article = await load_article(client, STDIN, stdin="Piped *text*.")
        assert (article.url, article.title, article.text) == (
            STDIN,
            "Standard input",
            "Piped text.",
        )


async def test_unreadable_local_sources_are_reported(tmp_path):
    binary = tmp_path / "binary.md"
    binary.write_bytes(b"\xff\xfe\x00bad")
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="Can't read .*missing.md"):
            await load_article(client, str(tmp_path / "missing.md"))
        with pytest.raises(ValueError, match="isn't UTF-8"):
            await load_article(client, str(binary))
        with pytest.raises(ValueError, match="Nothing was piped in"):
            await load_article(client, STDIN)
        with pytest.raises(ValueError, match="No readable article text found.$"):
            await load_article(client, STDIN, stdin="![image](image.png)")
