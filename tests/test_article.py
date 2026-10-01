import re

import httpx
import pytest

from qwen_reader.article import fetch_article, normalize_url, parse_article, speech_text, split_text


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
