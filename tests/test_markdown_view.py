from textual.app import App

from qwen_reader.article import inline_runs, markdown_parser, parse_article, speech_text
from qwen_reader.article_view import ArticleText
from qwen_reader.markdown_view import Document

LONG = " ".join(f"Sentence {i} of the long paragraph." for i in range(30))
BODY = f"""---
title: Rendering test
---
## A section heading

Some **bold**, *italic*, ~~struck~~, `inline code` and a [link](https://example.com).

- First bullet with enough words to wrap onto a second line in a narrow window.
- Second bullet
  1. Nested ordered
  2. Another one

> A quoted passage that is long enough to wrap and show the bar on every line.

```python
def hello():
    return "not spoken"
```

| Name | Value |
|------|-------|
| One  | 1     |

---

{LONG}
"""


class Reader(App):
    CSS = "ArticleText { height: auto; }"

    def __init__(self):
        super().__init__()
        self.selected: list[int] = []

    def compose(self):
        yield ArticleText(id="article-text")

    def on_article_text_selected(self, message: ArticleText.Selected) -> None:
        self.selected.append(message.index)


def screen_lines(view: ArticleText) -> list[str]:
    return [
        "".join(segment.text for segment in view.render_line(y)).rstrip()
        for y in range(view.content_size.height)
    ]


def highlighted(view: ArticleText) -> str:
    return "".join(
        segment.text
        for y in range(view.content_size.height)
        for segment in view.render_line(y)
        if segment.style
        and segment.style.bgcolor
        and segment.style.bgcolor.triplet == (131, 223, 205)
    )


def test_inline_runs_carry_marks_and_join_to_the_spoken_line():
    token = next(
        t for t in markdown_parser().parse("A **b *c*** `d` [e](x) ~~f~~") if t.type == "inline"
    )
    runs = inline_runs(token.children)
    assert "".join(text for text, _ in runs) == "A b c d e f"
    assert ("c", frozenset({"strong", "em"})) in runs
    assert ("d", frozenset({"code"})) in runs
    assert ("e", frozenset({"link"})) in runs
    assert ("f", frozenset({"strike"})) in runs


def test_tables_and_strikethrough_are_spoken_without_markup():
    assert speech_text("| A | B |\n|---|---|\n| 1 | 2 |\n\n~~gone~~ text") == (
        "A\n\nB\n\n1\n\n2\n\ngone text"
    )


def test_every_rendered_block_shows_exactly_its_speech_text():
    article = parse_article("https://example.test", BODY)
    document = Document.from_markdown(article.markdown, article.text)
    assert [type(block).__name__ for block in document.root.children][:3] == [
        "Heading",
        "Heading",
        "Paragraph",
    ]
    assert len(document.spoken) == len(article.text.split("\n\n"))
    for spoken in document.spoken[1:]:
        assert spoken.text.plain == article.text[spoken.start : spoken.start + spoken.length]
    # The title block shows the title without the period added for speech.
    assert document.spoken[0].text.plain == "Rendering test"
    assert "not spoken" not in article.text


def test_mismatched_markdown_falls_back_to_plain_paragraphs():
    document = Document.from_markdown("Different text.", "First.\n\nSecond.")
    assert [spoken.text.plain for spoken in document.spoken] == ["First.", "Second."]


async def test_glow_style_layout_and_split_paragraph_chunks_highlight_separately():
    article = parse_article("https://example.test", BODY)
    app = Reader()
    async with app.run_test(size=(60, 80)) as pilot:
        view = app.query_one(ArticleText)
        view.set_article(article.text, 400, article.markdown)
        await pilot.pause()
        lines = screen_lines(view)
        assert "## A section heading" in lines
        assert any(line.startswith("• First bullet") for line in lines)
        assert any(line.startswith("  1. Nested ordered") for line in lines)
        quote = [line for line in lines if line.startswith("│ ")]
        assert len(quote) >= 2  # The bar continues on wrapped lines.
        assert any("def hello():" in line for line in lines)
        assert any("Name" in line and "│" in line for line in lines)

        long_units = [
            index
            for index, span in enumerate(view.spans)
            if article.text[span.start : span.end].startswith("Sentence")
        ]
        assert len(long_units) == 3
        paragraph_lines = [line for line in lines if "Sentence" in line]
        assert all(line for line in paragraph_lines)  # One paragraph, no blank lines inside.
        middle = long_units[1]
        view.highlight(middle)
        await pilot.pause()
        span = view.spans[middle]
        assert highlighted(view).replace(" ", "") == article.text[span.start : span.end].replace(
            " ", ""
        )

        # Clicking a chunk selects that chunk, not the whole paragraph.
        region = view.reading_region(long_units[2])
        await pilot.click("#article-text", offset=(1, region.bottom - 1))
        await pilot.pause()
        assert app.selected == [long_units[2]]
