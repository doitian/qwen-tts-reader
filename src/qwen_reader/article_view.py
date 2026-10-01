from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual import events
from textual.containers import VerticalScroll
from textual.geometry import Region
from textual.message import Message
from textual.scrollbar import ScrollDown, ScrollTo, ScrollUp
from textual.widgets import Static

from .markdown_view import Document, Spoken
from .narration import TextSpan, narration_spans


class ArticleText(Static):
    """Rendered article with clickable reading units, whose metadata survives line wrapping."""

    ALLOW_SELECT = False

    class Selected(Message):
        def __init__(self, index: int):
            super().__init__()
            self.index = index

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = ""
        self.spans: list[TextSpan] = []
        self.current: int | None = None
        self.pending: int | None = None
        self.document: Document | None = None
        self._line_ranges: dict[int, tuple[int, int]] = {}
        self._line_size = (0, 0)

    def set_article(self, text: str, limit: int, markdown: str | None = None) -> None:
        self.text = text
        self.spans = narration_spans(text, limit)
        self.document = Document.from_markdown(markdown, text) if markdown else Document.plain(text)
        for spoken in self.document.spoken:
            end = spoken.start + spoken.length
            spoken.units = [
                index
                for index, span in enumerate(self.spans)
                if span.start < end and span.end > spoken.start
            ]
        self.current = self.pending = None
        self._line_ranges.clear()
        self._line_size = (0, 0)
        self.redraw()

    def redraw(self) -> None:
        if self.document:
            self.update(self.document.render(self.styled))

    def styled(self, spoken: Spoken) -> Text:
        text = spoken.text.copy()
        for index in spoken.units:
            span = self.spans[index]
            style = Style(meta={"reading_unit": index})
            if index == self.current:
                style += Style(color="#111821", bgcolor="#83dfcd", bold=True)
            elif index == self.pending:
                style += Style(color="#f1ce83", underline=True)
            text.stylize(style, max(0, span.start - spoken.start), span.end - spoken.start)
        return text

    def highlight(self, index: int | None, pending: int | None = None) -> None:
        if (index, pending) != (self.current, self.pending):
            self.current, self.pending = index, pending
            self.redraw()

    def on_click(self, event: events.Click) -> None:
        href = event.style.meta.get("href")
        if event.ctrl and isinstance(href, str):
            event.stop()
            self.app.open_url(href)
            self.notify(f"Opening {href}")
            return
        index = event.style.meta.get("reading_unit")
        if isinstance(index, int) and 0 <= index < len(self.spans):
            event.stop()
            self.post_message(self.Selected(index))

    def reading_region(self, index: int) -> Region | None:
        # Use the actual rendered lines: CJK, emoji, wrapping, and terminal resize
        # all affect screen coordinates differently from string lengths.
        width = self.content_size.width
        size = (width, self.content_size.height)
        if self._line_size != size:
            self._line_ranges.clear()
            self._line_size = size
            for y in range(self.content_size.height):
                for segment in self.render_line(y):
                    unit = segment.style.meta.get("reading_unit") if segment.style else None
                    if isinstance(unit, int):
                        first = self._line_ranges.get(unit, (y, y))[0]
                        self._line_ranges[unit] = (first, y)
        lines = self._line_ranges.get(index)
        return Region(0, lines[0], width, lines[1] - lines[0] + 1) if lines else None


class ArticleView(VerticalScroll):
    class FollowChanged(Message):
        def __init__(self, following: bool):
            super().__init__()
            self.following = following

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.following = True

    def set_following(self, following: bool) -> None:
        if following != self.following:
            self.following = following
            self.post_message(self.FollowChanged(following))

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.set_following(False)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.set_following(False)

    def on_key(self, event: events.Key) -> None:
        if event.key in {
            "up",
            "down",
            "pageup",
            "pagedown",
            "home",
            "end",
            "ctrl+pageup",
            "ctrl+pagedown",
        }:
            self.set_following(False)

    def on_scroll_to(self, event: ScrollTo) -> None:
        self.set_following(False)

    def on_scroll_up(self, event: ScrollUp) -> None:
        self.set_following(False)

    def on_scroll_down(self, event: ScrollDown) -> None:
        self.set_following(False)

    def follow(self, index: int | None) -> None:
        if not self.following or index is None:
            return
        region = self.query_one(ArticleText).reading_region(index)
        if region is not None:
            # A long paragraph starts at the top; shorter ones remain fully visible.
            self.scroll_to_region(
                region,
                animate=False,
                immediate=True,
                x_axis=False,
                top=region.height > self.scrollable_content_region.height,
            )

    def on_resize(self) -> None:
        self.call_after_refresh(self.follow, self.query_one(ArticleText).current)
