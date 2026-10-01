"""Glow-style terminal rendering of article Markdown that keeps speech text offsets.

Every spoken block is drawn with exactly the characters it has in the speech
text, so reading units can be highlighted and clicked by offset, including
units that split one long paragraph.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from markdown_it.tree import SyntaxTreeNode
from rich import box
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.measure import Measurement
from rich.rule import Rule
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .article import inline_runs, markdown_parser

MARK_STYLES = {
    "strong": Style(bold=True),
    "em": Style(italic=True),
    "strike": Style(strike=True),
    "code": Style(color="#ff5f5f", bgcolor="#303030"),
    "link": Style(color="#00af87", underline=True),
}
TITLE = Style(color="#ffff87", bgcolor="#5f5fff", bold=True)
HEADING = Style(color="#00afff", bold=True)
MUTED = Style(color="#6c7f93")
CODE_BACKGROUND = "#182330"


@dataclass
class Spoken:
    """One speech-text block on screen; `length` counts its characters in the speech text."""

    start: int
    length: int
    text: Text
    units: list[int] = field(default_factory=list)


Styler = Callable[[Spoken], Text]


class Prefixed:
    """Render a block narrower, with a marker on its first line and a gutter on the rest."""

    def __init__(self, renderable: RenderableType, first: str, rest: str, style: Style):
        self.renderable = renderable
        self.first = first
        self.rest = rest
        self.style = style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = max(1, options.max_width - cell_len(self.first))
        lines = console.render_lines(self.renderable, options.update_width(width), pad=False)
        for index, line in enumerate(lines):
            yield Segment(self.first if index == 0 else self.rest, self.style)
            yield from line
            yield Segment.line()

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        indent = cell_len(self.first)
        inner = Measurement.get(
            console, options.update_width(max(1, options.max_width - indent)), self.renderable
        )
        return Measurement(inner.minimum + indent, inner.maximum + indent)


class Block:
    def render(self, styled: Styler) -> RenderableType:
        raise NotImplementedError


@dataclass
class Paragraph(Block):
    spoken: Spoken

    def render(self, styled: Styler) -> RenderableType:
        return styled(self.spoken)


@dataclass
class Heading(Block):
    level: int
    spoken: Spoken

    def render(self, styled: Styler) -> RenderableType:
        if self.level == 1:
            return Text.assemble(
                Text(" ", TITLE), styled(self.spoken), Text(" ", TITLE), overflow="fold"
            )
        return Text.assemble(
            Text("#" * self.level + " ", HEADING), styled(self.spoken), overflow="fold"
        )


@dataclass
class Container(Block):
    children: list[Block]
    tight: bool = False

    def render(self, styled: Styler) -> RenderableType:
        parts: list[RenderableType] = []
        for index, child in enumerate(self.children):
            if index and not self.tight:
                parts.append(Text(""))
            parts.append(child.render(styled))
        return Group(*parts)


@dataclass
class Quote(Block):
    body: Container

    def render(self, styled: Styler) -> RenderableType:
        return Prefixed(self.body.render(styled), "│ ", "│ ", MUTED)


@dataclass
class ListBlock(Block):
    items: list[tuple[str, Container]]

    def render(self, styled: Styler) -> RenderableType:
        width = max(cell_len(marker) for marker, _ in self.items) + 1
        return Group(
            *(
                Prefixed(body.render(styled), marker.ljust(width), " " * width, HEADING)
                for marker, body in self.items
            )
        )


@dataclass
class Code(Block):
    code: str
    language: str

    def render(self, styled: Styler) -> RenderableType:
        syntax = Syntax(
            self.code,
            self.language or "text",
            theme="monokai",
            background_color=CODE_BACKGROUND,
            word_wrap=True,
            padding=(0, 1),
        )
        return Prefixed(syntax, "  ", "  ", MUTED)


@dataclass
class TableBlock(Block):
    header: list[Spoken | None]
    rows: list[list[Spoken | None]]

    def render(self, styled: Styler) -> RenderableType:
        table = Table(box=box.MINIMAL, show_edge=False, border_style=MUTED, header_style=HEADING)
        for cell in self.header:
            table.add_column(styled(cell) if cell else "")
        for row in self.rows:
            table.add_row(*(styled(cell) if cell else "" for cell in row))
        return table


class RuleBlock(Block):
    def render(self, styled: Styler) -> RenderableType:
        return Rule(style=MUTED)


@dataclass
class Document:
    root: Container
    spoken: list[Spoken]

    def render(self, styled: Styler) -> RenderableType:
        return self.root.render(styled)

    @classmethod
    def plain(cls, text: str) -> Document:
        spoken = []
        start = 0
        for paragraph in text.split("\n\n"):
            spoken.append(Spoken(start, len(paragraph), Text(paragraph, overflow="fold")))
            start += len(paragraph) + 2
        return cls(Container([Paragraph(value) for value in spoken]), spoken)

    @classmethod
    def from_markdown(cls, markdown: str, text: str) -> Document:
        """Render Markdown, or fall back to plain paragraphs if it doesn't match the text."""
        tokens = markdown_parser().parse(markdown)
        inline = [
            (token, runs)
            for token in tokens
            if token.type == "inline" and (runs := inline_runs(token.children or []))
        ]
        paragraphs = text.split("\n\n")
        # parse_article may prepend the title, which isn't part of the Markdown body.
        extra = len(paragraphs) - len(inline)
        lines = ["".join(part for part, _ in runs) for _, runs in inline]
        if extra not in (0, 1) or paragraphs[extra:] != lines:
            return cls.plain(text)
        starts = []
        offset = 0
        for paragraph in paragraphs:
            starts.append(offset)
            offset += len(paragraph) + 2
        by_token = {
            id(token): (starts[extra + index], runs) for index, (token, runs) in enumerate(inline)
        }
        spoken: list[Spoken] = []

        def spoken_for(node: SyntaxTreeNode | None, style: Style | None = None) -> Spoken | None:
            entry = by_token.get(id(node.token)) if node is not None and node.token else None
            if entry is None:
                return None
            start, runs = entry
            content = Text(overflow="fold", style=style or "")
            for part, marks in runs:
                styles = [MARK_STYLES[mark] for mark in sorted(marks)]
                content.append(part, Style.combine(styles) if styles else None)
            value = Spoken(start, len(content), content)
            spoken.append(value)
            return value

        def inline_child(node: SyntaxTreeNode) -> SyntaxTreeNode | None:
            return next((child for child in node.children if child.type == "inline"), None)

        def blocks(node: SyntaxTreeNode) -> list[Block]:
            return [block for child in node.children for block in walk(child)]

        def walk(node: SyntaxTreeNode) -> list[Block]:
            kind = node.type
            if kind == "paragraph":
                value = spoken_for(inline_child(node))
                return [Paragraph(value)] if value else []
            if kind == "heading":
                level = int(node.tag[1:])
                value = spoken_for(inline_child(node), TITLE if level == 1 else HEADING)
                return [Heading(level, value)] if value else []
            if kind == "blockquote":
                return [Quote(Container(blocks(node)))]
            if kind in ("bullet_list", "ordered_list"):
                first = int(node.attrs.get("start", 1)) if kind == "ordered_list" else 0
                items = [
                    (
                        f"{first + index}." if kind == "ordered_list" else "•",
                        Container(blocks(item), tight=True),
                    )
                    for index, item in enumerate(node.children)
                ]
                return [ListBlock(items)] if items else []
            if kind in ("fence", "code_block"):
                language = node.info.split(maxsplit=1)[0] if node.info.strip() else ""
                return [Code(node.content.rstrip("\n"), language)]
            if kind == "table":
                sections = {child.type: child for child in node.children}
                rows = [
                    [
                        spoken_for(inline_child(cell), HEADING if cell.type == "th" else None)
                        for cell in row.children
                    ]
                    for section in ("thead", "tbody")
                    if section in sections
                    for row in sections[section].children
                ]
                header = rows[0] if "thead" in sections and rows else []
                return [TableBlock(header, rows[1:] if header else rows)]
            if kind == "hr":
                return [RuleBlock()]
            if kind == "inline":
                value = spoken_for(node)
                return [Paragraph(value)] if value else []
            return blocks(node)

        children = blocks(SyntaxTreeNode(tokens))
        if extra:
            title = paragraphs[0]
            value = Spoken(
                0, len(title), Text(title.removesuffix("."), style=TITLE, overflow="fold")
            )
            spoken.insert(0, value)
            children.insert(0, Heading(1, value))
        if len(spoken) != len(paragraphs):
            return cls.plain(text)
        spoken.sort(key=lambda value: value.start)
        return cls(Container(children), spoken)
