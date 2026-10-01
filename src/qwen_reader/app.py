from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import httpx
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Footer, Header, Input, ProgressBar, Static
from textual.worker import Worker, WorkerCancelled, WorkerFailed

from .article import fetch_article, normalize_url
from .article_view import ArticleText, ArticleView
from .config import MODEL_VOICES, Settings
from .model_screen import ModelScreen
from .narration import Cue, TextSpan
from .playback_state import Bookmark, ModelChoice, PlaybackState
from .player import MpvPlayer, PlayerError
from .screens import ReaderCommandPalette
from .synthesis import AudioPart, Synthesizer
from .timeline import Timeline

log = logging.getLogger(__name__)

SEEK_DEBOUNCE = 0.25
# Generate at most this much listening time ahead of playback, in seconds.
LOOKAHEAD = 180
# Speech has caught up with a seek once mpv reports a position this close to it.
LANDED = 0.35


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


class Stage(Enum):
    QUEUED = "queued"  # Repeated presses may still replace it; speech carries on.
    BUFFERING = "buffering"  # Silent until the target's audio is in the playlist.
    LANDING = "landing"  # mpv was told to seek; waiting for it to report the target.


@dataclass
class Seek:
    """A position the UI asked for. The UI shows it until speech has caught up."""

    unit: int
    offset: float
    time: float  # The whole-article time the UI shows for it.
    due: float
    reason: str = "seek"  # "start" and "restore" have their own status messages.
    restore: Bookmark | None = None
    stage: Stage = Stage.QUEUED
    target: float = 0.0
    sent: float = 0.0


@dataclass
class Window:
    """The mpv playlist: audio from `anchor` onward, fed by one streaming worker."""

    anchor: int
    duration: float
    cues: dict[int, Cue] = field(default_factory=dict)
    parts: list[AudioPart] = field(default_factory=list)
    loaded: bool = False
    complete: bool = False
    failed: str = ""
    cued: bool = False
    streaming_unit: int = -1
    worker: Worker | None = None


class ReaderApp(App):
    """Speech follows the UI: handlers only record intent, and one controller drives mpv.

    While a seek is pending the UI shows its target. Speech is brought there, and the
    UI follows speech again only once mpv reports that position.
    """

    TITLE = "Qwen · Article Reader"
    CSS = """
    Screen { background: #111821; }
    Header { background: #1c2b3c; }
    #main { padding: 0 2; }
    #url-row { height: 3; margin-bottom: 1; }
    #url { width: 1fr; }
    #load { margin-left: 1; min-width: 10; }
    #choose-model { margin-left: 1; min-width: 9; }
    #article-title { color: #83dfcd; text-style: bold; height: 1; }
    #article-meta { color: #91a4b7; margin-bottom: 1; max-height: 2; }
    #article-view { height: 1fr; border: round #344b60; padding: 0 1; }
    #article-text { height: auto; }
    #status { height: auto; min-height: 1; max-height: 4; margin-top: 1; }
    #status.error { color: #ff9b9b; }
    #progress { width: 1fr; height: 1; }
    #progress Bar { width: 1fr; }
    #timeline { height: 1; color: #91a4b7; }
    #controls { height: 3; align: center middle; }
    #controls Button { min-width: 8; margin: 0 1 0 0; }
    #controls #follow { min-width: 13; margin-right: 0; }
    #speed { width: 8; content-align: center middle; height: 3; }
    #hint { color: #91a4b7; text-align: center; height: 1; }
    Footer { background: #1c2b3c; }
    """
    BINDINGS = [
        Binding("space", "toggle_pause", "Play/pause", priority=True),
        Binding("left", "seek(-10)", "−10s", priority=True),
        Binding("right", "seek(10)", "+10s", priority=True),
        Binding("minus", "speed(-0.1)", "Slower", priority=True),
        Binding("plus,equals", "speed(0.1)", "Faster", priority=True),
        # Letter keys are not priority bindings, so the URL input still receives them.
        Binding("h", "seek(-10)", "−10s", show=False),
        Binding("l", "seek(10)", "+10s", show=False),
        Binding("j", "paragraph(1)", "Next paragraph", show=False),
        Binding("k", "paragraph(-1)", "Previous paragraph", show=False),
        Binding("f", "follow", "Follow", show=False),
        Binding("r", "read_visible", "Read from here", show=False),
        Binding("ctrl+f", "scroll_page(1)", "Page down", show=False),
        Binding("ctrl+b", "scroll_page(-1)", "Page up", show=False),
        Binding("ctrl+l", "focus_url", "URL", priority=True),
        Binding("f2", "choose_model", "Model", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", priority=True, show=False),
    ]

    def __init__(
        self,
        settings: Settings,
        url: str = "",
        *,
        player: MpvPlayer | None = None,
        client: httpx.AsyncClient | None = None,
        restore_choices: bool = True,
    ):
        super().__init__()
        self.playback_state = PlaybackState(settings.cache_dir / "playback.json")
        self.startup_configuration = hashlib.sha256(
            json.dumps([settings.endpoint, settings.model, settings.voice]).encode()
        ).hexdigest()
        last = self.playback_state.get(
            self.playback_state.last_url, self.playback_state.last_narration
        )
        choice = self.playback_state.choice
        if (
            restore_choices
            and choice
            and choice.configuration == self.startup_configuration
            and choice.model in MODEL_VOICES
        ):
            settings = replace(settings, model=choice.model, voice=choice.voice)
        elif (
            not url
            and restore_choices
            and last
            and last.configuration == self.startup_configuration
            and last.model in MODEL_VOICES
        ):
            settings = replace(settings, model=last.model, voice=last.voice)
        self.settings = settings
        self.initial_url = url or self.playback_state.last_url
        self.player = player or MpvPlayer()
        self.client = client or httpx.AsyncClient(follow_redirects=True, timeout=60)
        self.synthesizer = Synthesizer(settings, self.client)
        self.prepare_worker: Worker | None = None
        self.controller: Worker | None = None
        self.wakeup = asyncio.Event()
        self.mpv_lock = asyncio.Lock()
        self.closing = False
        # The article.
        self.loading = False
        self.article_loaded = False
        self.halted = False
        self.current_url = ""
        self.narration_id = ""
        self.article_text = ""
        self.spans: list[TextSpan] = []
        self.unit_of: dict[int, int] = {}
        self.timeline = Timeline([])
        # What the UI asks for.
        self.seek: Seek | None = None
        self.want_paused = False
        self.speed = 1.0
        self.restored_finished = False
        # What speech is doing.
        self.window: Window | None = None
        self.spool: tempfile.TemporaryDirectory | None = None
        self.heard = 0.0
        self.heard_ended = False
        self.applied_paused: bool | None = None
        self.applied_speed: float | None = None
        self.generation_error = ""
        self.synthesis_message = ""
        self.last_saved_at = 0.0
        self.save_warning_shown = False
        self.buffering_since: float | None = None
        self.playing_sample: tuple[float, float] | None = None

    # --- Derived state ---------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """Whether mpv holds audio of the current playlist."""
        return bool(self.window and self.window.loaded)

    @property
    def preparing(self) -> bool:
        window = self.window
        return self.loading or bool(window and not window.complete and not window.failed)

    @property
    def generation_complete(self) -> bool:
        return bool(self.window and self.window.complete)

    @property
    def duration(self) -> float:
        """Whole-article time up to which audio is in mpv."""
        return self.window.duration if self.window else 0.0

    @property
    def cues(self) -> dict[int, Cue]:
        return self.window.cues if self.window else {}

    @property
    def ended(self) -> bool:
        return self.restored_finished or (
            self.seek is None and self.heard_ended and self.generation_complete
        )

    @property
    def paused(self) -> bool:
        return self.want_paused

    @property
    def position(self) -> float:
        """Whole-article position the UI shows: a pending seek's target, else speech."""
        if self.seek is not None:
            return self.seek_time(self.seek)
        if self.ended:
            return self.duration
        return self.heard

    def seek_time(self, seek: Seek) -> float:
        return seek.target if seek.stage is Stage.LANDING else seek.time

    def unit_time(self, unit: int) -> float:
        cue = self.cues.get(self.spans[unit].start)
        return cue.time if cue else self.timeline.start(unit)

    def position_unit(self, position: float) -> tuple[int | None, float]:
        """The unit playing at a whole-article `position`, and the offset into it."""
        if not self.spans:
            return None, 0.0
        if self.timeline.anchor_time <= position < max(self.duration, self.timeline.anchor_time):
            current = None
            for index, span in enumerate(self.spans[self.timeline.anchor :], self.timeline.anchor):
                cue = self.cues.get(span.start)
                if cue is None or cue.time > position + 0.001:
                    break
                current = (index, position - cue.time)
            if current:
                return current
        return self.timeline.locate(position)

    def ui_unit(self) -> int | None:
        if self.seek is not None:
            return self.seek.unit
        if not self.ready:
            return None
        return self.position_unit(self.duration - 0.001 if self.ended else self.heard)[0]

    def can_seek(self) -> bool:
        return self.article_loaded and not self.halted and bool(self.spans)

    # --- Layout ----------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main"):
            with Horizontal(id="url-row"):
                yield Input(self.initial_url, placeholder="Paste an article URL…", id="url")
                yield Button("Read", variant="primary", id="load")
                yield Button("Model", id="choose-model")
            yield Static("An article, at your pace.", id="article-title", markup=False)
            yield Static(
                f"{self.settings.model} · {self.settings.voice_id}", id="article-meta", markup=False
            )
            with ArticleView(id="article-view"):
                yield ArticleText(
                    "Paste a public article URL and press Enter.\n\n"
                    "The article will appear here while its narration is prepared.\n"
                    "Playback starts as audio arrives; the rest streams in the background.",
                    id="article-text",
                    markup=False,
                )
            yield Static("Ready for an article.", id="status", markup=False)
            yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="progress")
            yield Static("00:00 / 00:00", id="timeline", markup=False)
            with Horizontal(id="controls"):
                yield Button("−10s", id="rewind", disabled=True)
                yield Button("Play", id="play", variant="success", disabled=True)
                yield Button("+10s", id="forward", disabled=True)
                yield Button("−", id="slower")
                yield Static("1.0×", id="speed")
                yield Button("+", id="faster")
                yield Button("Following", id="follow", variant="primary", disabled=True)
            yield Static("Click text to seek · Scroll to browse · Resume sync to follow", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self.controller = self.run_worker(self.control(), group="control", exit_on_error=True)
        self.query_one("#url", Input).focus()
        if self.initial_url:
            self.begin_load()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if self.closing:
            return False
        if action == "quit":
            return True
        if isinstance(self.screen, ModelScreen):
            return False
        if action in {
            "toggle_pause",
            "seek",
            "speed",
            "paragraph",
            "follow",
            "scroll_page",
            "read_visible",
        } and isinstance(self.focused, Input):
            return False
        return True

    def set_status(self, message: str, *, error: bool = False) -> None:
        widget = self.query_one("#status", Static)
        widget.update(message)
        widget.set_class(error, "error")

    def wake(self) -> None:
        self.wakeup.set()

    # --- Loading an article ----------------------------------------------------------

    @on(Input.Submitted, "#url")
    @on(Button.Pressed, "#load")
    def begin_load(self) -> None:
        if self.closing:
            return
        try:
            url = normalize_url(self.query_one("#url", Input).value)
        except ValueError as exc:
            self.set_status(str(exc), error=True)
            return
        self.query_one("#url", Input).value = url
        self.query_one("#article-view", VerticalScroll).focus()
        self.prepare_worker = self.prepare(url)

    @work(exclusive=True, group="prepare")
    async def prepare(self, url: str) -> None:
        self.checkpoint(force=True)
        self.loading = True
        self.article_loaded = False
        self.halted = False
        self.seek = None
        self.restored_finished = False
        self.generation_error = ""
        self.synthesis_message = ""
        self.heard, self.heard_ended = 0.0, False
        self.reset_reading()
        self.query_one("#timeline", Static).update("00:00 / 00:00")
        self.query_one("#play", Button).label = "Play"
        self.query_one("#progress", ProgressBar).update(total=None)
        self.set_status("Fetching article with Defuddle… · Esc to cancel")
        try:
            async with self.mpv_lock:
                await self.close_window()
            article = await fetch_article(self.client, url, self.settings.defuddle_key)
            self.query_one("#article-title", Static).update(article.title)
            self.query_one("#article-meta", Static).update(
                " · ".join(
                    filter(
                        None,
                        [
                            article.author,
                            self.settings.model,
                            self.settings.voice_id,
                            f"{len(article.text):,} characters",
                        ],
                    )
                )
            )
            text = self.query_one(ArticleText)
            text.set_article(article.text, self.settings.chunk_chars, article.markdown)
            self.query_one("#article-view", VerticalScroll).scroll_home(
                animate=False, immediate=True
            )
            self.current_url = url
            self.article_text = article.text
            self.spans = text.spans
            self.unit_of = {span.start: index for index, span in enumerate(self.spans)}
            self.timeline = Timeline(
                [span.end - span.start for span in text.spans],
                self.synthesizer.known_durations(article.text),
                article.text,
            )
            self.narration_id = self.synthesizer.cache_path(article.text, "article").stem
            self.settings.validate_tts()
            bookmark = self.playback_state.get(url, self.narration_id)
            if bookmark:
                self.speed = bookmark.speed
                self.query_one("#speed", Static).update(f"{self.speed:.1f}×")
                self.want_paused = bookmark.paused or bookmark.completed
                unit, offset = self.bookmark_target(bookmark)
                shown = self.timeline.start(unit) + offset
                self.seek = Seek(unit, offset, shown, 0.0, "restore", bookmark)
            else:
                self.want_paused = False
                self.seek = Seek(0, 0.0, 0.0, 0.0, "start")
            self.article_loaded = True
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ValueError, OSError, PlayerError, TimeoutError) as exc:
            self.generation_error = self.describe_failure(exc)
            self.set_status(self.generation_error, error=True)
            self.query_one("#progress", ProgressBar).update(total=100, progress=0)
        finally:
            self.loading = False
            self.wake()

    def bookmark_target(self, bookmark: Bookmark) -> tuple[int, float]:
        last = len(self.timeline.lengths) - 1
        if bookmark.completed:
            return last, self.timeline.duration(last)
        if 0 <= bookmark.unit <= last:
            return bookmark.unit, bookmark.unit_offset
        # Older bookmarks store only seconds; their earlier paragraphs are cached.
        return self.timeline.locate(bookmark.position)

    def describe_failure(self, exc: BaseException) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"Request failed (HTTP {exc.response.status_code}). Try again."
        if isinstance(exc, httpx.HTTPError):
            return "Network request failed. Check your connection and try again."
        return str(exc) or "Audio preparation timed out. Try again."

    # --- The controller: the only code that commands mpv -----------------------------

    async def control(self) -> None:
        while not self.closing:
            delay = 0.1
            if self.seek and self.seek.stage is Stage.QUEUED:
                delay = max(0.0, min(delay, self.seek.due - time.monotonic()))
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(delay):
                    await self.wakeup.wait()
            self.wakeup.clear()
            if self.closing:
                return
            try:
                async with self.mpv_lock:
                    await self.step()
            except PlayerError as exc:
                log.warning("player error: %s", exc)
                self.generation_error = str(exc)
                self.seek = None
                async with self.mpv_lock:
                    await self.close_window()
            try:
                self.render_playback()
            except NoMatches:
                return  # The screen is gone; the app is shutting down.

    async def refresh_playback(self) -> None:
        """Run one controller step now instead of on the next tick."""
        async with self.mpv_lock:
            await self.step()
        self.render_playback()

    async def step(self) -> None:
        if not self.article_loaded or self.halted:
            return
        seek, now = self.seek, time.monotonic()
        if seek and seek.stage is Stage.QUEUED and now >= seek.due:
            await self.dispatch(seek)
        window = self.window
        if window is None:
            return
        await self.feed(window)
        if not window.loaded:
            return
        position, _, ended = await self.player.status()
        self.observe(self.timeline.anchor_time + position, ended)
        if seek is self.seek and seek is not None:
            await self.advance(seek, window, now)
        await self.reconcile()
        self.checkpoint()

    async def dispatch(self, seek: Seek) -> None:
        """Decide how speech reaches a seek once its debounce is over."""
        seek.stage = Stage.BUFFERING
        window = self.window
        if window and (self.landable(seek, window) is not None or self.near_live_edge(seek.unit)):
            log.debug("seek to unit %d + %.2fs within the playlist", seek.unit, seek.offset)
            return
        await self.open_window(seek.unit, seek.time - seek.offset)

    async def advance(self, seek: Seek, window: Window, now: float) -> None:
        if seek.stage is Stage.BUFFERING:
            target = self.landable(seek, window)
            if target is not None:
                # A freshly loaded playlist may already be there; at the end, seek to replay.
                if self.heard_ended or abs(self.heard - target) > 0.01:
                    await self.player.seek(max(0.0, target - self.timeline.anchor_time))
                seek.stage, seek.target, seek.sent = Stage.LANDING, target, now
                log.debug("seek sent: %.2fs (unit %d)", target, seek.unit)
            elif window.failed or window.complete:
                log.info("seek to unit %d abandoned: its audio is unavailable", seek.unit)
                self.seek = None
        elif seek.stage is Stage.LANDING:
            # mpv may report the old position for a moment; follow it only once it arrives.
            if abs(self.heard - seek.target) < LANDED or now - seek.sent > 1.5:
                log.debug("seek landed at %.2fs", self.heard)
                self.seek = None
                if seek.restore:
                    self.restored_finished = seek.restore.completed
                self.checkpoint(force=True)

    def landable(self, seek: Seek, window: Window) -> float | None:
        """Where mpv can seek for `seek` now, or None while its audio is still to come."""
        if not window.loaded:
            return None
        cue = window.cues.get(self.spans[seek.unit].start)
        if cue is None or cue.time >= window.duration:
            return None
        target = cue.time + seek.offset
        complete = cue.end_time is not None or window.complete
        if target >= window.duration - 0.01 and not complete:
            return None
        end = cue.end_time if cue.end_time is not None else window.duration
        return max(cue.time, min(target, end - 0.01, window.duration - 0.01))

    def near_live_edge(self, unit: int) -> bool:
        """Whether streaming will reach `unit` next, so waiting beats restarting."""
        window = self.window
        return bool(
            window
            and window.cued
            and not window.complete
            and not window.failed
            and window.anchor <= unit <= window.streaming_unit + 1
        )

    async def reconcile(self) -> None:
        """Make mpv's pause and speed match the UI."""
        seek = self.seek
        if seek and seek.stage is Stage.BUFFERING:
            desired: bool | None = True  # Silent while speech is brought to the target.
        elif seek and seek.stage is Stage.QUEUED:
            desired = None  # Repeated presses may follow; leave speech as it is.
        else:
            desired = self.want_paused
        if desired is not None and desired != self.applied_paused:
            await self.player.set_paused(desired)
            self.applied_paused = desired
        if self.speed != self.applied_speed:
            await self.player.set_speed(self.speed)
            self.applied_speed = self.speed

    async def feed(self, window: Window) -> None:
        """Give mpv the audio streamed since the last step."""
        while window.parts:
            part = window.parts.pop(0)
            if not window.loaded:
                # Always load silently; reconcile decides when speech starts.
                await self.player.load(part.path, self.speed, paused=True)
                self.applied_paused, self.applied_speed = True, self.speed
                window.loaded = True
                self.heard, self.heard_ended = self.timeline.anchor_time, False
            else:
                await self.player.append(part.path, part.duration)
            window.duration += part.duration
            log.debug(
                "appended %s (%.2fs); %.2fs buffered ahead of %.2fs",
                part.path.name,
                part.duration,
                window.duration - self.heard,
                self.heard,
            )

    def observe(self, position: float, ended: bool) -> None:
        self.log_playback(position, bool(self.applied_paused), ended)
        self.heard, self.heard_ended = position, ended

    async def open_window(self, unit: int, start: float) -> None:
        """Start a playlist from the cached run just before `unit`, which starts at `start`.

        Earlier paragraphs are never generated, but cached ones join the playlist,
        so seeking back into them needs no restart. Placing `unit` where the UI showed
        it keeps the time continuous even if estimates changed since.
        """
        anchor = unit
        while anchor > 0 and anchor - 1 in self.timeline.known:
            anchor -= 1
        log.info("restarting playback at unit %d (playlist from unit %d)", unit, anchor)
        await self.close_window()
        self.timeline.set_anchor(anchor)
        if anchor:
            shift = start - self.timeline.start(unit)
            self.timeline.anchor_time = max(0.0, self.timeline.anchor_time + shift)
        window = Window(anchor, self.timeline.anchor_time)
        self.window = window
        self.generation_error = ""
        await self.player.start()
        # On Windows, files mpv still holds open cannot be deleted.
        self.spool = tempfile.TemporaryDirectory(prefix="qwen-stream-", ignore_cleanup_errors=True)
        window.worker = self.run_worker(
            self.stream_window(window, Path(self.spool.name)), group="stream"
        )

    async def close_window(self) -> None:
        window, self.window = self.window, None
        if window and window.worker:
            window.worker.cancel()
            with contextlib.suppress(WorkerCancelled, WorkerFailed, TimeoutError):
                async with asyncio.timeout(2):
                    await window.worker.wait()
        if window:
            with contextlib.suppress(PlayerError):
                await self.player.stop()
        self.applied_paused = None
        self.heard_ended = False
        self.cleanup_spool()

    async def stream_window(self, window: Window, spool: Path) -> None:
        """Queue streamed audio for the controller; never touches mpv itself."""
        stream = self.synthesizer.stream(
            self.article_text,
            self.synthesis_progress,
            spool,
            lambda cue: self.receive_cue(window, cue),
            first=window.anchor,
            room=lambda: self.has_room(window),
        )
        try:
            async with contextlib.aclosing(stream):
                async for part in stream:
                    window.parts.append(part)
                    self.wake()
            window.complete = True
        except (httpx.HTTPError, ValueError, OSError, TimeoutError) as exc:
            window.failed = self.describe_failure(exc)
            if window is self.window:
                self.generation_error = window.failed
        finally:
            self.wake()

    def receive_cue(self, window: Window, cue: Cue) -> None:
        unit = self.unit_of.get(cue.start)
        if unit is None or window is not self.window:
            return
        if not window.cued:
            window.cued = True
            # A fully cached article plays from its start, whatever was asked for.
            if unit != window.anchor:
                self.timeline.set_anchor(unit)
                window.anchor, window.duration = unit, self.timeline.anchor_time
        start = self.timeline.anchor_time
        end = None if cue.end_time is None else start + cue.end_time
        window.cues[cue.start] = Cue(cue.start, cue.end, start + cue.time, end)
        if cue.end_time is not None:
            self.timeline.known[unit] = cue.end_time - cue.time
        window.streaming_unit = max(window.streaming_unit, unit)

    def has_room(self, window: Window) -> bool:
        if window is not self.window:
            return False
        # Measured from what the UI shows: streaming reaches a seek target, then stops
        # three minutes past it, as it does past speech.
        buffered = window.duration + sum(part.duration for part in window.parts)
        return buffered - self.position < LOOKAHEAD * self.speed

    def synthesis_progress(self, done: int, total: int, message: str) -> None:
        self.synthesis_message = message

    def log_playback(self, position: float, paused: bool, ended: bool) -> None:
        now = time.monotonic()
        buffering = ended and self.preparing
        if buffering and self.buffering_since is None:
            self.buffering_since = now
            log.warning("ran out of audio at %.2fs; %.2fs received", position, self.duration)
        elif not buffering and self.buffering_since is not None:
            log.info("audio resumed after %.2fs of buffering", now - self.buffering_since)
            self.buffering_since = None
        if paused or ended or self.seek is not None:
            self.playing_sample = None
            return
        if self.playing_sample:
            then, previous = self.playing_sample
            elapsed, advanced = now - then, position - previous
            if elapsed > 0.5:
                log.warning("playback poll delayed: %.2fs since the previous poll", elapsed)
            # Seeks move the position backwards or too far forwards; only a shortfall is a stall.
            if 0 <= advanced < elapsed * self.speed - 0.25:
                log.warning(
                    "playback stalled near %.2fs: advanced %.2fs in %.2fs",
                    position,
                    advanced,
                    elapsed,
                )
        self.playing_sample = (now, position)

    def checkpoint(self, *, force: bool = False) -> None:
        """Save the position the UI shows; never asks mpv, so it is safe while quitting."""
        if not self.article_loaded or not self.current_url:
            return
        # Until a restore lands, the saved bookmark is still the truth.
        if self.seek and self.seek.restore or (self.window is None and self.seek is None):
            return
        if not force and time.monotonic() - self.last_saved_at < 2:
            return
        position = self.position
        if self.seek:
            unit, offset = self.seek.unit, self.seek.offset
        else:
            unit, offset = self.position_unit(position)
        bookmark = Bookmark(
            url=self.current_url,
            narration=self.narration_id,
            position=max(0.0, position),
            speed=self.speed,
            paused=self.want_paused,
            completed=self.ended,
            model=self.settings.model,
            voice=self.settings.voice,
            configuration=self.startup_configuration,
            unit=-1 if unit is None else unit,
            unit_offset=offset,
        )
        self.last_saved_at = time.monotonic()
        try:
            self.playback_state.save(bookmark)
            self.save_warning_shown = False
        except OSError:
            if self.is_running and not self.save_warning_shown:
                self.notify(
                    "Could not save playback position. Check the cache directory permissions.",
                    severity="warning",
                )
                self.save_warning_shown = True

    # --- Rendering: the UI from state, without asking mpv ------------------------------

    def render_playback(self) -> None:
        loaded = self.article_loaded and not self.halted
        for button in ("play", "rewind", "forward"):
            self.query_one(f"#{button}", Button).disabled = not loaded
        if not loaded:
            return
        position = self.position
        total = max(self.timeline.total(), position)
        self.query_one("#progress", ProgressBar).update(total=total, progress=position)
        # ≈ marks times that include estimates for paragraphs not generated yet.
        approx = "" if self.timeline.exact() else "≈ "
        timeline = f"{approx}{timestamp(position)} / {approx}{timestamp(total)}"
        if self.preparing:
            ahead = max(0.0, self.duration - position)
            timeline += f"    ·    {timestamp(ahead)} buffered ahead"
        else:
            timeline += f"    ·    {timestamp((total - position) / self.speed)} remaining"
        self.query_one("#timeline", Static).update(timeline)
        self.query_one("#play", Button).label = (
            "Replay" if self.ended else "Play" if self.want_paused else "Pause"
        )
        self.update_reading()
        self.update_playback_status()

    def update_playback_status(self) -> None:
        seek = self.seek
        if self.generation_error:
            self.set_status(f"Stopped · {self.generation_error}", error=True)
            return
        if seek and seek.stage is Stage.BUFFERING:
            if seek.reason == "restore":
                self.set_status(
                    f"Restoring {timestamp(self.position)} · waiting for audio… · Esc to cancel"
                )
            elif seek.reason == "start":
                self.set_status(f"{self.synthesis_message or 'Preparing'} · Esc to cancel")
            else:
                self.set_status(f"Waiting for paragraph {seek.unit + 1} to buffer… · Esc to cancel")
            return
        if self.ended:
            state = "Finished · Space to replay"
        elif self.want_paused:
            state = "Paused"
        elif self.heard_ended and self.preparing and seek is None:
            state = "Buffering…"
        else:
            state = "Playing"
        unit = self.ui_unit()
        if unit is not None:
            state = f"Paragraph {unit + 1}/{len(self.spans)} · {state}"
        if self.preparing and self.synthesis_message:
            state += f" · {self.synthesis_message} · Esc to cancel"
        self.set_status(state)

    def update_reading(self) -> None:
        text = self.query_one(ArticleText)
        current = pending = None
        if self.article_loaded:
            seek = self.seek
            if seek is None or seek.reason == "seek" and seek.stage is not Stage.BUFFERING:
                current = self.ui_unit()
            elif seek.reason == "seek":
                pending = seek.unit
        text.highlight(current, pending)
        self.query_one(ArticleView).follow(current if current is not None else pending)

    def reset_reading(self) -> None:
        self.query_one(ArticleText).highlight(None)
        self.query_one(ArticleView).set_following(True)
        self.follow_changed()

    @on(ArticleView.FollowChanged)
    def follow_changed(self) -> None:
        following = self.query_one(ArticleView).following
        button = self.query_one("#follow", Button)
        button.label = "Following" if following else "Resume sync"
        button.disabled = following
        button.variant = "primary" if following else "warning"

    # --- What the UI asks for: handlers record intent and return -----------------------

    def request_seek(
        self, unit: int, offset: float = 0.0, *, debounce: bool = False, at: float | None = None
    ) -> None:
        delay = SEEK_DEBOUNCE if debounce else 0.0
        shown = self.unit_time(unit) + offset if at is None else at
        self.seek = Seek(unit, offset, shown, time.monotonic() + delay)
        self.restored_finished = False
        log.debug("seek requested: unit %d + %.2fs", unit, offset)
        self.render_playback()
        self.wake()

    def action_scroll_page(self, direction: int) -> None:
        view = self.query_one(ArticleView)
        view.set_following(False)
        if direction > 0:
            view.scroll_page_down()
        else:
            view.scroll_page_up()

    def action_follow(self) -> None:
        self.resume_following()

    @on(Button.Pressed, "#follow")
    def resume_following(self) -> None:
        self.query_one(ArticleView).set_following(True)
        self.update_reading()
        self.query_one(ArticleView).focus()

    @on(ArticleText.Selected)
    def select_text(self, event: ArticleText.Selected) -> None:
        if self.can_seek() and 0 <= event.index < len(self.spans):
            self.request_seek(event.index)

    def action_paragraph(self, delta: int) -> None:
        if not self.can_seek():
            return
        base = self.ui_unit()
        if base is None:
            base = -1 if delta > 0 else 0
        index = max(0, min(len(self.spans) - 1, base + delta))
        if delta > 0 and index == base:
            return
        # Show the chosen paragraph even while browsing without following.
        self.query_one(ArticleView).reveal(index)
        self.request_seek(index, debounce=True)

    def action_read_visible(self) -> None:
        if not self.can_seek():
            return
        view = self.query_one(ArticleView)
        text = self.query_one(ArticleText)
        top = view.scroll_y - text.virtual_region.y
        index = next(
            (
                index
                for index in range(len(text.spans))
                if (region := text.reading_region(index)) and region.bottom > top
            ),
            None,
        )
        if index is None:
            return
        self.want_paused = False
        self.request_seek(index)
        self.resume_following()

    @on(Button.Pressed, "#play")
    def action_toggle_pause(self) -> None:
        if not self.can_seek():
            return
        if self.ended:
            self.want_paused = False
            self.request_seek(0)
            return
        self.want_paused = not self.want_paused
        self.checkpoint(force=True)
        self.render_playback()
        self.wake()

    def action_seek(self, offset: float) -> None:
        if not self.can_seek():
            return
        end = self.duration if self.generation_complete else self.timeline.total()
        target = max(0.0, min(end - 0.01, self.position + offset))
        unit, into = self.position_unit(target)
        if unit is not None:
            self.request_seek(unit, into, debounce=True, at=target)

    def action_speed(self, delta: float) -> None:
        self.speed = round(max(0.5, min(3.0, self.speed + delta)), 1)
        self.query_one("#speed", Static).update(f"{self.speed:.1f}×")
        self.checkpoint(force=True)
        self.wake()

    @on(Button.Pressed, "#rewind")
    def rewind(self) -> None:
        self.action_seek(-10)

    @on(Button.Pressed, "#forward")
    def forward(self) -> None:
        self.action_seek(10)

    @on(Button.Pressed, "#slower")
    def slower(self) -> None:
        self.action_speed(-0.1)

    @on(Button.Pressed, "#faster")
    def faster(self) -> None:
        self.action_speed(0.1)

    def action_focus_url(self) -> None:
        self.query_one("#url", Input).focus()

    def action_command_palette(self) -> None:
        if not self.closing and self.use_command_palette and not ReaderCommandPalette.is_open(self):
            self.push_screen(ReaderCommandPalette(id="--command-palette"))

    @on(Button.Pressed, "#choose-model")
    @work(group="model-choice", exclusive=True)
    async def action_choose_model(self) -> None:
        if self.closing:
            return
        choice = await self.push_screen_wait(ModelScreen(self.settings))
        if choice is None or self.closing:
            return
        model, voice = choice
        if model == self.settings.model and voice == self.settings.voice:
            return
        self.checkpoint(force=True)
        self.settings = replace(self.settings, model=model, voice=voice)
        self.synthesizer = Synthesizer(self.settings, self.client)
        await self.halt()
        try:
            self.playback_state.save_choice(ModelChoice(model, voice, self.startup_configuration))
        except OSError:
            self.notify(
                "Could not save the model choice. Check the cache directory permissions.",
                severity="warning",
            )
        self.reset_reading()
        self.query_one("#progress", ProgressBar).update(total=100, progress=0)
        self.query_one("#timeline", Static).update("00:00 / 00:00")
        self.query_one("#article-meta", Static).update(f"{model} · {self.settings.voice_id}")
        self.set_status("Model changed. Press Read to stream this article.")

    async def halt(self) -> None:
        """Stop speech until the next Read; the article stays on screen."""
        self.halted = True
        self.seek = None
        if self.prepare_worker and self.loading:
            self.prepare_worker.cancel()
        async with self.mpv_lock:
            await self.close_window()
        self.render_playback()

    async def action_cancel(self) -> None:
        if self.preparing or (self.seek is not None and not self.halted):
            self.checkpoint(force=True)
            await self.halt()
            self.reset_reading()
            self.set_status("Preparation cancelled. Completed audio chunks are cached.")
        self.query_one("#article-view", VerticalScroll).focus()

    # --- Shutdown --------------------------------------------------------------------

    def action_quit(self) -> None:
        if self.closing:
            return
        log.info("quit requested")
        self.checkpoint(force=True)
        self.closing = True
        self.wake()
        self.exit()

    async def on_unmount(self) -> None:
        self.closing = True
        self.wake()
        started = time.monotonic()
        workers = [self.controller, self.prepare_worker]
        if self.window:
            workers.append(self.window.worker)
        for worker in filter(None, workers):
            worker.cancel()
        for worker in filter(None, workers):
            with contextlib.suppress(WorkerCancelled, WorkerFailed, TimeoutError):
                async with asyncio.timeout(2):
                    await worker.wait()
        # Never let a stuck player keep the reader open.
        with contextlib.suppress(TimeoutError, PlayerError):
            async with asyncio.timeout(5):
                await self.player.close()
        log.info("unmount: stopped in %.2fs", time.monotonic() - started)
        self.cleanup_spool()
        await self.client.aclose()

    def cleanup_spool(self) -> None:
        if self.spool:
            self.spool.cleanup()
            self.spool = None
