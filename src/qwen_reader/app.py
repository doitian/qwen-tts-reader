from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import httpx
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.timer import Timer
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
from .synthesis import Synthesizer
from .timeline import Timeline

log = logging.getLogger(__name__)

SEEK_DEBOUNCE = 0.25
# Generate at most this much listening time ahead of playback, in seconds.
LOOKAHEAD = 180


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


class ReaderApp(App):
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
        self.ready = False
        self.preparing = False
        self.generation_complete = False
        self.position = 0.0
        self.duration = 0.0
        self.speed = 1.0
        self.paused = False
        self.ended = False
        self.prepare_worker = None
        self.spool: tempfile.TemporaryDirectory | None = None
        self.synthesis_message = ""
        self.generation_error = ""
        self.cues: dict[int, Cue] = {}
        self.pending_selection: int | None = None
        self.current_url = ""
        self.narration_id = ""
        self.playback_loaded = False
        self.restoring: Bookmark | None = None
        self.restored_finished = False
        self.last_saved_at = 0.0
        self.save_warning_shown = False
        self.closing = False
        self.buffering_since: float | None = None
        self.seek_target: float | None = None
        self.seek_timer: Timer | None = None
        self.playing_sample: tuple[float, float] | None = None
        self.timeline = Timeline([])
        self.article_text = ""
        self.spans: list[TextSpan] = []
        self.unit_of: dict[int, int] = {}
        self.window_cued = False
        self.streaming_unit = -1
        self.pending_offset = 0.0
        self.unpause_after_seek = False
        self.restore_target: tuple[int, float] = (0, 0.0)

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
        self.set_interval(0.1, self.refresh_playback)
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

    def set_ready(self, ready: bool) -> None:
        self.ready = ready
        for button in ("play", "rewind", "forward"):
            self.query_one(f"#{button}", Button).disabled = not ready

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
        self.set_ready(False)
        self.prepare_worker = self.prepare(url, self.prepare_worker)

    @work(exclusive=True, group="prepare")
    async def prepare(self, url: str, previous: Worker | None = None) -> None:
        if previous:
            with contextlib.suppress(WorkerCancelled, WorkerFailed):
                await previous.wait()
        await self.checkpoint(force=True, query_player=True)
        self.playback_loaded = False
        self.restoring = None
        self.restored_finished = False
        self.set_ready(False)
        self.preparing = True
        self.generation_complete = False
        self.generation_error = ""
        self.synthesis_message = ""
        self.position = self.duration = 0
        self.reset_reading()
        self.query_one("#timeline", Static).update("00:00 / 00:00")
        self.query_one("#play", Button).label = "Play"
        self.query_one("#progress", ProgressBar).update(total=None)
        self.set_status("Fetching article with Defuddle… · Esc to cancel")
        try:
            await self.player.stop()
            self.cleanup_spool()
            # On Windows, files mpv still holds open cannot be deleted.
            self.spool = tempfile.TemporaryDirectory(
                prefix="qwen-stream-", ignore_cleanup_errors=True
            )
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
            self.query_one("#article-view", VerticalScroll).scroll_home(animate=False)
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
            self.restoring = self.playback_state.get(url, self.narration_id)
            if self.restoring:
                self.speed = self.restoring.speed
                self.query_one("#speed", Static).update(f"{self.speed:.1f}×")
                self.restore_target = self.bookmark_target(self.restoring)
            self.settings.validate_tts()
            await self.stream_window(self.restore_target[0] if self.restoring else 0)
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ValueError, OSError, PlayerError, TimeoutError) as exc:
            self.generation_error = self.describe_failure(exc)
        finally:
            self.finish_preparing()

    @work(exclusive=True, group="prepare")
    async def restart_window(
        self, unit: int, offset: float, paused: bool, previous: Worker | None = None
    ) -> None:
        if previous:
            with contextlib.suppress(WorkerCancelled, WorkerFailed):
                await previous.wait()
        self.restoring = None
        self.pending_selection, self.pending_offset = unit, offset
        try:
            await self.stream_window(unit, paused=paused)
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ValueError, OSError, PlayerError, TimeoutError) as exc:
            self.generation_error = self.describe_failure(exc)
        finally:
            self.finish_preparing()

    def restart_at(self, unit: int, offset: float = 0.0, *, paused: bool | None = None) -> None:
        """Replace the playlist with one from which `unit` (plus `offset`) can play."""
        log.info("restarting playback at unit %d + %.2fs", unit, offset)
        self.position = self.timeline.start(unit) + offset
        self.prepare_worker = self.restart_window(
            unit, offset, self.paused if paused is None else paused, self.prepare_worker
        )

    async def stream_window(self, unit: int, *, paused: bool = False) -> None:
        """Play from the cached run just before `unit`, then stream on while there is room.

        Earlier paragraphs are never generated, but cached ones join the playlist,
        so seeking back into them needs no restart.
        """
        anchor = unit
        while anchor > 0 and anchor - 1 in self.timeline.known:
            anchor -= 1
        self.preparing = True
        self.generation_complete = False
        self.generation_error = ""
        self.playback_loaded = False
        self.set_ready(False)
        self.cues.clear()
        self.window_cued = False
        self.streaming_unit = -1
        self.timeline.set_anchor(anchor)
        self.duration = self.timeline.anchor_time
        if self.restoring is None and self.pending_selection is None:
            self.position = self.duration
        await self.player.stop()
        self.cleanup_spool()
        # On Windows, files mpv still holds open cannot be deleted.
        self.spool = tempfile.TemporaryDirectory(prefix="qwen-stream-", ignore_cleanup_errors=True)
        await self.player.start()
        stream = self.synthesizer.stream(
            self.article_text,
            self.synthesis_progress,
            Path(self.spool.name),
            self.receive_cue,
            first=anchor,
            room=self.has_room,
        )
        async with contextlib.aclosing(stream):
            async for part in stream:
                if not self.is_running:
                    raise asyncio.CancelledError
                if not self.ready:
                    # Stay silent until a restore or a seek reaches its target.
                    waiting = self.restoring is not None or self.pending_selection is not None
                    await self.player.load(part.path, self.speed, paused=paused or waiting)
                    await self.player.set_speed(self.speed)
                    self.paused = paused or waiting
                    self.unpause_after_seek = waiting and not paused and self.restoring is None
                    self.ended = False
                    self.playback_loaded = True
                    self.set_ready(True)
                else:
                    await self.player.append(part.path, part.duration)
                self.duration += part.duration
                log.debug(
                    "received %s (%.2fs); %.2fs buffered ahead of %.2fs",
                    part.path.name,
                    part.duration,
                    self.duration - self.position,
                    self.position,
                )
                await self.restore_position()
                await self.apply_pending_seek()
                await self.checkpoint()
                try:
                    self.update_reading()
                    self.query_one("#progress", ProgressBar).update(
                        total=self.timeline.total(), progress=self.position
                    )
                    self.update_playback_status()
                except NoMatches:
                    # The screen is gone before on_unmount cancels this worker.
                    raise asyncio.CancelledError from None
        self.generation_complete = True
        await self.restore_position(final=True)
        await self.apply_pending_seek(final=True)

    def describe_failure(self, exc: BaseException) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"Request failed (HTTP {exc.response.status_code}). Try again."
        if isinstance(exc, httpx.HTTPError):
            return "Network request failed. Check your connection and try again."
        return str(exc) or "Audio preparation timed out. Try again."

    def finish_preparing(self) -> None:
        self.preparing = False
        self.pending_selection = None
        if self.is_running:
            self.update_reading()
            if not self.ready:
                self.query_one("#progress", ProgressBar).update(total=100, progress=0)
            if self.generation_error:
                self.set_status(self.generation_error, error=True)
            elif self.ready:
                self.update_playback_status()

    def has_room(self) -> bool:
        # Until a restore or seek reaches its target, keep streaming toward it.
        if self.restoring is not None or self.pending_selection is not None:
            return True
        return self.duration - self.position < LOOKAHEAD * self.speed

    def bookmark_target(self, bookmark: Bookmark) -> tuple[int, float]:
        last = len(self.timeline.lengths) - 1
        if bookmark.completed:
            return last, self.timeline.duration(last)
        if 0 <= bookmark.unit <= last:
            return bookmark.unit, bookmark.unit_offset
        # Older bookmarks store only seconds; their earlier paragraphs are cached.
        return self.timeline.locate(bookmark.position)

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

    def near_live_edge(self, unit: int) -> bool:
        """Whether streaming will reach `unit` next, so waiting beats restarting."""
        return (
            self.preparing
            and self.window_cued
            and self.timeline.anchor <= unit <= self.streaming_unit + 1
        )

    def synthesis_progress(self, done: int, total: int, message: str) -> None:
        if not self.is_running:
            return
        self.synthesis_message = message
        if self.ready:
            self.update_playback_status()
        else:
            self.set_status(message + " · Esc to cancel")

    def update_playback_status(self, *, buffering: bool = False) -> None:
        state = "Paused" if self.paused else "Buffering…" if buffering else "Playing"
        if self.ended:
            state = "Finished · Space to replay"
        current = next((text.current for text in self.query(ArticleText)), None)
        if current is not None:
            state = f"Paragraph {current + 1}/{len(self.spans)} · {state}"
        if self.generation_error:
            self.set_status(f"{state} received audio · {self.generation_error}", error=True)
        elif self.restoring:
            self.set_status(
                f"Restoring {timestamp(self.restoring.position)}"
                " · waiting for audio… · Esc to cancel"
            )
        elif self.pending_selection is not None:
            self.set_status(
                f"Waiting for paragraph {self.pending_selection + 1} to buffer… · Esc to cancel"
            )
        elif self.preparing:
            self.set_status(f"{state} · {self.synthesis_message} · Esc to cancel")
        else:
            self.set_status(state)

    async def refresh_playback(self) -> None:
        if not self.ready or not self.is_running or self.closing:
            return
        try:
            position, paused, ended = await self.player_status()
        except PlayerError as exc:
            if not self.ready or not self.is_running:
                return
            self.set_ready(False)
            self.set_status(str(exc), error=True)
            return
        if not self.ready or not self.is_running or self.closing:
            return
        self.log_playback(position, paused, ended)
        if self.seek_target is not None:
            position, ended = self.seek_target, False
        self.position, self.paused = position, paused
        self.ended = self.restored_finished or (ended and not self.preparing)
        position = self.duration if self.ended else self.position
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
            "Replay" if self.ended else "Play" if self.paused else "Pause"
        )
        self.update_reading()
        self.update_playback_status(buffering=ended and self.preparing)
        await self.checkpoint()

    async def player_status(self) -> tuple[float, bool, bool]:
        position, paused, ended = await self.player.status()
        return self.timeline.anchor_time + position, paused, ended

    async def player_seek(self, position: float) -> None:
        await self.player.seek(max(0.0, position - self.timeline.anchor_time))

    async def seek_to(self, target: float) -> None:
        """Seek within the playlist, or restart it where `target` falls."""
        unit, offset = self.timeline.locate(target)
        loaded = self.timeline.anchor_time <= target < self.duration
        # At or past the buffered end, wait at the live edge if streaming gets there soon.
        live = target >= self.duration and (self.generation_complete or self.near_live_edge(unit))
        if loaded or live:
            await self.player_seek(min(target, self.duration - 0.01))
        else:
            self.restart_at(unit, offset)

    def log_playback(self, position: float, paused: bool, ended: bool) -> None:
        now = time.monotonic()
        buffering = ended and self.preparing
        if buffering and self.buffering_since is None:
            self.buffering_since = now
            log.warning("ran out of audio at %.2fs; %.2fs received", position, self.duration)
        elif not buffering and self.buffering_since is not None:
            log.info("audio resumed after %.2fs of buffering", now - self.buffering_since)
            self.buffering_since = None
        if paused or ended:
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

    async def checkpoint(self, *, force: bool = False, query_player: bool = False) -> None:
        if not self.playback_loaded or self.restoring or not self.current_url:
            return
        if not force and time.monotonic() - self.last_saved_at < 2:
            return
        if query_player and self.seek_target is None:
            with contextlib.suppress(PlayerError):
                self.position, self.paused, ended = await self.player_status()
                self.ended = self.restored_finished or (ended and not self.preparing)
        position = self.duration if self.ended else max(0, min(self.position, self.duration))
        unit, offset = self.position_unit(position)
        bookmark = Bookmark(
            url=self.current_url,
            narration=self.narration_id,
            position=position,
            speed=self.speed,
            paused=self.paused,
            completed=self.restored_finished or (self.ended and self.generation_complete),
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

    async def restore_position(self, *, final: bool = False) -> None:
        bookmark = self.restoring
        if bookmark is None or not self.ready:
            return
        unit, offset = self.restore_target
        cue = self.cues.get(self.spans[unit].start)
        if cue is None and not final:
            return
        target = self.duration if cue is None else cue.time + offset
        if not final and (bookmark.completed or self.duration <= target):
            return
        target = max(self.timeline.anchor_time, min(target, self.duration - 0.01))
        await self.player_seek(target)
        await self.player.set_paused(bookmark.paused or bookmark.completed)
        self.restoring = None
        self.restored_finished = bookmark.completed
        self.position, self.paused = target, bookmark.paused or bookmark.completed
        self.ended = bookmark.completed
        self.last_saved_at = 0
        await self.refresh_playback()
        await self.checkpoint(force=True)

    def reset_reading(self) -> None:
        self.cancel_queued_seek()
        self.cues.clear()
        self.pending_selection = None
        self.query_one(ArticleText).highlight(None)
        self.query_one(ArticleView).set_following(True)
        self.follow_changed()

    def receive_cue(self, cue: Cue) -> None:
        unit = self.unit_of.get(cue.start)
        if unit is None:
            return
        if not self.window_cued:
            self.window_cued = True
            # A fully cached article plays from its start, whatever was asked for.
            if unit != self.timeline.anchor:
                self.timeline.set_anchor(unit)
                self.duration = self.timeline.anchor_time
        start = self.timeline.anchor_time
        end = None if cue.end_time is None else start + cue.end_time
        self.cues[cue.start] = Cue(cue.start, cue.end, start + cue.time, end)
        if cue.end_time is not None:
            self.timeline.known[unit] = cue.end_time - cue.time
        self.streaming_unit = max(self.streaming_unit, unit)

    def update_reading(self) -> None:
        text = self.query_one(ArticleText)
        current = None
        if self.ready and not self.restoring:
            # mpv's source position already accounts for pause, speed, and seeks.
            position = self.duration - 0.001 if self.ended else self.position
            current = self.position_unit(position)[0]
        text.highlight(current, self.pending_selection)
        self.query_one(ArticleView).follow(current)

    @on(ArticleView.FollowChanged)
    def follow_changed(self) -> None:
        following = self.query_one(ArticleView).following
        button = self.query_one("#follow", Button)
        button.label = "Following" if following else "Resume sync"
        button.disabled = following
        button.variant = "primary" if following else "warning"

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
    async def select_text(self, event: ArticleText.Selected) -> None:
        await self.seek_paragraph(event.index)

    async def action_paragraph(self, delta: int) -> None:
        text = self.query_one(ArticleText)
        # Repeated presses move a queued selection that is still waiting for audio.
        base = self.pending_selection if self.pending_selection is not None else text.current
        if base is None:
            base = -1 if delta > 0 else 0
        index = max(0, min(len(text.spans) - 1, base + delta))
        if delta > 0 and index == base:
            return
        cue = self.cues.get(text.spans[index].start)
        if self.ready and cue is not None and cue.time < self.duration:
            self.restoring = None
            self.pending_selection = None
            await self.queue_seek(cue.time)
        else:
            await self.seek_paragraph(index)

    async def action_read_visible(self) -> None:
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
        if not await self.seek_paragraph(index, paused=False):
            return
        try:
            await self.player.set_paused(False)
            await self.refresh_playback()
        except PlayerError as exc:
            self.set_status(str(exc), error=True)
        self.resume_following()

    async def seek_paragraph(self, index: int, *, paused: bool | None = None) -> bool:
        """Seek to a paragraph, or restart there; False if waiting for it to stream."""
        if not self.ready and not self.preparing:
            return False
        text = self.query_one(ArticleText)
        if not 0 <= index < len(text.spans):
            return False
        self.cancel_queued_seek()
        self.restoring = None
        self.pending_selection, self.pending_offset = index, 0.0
        seeked = await self.apply_pending_seek()
        if self.pending_selection is not None and not self.near_live_edge(index):
            self.pending_selection = None
            self.restart_at(index, paused=paused)
            seeked = True
        self.update_playback_status()
        self.update_reading()
        return seeked

    async def apply_pending_seek(self, *, final: bool = False) -> bool:
        if self.pending_selection is None or not self.ready:
            return False
        span = self.spans[self.pending_selection]
        cue = self.cues.get(span.start)
        if cue is None or cue.time >= self.duration:
            return False
        target = cue.time + self.pending_offset
        # An offset into the unit waits for its audio, unless the unit is already complete.
        if target >= self.duration and cue.end_time is None and not final:
            return False
        target = min(target, (cue.end_time or self.duration) - 0.01, self.duration - 0.01)
        self.pending_selection, self.pending_offset = None, 0.0
        try:
            self.restored_finished = False
            await self.player_seek(target)
            if self.unpause_after_seek:
                self.unpause_after_seek = False
                await self.player.set_paused(False)
            self.position = target
            self.ended = False
            await self.refresh_playback()
            await self.checkpoint(force=True)
            return True
        except PlayerError as exc:
            self.set_status(str(exc), error=True)
            return False

    @on(Button.Pressed, "#play")
    async def action_toggle_pause(self) -> None:
        await self.flush_seek()
        if not self.ready:
            return
        try:
            self.restoring = None
            self.pending_selection = None
            _, paused, ended = await self.player.status()
            replay = self.restored_finished or (ended and not self.preparing)
            self.restored_finished = False
            if replay and self.timeline.anchor:
                self.restart_at(0, paused=False)
                return
            if replay:
                await self.player.seek(0)
            await self.player.set_paused(False if replay else not paused)
            await self.refresh_playback()
            await self.checkpoint(force=True)
        except PlayerError as exc:
            self.set_ready(False)
            self.set_status(str(exc), error=True)

    async def action_seek(self, offset: float) -> None:
        if not self.ready:
            return
        self.restoring = None
        self.pending_selection = None
        # Repeated presses build on the queued target, not mpv's not-yet-seeked position.
        if self.seek_target is not None:
            position = self.seek_target
        else:
            try:
                position, _, ended = await self.player_status()
            except PlayerError as exc:
                self.set_ready(False)
                self.set_status(str(exc), error=True)
                return
            if ended or self.restored_finished:
                position = self.duration
        end = self.duration if self.generation_complete else self.timeline.total()
        await self.queue_seek(max(0.0, min(end - 0.01, position + offset)))

    async def queue_seek(self, target: float) -> None:
        """Show the target at once, but seek mpv only after SEEK_DEBOUNCE without more seeks."""
        self.restored_finished = False
        self.seek_target = target
        if self.seek_timer:
            self.seek_timer.stop()
        self.seek_timer = self.set_timer(SEEK_DEBOUNCE, self.flush_seek)
        await self.refresh_playback()

    def cancel_queued_seek(self) -> None:
        if self.seek_timer:
            self.seek_timer.stop()
            self.seek_timer = None
        self.seek_target = None

    async def flush_seek(self) -> None:
        if self.seek_timer:
            self.seek_timer.stop()
            self.seek_timer = None
        target = self.seek_target
        if target is None:
            return
        if not self.ready:
            self.seek_target = None
            return
        try:
            await self.seek_to(target)
        except PlayerError as exc:
            self.seek_target = None
            self.set_ready(False)
            self.set_status(str(exc), error=True)
            return
        # A newer target queued during the seek keeps its own timer.
        if self.seek_target == target:
            self.seek_target = None
        await self.refresh_playback()
        await self.checkpoint(force=True)

    async def action_speed(self, delta: float) -> None:
        speed = round(max(0.5, min(3.0, self.speed + delta)), 1)
        try:
            if self.ready:
                await self.player.set_speed(speed)
            self.speed = speed
            self.query_one("#speed", Static).update(f"{speed:.1f}×")
            if self.restoring:
                self.restoring = replace(self.restoring, speed=speed)
            await self.checkpoint(force=True)
        except PlayerError as exc:
            self.set_status(str(exc), error=True)

    @on(Button.Pressed, "#rewind")
    async def rewind(self) -> None:
        await self.action_seek(-10)

    @on(Button.Pressed, "#forward")
    async def forward(self) -> None:
        await self.action_seek(10)

    @on(Button.Pressed, "#slower")
    async def slower(self) -> None:
        await self.action_speed(-0.1)

    @on(Button.Pressed, "#faster")
    async def faster(self) -> None:
        await self.action_speed(0.1)

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
        await self.action_cancel()
        await self.checkpoint(force=True, query_player=True)
        self.playback_loaded = False
        self.restoring = None
        self.restored_finished = False
        self.set_ready(False)
        with contextlib.suppress(PlayerError):
            await self.player.stop()
        self.cleanup_spool()
        self.settings = replace(self.settings, model=model, voice=voice)
        self.synthesizer = Synthesizer(self.settings, self.client)
        try:
            self.playback_state.save_choice(ModelChoice(model, voice, self.startup_configuration))
        except OSError:
            self.notify(
                "Could not save the model choice. Check the cache directory permissions.",
                severity="warning",
            )
        self.position = self.duration = 0
        self.reset_reading()
        self.query_one("#progress", ProgressBar).update(total=100, progress=0)
        self.query_one("#timeline", Static).update("00:00 / 00:00")
        self.query_one("#article-meta", Static).update(f"{model} · {self.settings.voice_id}")
        self.set_status("Model changed. Press Read to stream this article.")

    async def action_cancel(self) -> None:
        self.pending_selection = None
        if self.preparing and self.prepare_worker:
            self.prepare_worker.cancel()
            with contextlib.suppress(WorkerCancelled, WorkerFailed):
                await self.prepare_worker.wait()
            await self.checkpoint(force=True, query_player=True)
            self.playback_loaded = False
            self.set_ready(False)
            with contextlib.suppress(PlayerError):
                await self.player.stop()
            self.cleanup_spool()
            self.reset_reading()
            self.set_status("Preparation cancelled. Completed audio chunks are cached.")
        self.query_one("#article-view", VerticalScroll).focus()

    async def action_quit(self) -> None:
        if self.closing:
            return
        self.closing = True
        if self.prepare_worker:
            self.prepare_worker.cancel()
            with contextlib.suppress(WorkerCancelled, WorkerFailed):
                await self.prepare_worker.wait()
        await self.checkpoint(force=True, query_player=True)
        self.playback_loaded = False
        self.exit()

    async def on_unmount(self) -> None:
        self.closing = True
        if self.prepare_worker:
            self.prepare_worker.cancel()
            with contextlib.suppress(WorkerCancelled, WorkerFailed):
                await self.prepare_worker.wait()
        await self.checkpoint(force=True, query_player=True)
        self.playback_loaded = False
        await self.player.close()
        self.cleanup_spool()
        await self.client.aclose()

    def cleanup_spool(self) -> None:
        if self.spool:
            self.spool.cleanup()
            self.spool = None
