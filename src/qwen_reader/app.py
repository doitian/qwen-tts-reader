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
from textual.timer import Timer
from textual.widgets import Button, Footer, Header, Input, ProgressBar, Static
from textual.worker import Worker, WorkerCancelled, WorkerFailed

from .article import fetch_article, normalize_url
from .article_view import ArticleText, ArticleView
from .config import MODEL_VOICES, Settings
from .model_screen import ModelScreen
from .narration import Cue
from .playback_state import Bookmark, PlaybackState
from .player import MpvPlayer, PlayerError
from .screens import ReaderCommandPalette
from .synthesis import Synthesizer

log = logging.getLogger(__name__)

SEEK_DEBOUNCE = 0.25


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
        if (
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
            self.query_one(ArticleText).set_article(article.text, self.settings.chunk_chars)
            self.query_one("#article-view", VerticalScroll).scroll_home(animate=False)
            self.current_url = url
            self.narration_id = self.synthesizer.cache_path(article.text, "article").stem
            self.restoring = self.playback_state.get(url, self.narration_id)
            if self.restoring:
                self.speed = self.restoring.speed
                self.query_one("#speed", Static).update(f"{self.speed:.1f}×")
            self.settings.validate_tts()
            await self.player.start()
            stream = self.synthesizer.stream(
                article.text, self.synthesis_progress, Path(self.spool.name), self.receive_cue
            )
            async with contextlib.aclosing(stream):
                async for part in stream:
                    if not self.is_running:
                        raise asyncio.CancelledError
                    if not self.ready:
                        await self.player.load(
                            part.path, self.speed, paused=self.restoring is not None
                        )
                        await self.player.set_speed(self.speed)
                        self.paused = self.restoring is not None
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
                    self.update_reading()
                    self.query_one("#progress", ProgressBar).update(
                        total=self.duration, progress=self.position
                    )
                    self.update_playback_status()
            self.generation_complete = True
            await self.restore_position(final=True)
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as exc:
            self.generation_error = f"Request failed (HTTP {exc.response.status_code}). Try again."
        except httpx.RequestError:
            self.generation_error = "Network request failed. Check your connection and try again."
        except (ValueError, OSError, PlayerError, TimeoutError) as exc:
            self.generation_error = str(exc) or "Audio preparation timed out. Try again."
        finally:
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
            position, paused, ended = await self.player.status()
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
        position = self.duration if self.ended else min(self.position, self.duration)
        self.query_one("#progress", ProgressBar).update(progress=position)
        if self.preparing:
            timeline = (
                f"{timestamp(position)} / {timestamp(self.duration)} buffered · receiving audio"
            )
        else:
            timeline = (
                f"{timestamp(position)} / {timestamp(self.duration)}"
                f"    ·    {timestamp((self.duration - position) / self.speed)} remaining"
            )
        self.query_one("#timeline", Static).update(timeline)
        self.query_one("#play", Button).label = (
            "Replay" if self.ended else "Play" if self.paused else "Pause"
        )
        self.update_reading()
        self.update_playback_status(buffering=ended and self.preparing)
        await self.checkpoint()

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
                self.position, self.paused, ended = await self.player.status()
                self.ended = self.restored_finished or (ended and not self.preparing)
        bookmark = Bookmark(
            url=self.current_url,
            narration=self.narration_id,
            position=self.duration if self.ended else max(0, min(self.position, self.duration)),
            speed=self.speed,
            paused=self.paused,
            completed=self.restored_finished or (self.ended and self.generation_complete),
            model=self.settings.model,
            voice=self.settings.voice,
            configuration=self.startup_configuration,
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
        if not final and (bookmark.completed or self.duration <= bookmark.position):
            return
        target = max(0, min(bookmark.position, self.duration - 0.01))
        await self.player.seek(target)
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
        self.cues[cue.start] = cue

    def update_reading(self) -> None:
        text = self.query_one(ArticleText)
        current = None
        if self.ready and not self.restoring:
            # mpv's source position already accounts for pause, speed, and seeks.
            position = self.duration if self.ended else self.position
            for index, span in enumerate(text.spans):
                cue = self.cues.get(span.start)
                if cue and cue.time <= position + 0.001 and cue.time < self.duration:
                    current = index
                elif cue and cue.time > position:
                    break
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
        if not await self.seek_paragraph(index):
            return
        try:
            await self.player.set_paused(False)
            await self.refresh_playback()
        except PlayerError as exc:
            self.set_status(str(exc), error=True)
        self.resume_following()

    async def seek_paragraph(self, index: int) -> bool:
        """Seek now if the paragraph is buffered, otherwise queue it; True if it seeked."""
        if not self.ready and not self.preparing:
            return False
        text = self.query_one(ArticleText)
        if not 0 <= index < len(text.spans):
            return False
        self.cancel_queued_seek()
        self.restoring = None
        self.pending_selection = index
        seeked = await self.apply_pending_seek()
        if self.pending_selection is not None and not self.preparing:
            self.pending_selection = None
            self.set_status(
                "This paragraph has no buffered audio. Press Read to retry.", error=True
            )
        else:
            self.update_playback_status()
        self.update_reading()
        return seeked

    async def apply_pending_seek(self) -> bool:
        if self.pending_selection is None or not self.ready:
            return False
        span = self.query_one(ArticleText).spans[self.pending_selection]
        cue = self.cues.get(span.start)
        if cue is None or cue.time >= self.duration:
            return False
        self.pending_selection = None
        try:
            self.restored_finished = False
            await self.player.seek(cue.time)
            self.position = cue.time
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
                position, _, ended = await self.player.status()
            except PlayerError as exc:
                self.set_ready(False)
                self.set_status(str(exc), error=True)
                return
            if ended or self.restored_finished:
                position = self.duration
        await self.queue_seek(max(0.0, min(self.duration - 0.01, position + offset)))

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
            await self.player.seek(target)
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
