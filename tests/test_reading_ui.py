import asyncio
import json

import httpx
import pytest
from test_app import FakePlayer
from test_streaming import STOP, ByteStream, pcm_event
from textual import events
from textual.scrollbar import ScrollTo
from textual.widgets import Button, Input, Static

from qwen_reader.app import ReaderApp
from qwen_reader.article_view import ArticleText, ArticleView
from qwen_reader.config import Settings
from qwen_reader.narration import Cue


@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_highlight_follow_browse_click_resume_resize_and_cached_replay(
    tmp_path, tts_endpoint, sse_audio, size
):
    paragraphs = ["Opening"] + [f"第 {i} 段。阅读中文与 English 👋。 " * 4 for i in range(1, 14)]
    body = "---\ntitle: Opening\n---\n" + "\n\n".join(paragraphs)
    calls = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, text=body)
        calls.append(json.loads(request.content)["input"]["text"])
        return sse_audio(duration=2)

    player = FakePlayer()
    app = ReaderApp(
        Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path),
        "https://example.test",
        player=player,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    async with app.run_test(size=size) as pilot:
        await app.prepare_worker.wait()
        await pilot.pause()
        text = app.query_one(ArticleText)
        view = app.query_one(ArticleView)
        assert len(text.spans) == 14
        assert text.current == 0
        assert view.following
        assert app.query_one("#follow").region.right <= size[0]
        assert app.query_one("#article-view").content_region.height >= 4

        player.position = 16.5
        await app.refresh_playback()
        await pilot.pause()
        assert text.current == 8
        assert view.scroll_y > 0
        assert view.following  # Automatic scrolling never disables itself.
        highlighted = [
            s for s in text.render_line(text.reading_region(8).y) if s.style and s.style.bgcolor
        ]
        assert any(s.style.bgcolor.triplet == (131, 223, 205) for s in highlighted)

        view.focus()
        await pilot.press("home")
        await pilot.pause()
        assert not view.following
        assert str(app.query_one("#follow", Button).label) == "Resume sync"
        assert view.scroll_y == 0
        player.position = 22.5
        await app.refresh_playback()
        assert text.current == 11
        assert view.scroll_y == 0

        # Click the visible second paragraph while paused; it stays in browse mode.
        player.paused = True
        await app.refresh_playback()
        target = text.reading_region(1)
        assert target is not None
        await pilot.click("#article-text", offset=(1, target.y))
        await pilot.pause()
        assert player.position == 2
        assert player.paused
        assert text.current == 1
        assert not view.following

        await pilot.click("#follow")
        await pilot.pause()
        assert view.following
        player.position = 24.5
        await app.refresh_playback()
        await pilot.pause()
        assert text.current == 12
        assert view.scroll_y > 0
        await pilot.resize_terminal(70, 26)
        await pilot.pause()
        current_region = text.reading_region(12)
        assert current_region is not None
        assert view.scroll_y <= current_region.y < view.scroll_y + view.content_size.height
        assert view.following

        # Speed and pause use the source timeline; replay returns to the first unit.
        await app.action_speed(1)
        assert text.current == 12
        player.ended = True
        await app.refresh_playback()
        await app.action_toggle_pause()
        assert text.current == 0
        assert not player.paused
        calls_before = len(calls)
        app.begin_load()
        await app.prepare_worker.wait()
        await pilot.pause()
        assert len(calls) == calls_before
        player.position = 8.1
        await app.refresh_playback()
        assert text.current == 4


@pytest.mark.parametrize(
    "gesture",
    [
        "wheel_up",
        "wheel_down",
        "scrollbar",
        "pagedown",
        "pageup",
        "ctrl+f",
        "ctrl+b",
        "down",
        "up",
        "home",
        "end",
    ],
)
async def test_manual_scroll_disables_follow_until_button(gesture, tmp_path):
    app = ReaderApp(Settings(cache_dir=tmp_path), player=FakePlayer(), client=httpx.AsyncClient())
    async with app.run_test() as pilot:
        text = app.query_one(ArticleText)
        view = app.query_one(ArticleView)
        text.set_article("\n\n".join("Paragraph " + str(i) for i in range(40)), 1800)
        await pilot.pause()
        view.focus()
        if gesture.startswith("wheel"):
            event_type = events.MouseScrollUp if gesture == "wheel_up" else events.MouseScrollDown
            text.post_message(event_type(text, 0, 0, 0, 0, 0, False, False, False))
        elif gesture == "scrollbar":
            view.post_message(ScrollTo(y=10, animate=False))
        else:
            await pilot.press(gesture)
        await pilot.pause()
        assert not view.following
        await pilot.click("#follow")
        await pilot.pause()
        assert view.following


async def test_vim_keys_seek_paragraphs_and_time_scroll_and_follow(
    tmp_path, tts_endpoint, sse_audio
):
    paragraphs = [f"Paragraph {i} reads aloud. " * 6 for i in range(30)]
    body = "---\ntitle: Opening\n---\n" + "\n\n".join(paragraphs)

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, text=body)
        return sse_audio(duration=2)

    player = FakePlayer()
    app = ReaderApp(
        Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path),
        "https://example.test",
        player=player,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await app.prepare_worker.wait()
        await pilot.pause()
        text = app.query_one(ArticleText)
        view = app.query_one(ArticleView)

        # Within the debounce, each press builds on the previous target.
        await pilot.press("j", "j", "j")
        assert app.position == 6
        assert text.current == 3
        await pilot.press("k")
        assert app.position == 4
        await pilot.press("l")
        assert app.position == 14
        assert text.current == 7
        await pilot.press("h")
        assert app.position == 4
        assert player.seeks == []
        await pilot.pause(0.3)
        assert player.seeks == [4]
        assert player.position == 4
        await pilot.press("k", "k", "k")
        await pilot.pause(0.3)
        assert player.seeks == [4, 0]
        # Play/pause applies a queued seek first instead of waiting for the debounce.
        await pilot.press("l", "space")
        assert player.seeks == [4, 0, 10]
        await pilot.press("space")
        await pilot.pause(0.3)
        assert player.seeks == [4, 0, 10]
        last = app.duration - 1
        player.position = last
        await app.refresh_playback()
        await pilot.press("j")
        await pilot.pause(0.3)
        assert player.position == last  # Already in the last paragraph.

        await pilot.press("ctrl+b")
        await pilot.pause()
        assert not view.following
        browsed = view.scroll_y
        await pilot.press("f")
        await pilot.pause()
        assert view.following
        assert view.scroll_y > browsed

        # r reads from the first paragraph with any line on screen, even a partial one.
        target = text.reading_region(10)
        assert target is not None and target.height > 1
        view.set_following(False)
        view.scroll_to(y=target.y + 1, animate=False)
        player.paused = True
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()
        assert player.position == 20
        assert not player.paused
        assert text.current == 10
        assert view.following

        last = player.position
        app.query_one("#url", Input).focus()
        await pilot.press("end", "j", "k", "h", "l", "f", "r")
        assert player.position == last
        assert app.query_one("#url", Input).value.endswith("jkhlfr")


async def test_click_unbuffered_paragraph_waits_then_seeks_without_resuming_follow(
    tmp_path, tts_endpoint
):
    gate = asyncio.Event()
    started = asyncio.Event()

    class Delayed(ByteStream):
        async def __aiter__(self):
            started.set()
            await gate.wait()
            yield pcm_event(b"\0\0" * 48000)
            yield STOP

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, text="---\ntitle: First\n---\nFirst\n\nSecond paragraph.")
        content = json.loads(request.content)["input"]["text"]
        stream = (
            Delayed([])
            if content.startswith("Second")
            else ByteStream([pcm_event(b"\0\0" * 24000), STOP])
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    player = FakePlayer()
    app = ReaderApp(
        Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path),
        "https://example.test",
        player=player,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    async with app.run_test() as pilot:
        await asyncio.wait_for(started.wait(), 2)
        await pilot.pause()
        assert app.preparing and app.ready
        text = app.query_one(ArticleText)
        view = app.query_one(ArticleView)
        view.set_following(False)
        player.paused = True
        await app.refresh_playback()
        await pilot.click("#article-text", offset=(0, text.reading_region(1).y))
        await pilot.pause()
        assert app.pending_selection == 1
        assert "Waiting for paragraph 2" in str(app.query_one("#status", Static).render())
        assert player.position == 0
        gate.set()
        await app.prepare_worker.wait()
        await pilot.pause()
        assert app.pending_selection is None
        assert player.position == 1
        assert player.paused
        assert text.current == 1
        assert not view.following
        await app.action_seek(-10)
        assert text.current == 0


async def test_text_cannot_seek_to_stale_audio_after_model_change(tmp_path):
    app = ReaderApp(Settings(cache_dir=tmp_path), player=FakePlayer(), client=httpx.AsyncClient())
    async with app.run_test() as pilot:
        text = app.query_one(ArticleText)
        text.set_article("First.\n\nSecond.", 1800)
        app.cues = {0: Cue(0, 6, 0, 1), 8: Cue(8, 15, 1, 2)}
        app.duration = 2
        app.set_ready(True)
        app.reset_reading()
        app.set_ready(False)
        await app.select_text(ArticleText.Selected(1))
        await pilot.pause()
        assert text.current is None
        assert app.player.position == 0
