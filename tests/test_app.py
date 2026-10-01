import asyncio

import httpx
import pytest
from textual.widgets import Button, Input, Select, Static

from qwen_reader.app import ReaderApp
from qwen_reader.config import Settings
from qwen_reader.model_screen import ModelScreen


class FakePlayer:
    def __init__(self):
        self.position = 0.0
        self.paused = False
        self.ended = False
        self.speed = 1.0
        self.closed = False
        self.loaded = []
        self.appended = []
        self.seeks = []

    async def start(self):
        pass

    async def stop(self):
        self.position = 0.0

    async def load(self, path, speed, *, paused=False):
        self.loaded.append(path)
        self.speed = speed
        self.paused = paused
        self.ended = False

    async def status(self):
        return self.position, self.paused, self.ended

    async def append(self, path, duration):
        self.appended.append(path)
        self.ended = False

    async def set_paused(self, paused):
        self.paused = paused

    async def seek(self, position):
        self.seeks.append(position)
        self.position = position
        self.ended = False

    async def set_speed(self, speed):
        self.speed = speed

    async def close(self):
        self.closed = True


def mock_services(sse_audio):
    def handler(request):
        if request.url.host == "defuddle.md":
            return httpx.Response(
                200, text='---\ntitle: "Test article"\n---\nAn interesting story.'
            )
        if request.method == "POST":
            return sse_audio(duration=30)
        pytest.fail("Unexpected network request")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_article_to_playback_keyboard_buttons_and_replay(
    tmp_path, sse_audio, size, tts_endpoint
):
    player = FakePlayer()
    app = ReaderApp(
        Settings(api_key="test", endpoint=tts_endpoint, cache_dir=tmp_path),
        player=player,
        client=mock_services(sse_audio),
    )
    async with app.run_test(size=size) as pilot:
        app.query_one("#url", Input).value = "https://example.com/story"
        await pilot.press("enter")
        await app.prepare_worker.wait()
        await pilot.pause()
        assert app.ready
        assert "Test article" in str(app.query_one("#article-title", Static).render())
        assert len(player.loaded) == 1
        assert not app.query_one("#play", Button).disabled
        assert app.query_one("#controls").region.bottom <= size[1] - 1
        assert app.query_one("#hint").region.bottom <= size[1] - 1
        assert app.query_one("#article-view").content_region.height >= 4
        await pilot.press("space")
        assert player.paused
        await pilot.press("right")
        assert app.position == 10  # Shown at once; mpv seeks after the debounce.
        assert player.seeks == []
        await pilot.pause(0.3)
        assert player.seeks == [10]
        assert player.paused
        await pilot.press("left", "left")
        await pilot.pause(0.3)
        assert player.seeks == [10, 0]  # Rapid presses collapse into one seek.
        await pilot.press("plus", "equals")
        assert player.speed == 1.2
        await pilot.click("#slower")
        assert player.speed == 1.1
        await pilot.click("#play")
        assert not player.paused
        await pilot.press("ctrl+l")
        assert isinstance(app.focused, Input)
        await pilot.press("left", "minus")
        assert player.speed == 1.1
        assert player.position == 0
        await pilot.press("escape")
        player.ended = True
        player.paused = True
        await app.refresh_playback()
        await pilot.press("space")
        assert not player.ended
        assert not player.paused
        assert player.position == 0
    assert player.closed
    assert app.client.is_closed


async def test_missing_key_still_shows_article_and_keeps_app_alive(
    tmp_path, sse_audio, tts_endpoint
):
    app = ReaderApp(
        Settings(endpoint=tts_endpoint, cache_dir=tmp_path),
        "https://example.com",
        player=FakePlayer(),
        client=mock_services(sse_audio),
    )
    async with app.run_test() as pilot:
        await app.prepare_worker.wait()
        await pilot.pause()
        assert not app.ready
        assert "QWEN_TTS_API_KEY" in str(app.query_one("#status", Static).render())
        assert "interesting story" in str(app.query_one("#article-text", Static).render())


async def test_cancel_and_replace_an_inflight_article(tmp_path, sse_audio, tts_endpoint):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    replacement = mock_services(sse_audio)

    async def handler(request):
        if "slow.test" in str(request.url):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return await replacement.send(request)

    app = ReaderApp(
        Settings(api_key="test", endpoint=tts_endpoint, cache_dir=tmp_path),
        "https://slow.test",
        player=FakePlayer(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    async with app.run_test() as pilot:
        await asyncio.wait_for(started.wait(), timeout=2)
        await pilot.press("escape")
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        assert "cancelled" in str(app.query_one("#status", Static).render())
        assert not app.ready
        app.query_one("#url", Input).value = "https://example.com"
        await pilot.click("#load")
        await app.prepare_worker.wait()
        assert app.ready
        assert len(app.player.loaded) == 1
    await replacement.aclose()


async def test_model_picker_resets_voice_and_changes_the_next_request(
    tmp_path, sse_audio, tts_endpoint
):
    player = FakePlayer()
    app = ReaderApp(
        Settings(api_key="test", endpoint=tts_endpoint, cache_dir=tmp_path),
        "https://example.com",
        player=player,
        client=mock_services(sse_audio),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await app.prepare_worker.wait()
        await pilot.click("#choose-model")
        await pilot.pause()
        assert isinstance(app.screen, ModelScreen)
        picker = app.screen
        picker.query_one("#model-choice", Select).value = "qwen-audio-3.0-tts-plus"
        await pilot.pause()
        assert picker.query_one("#voice-choice", Select).value == ""
        await pilot.click("#model-apply")
        await pilot.pause()
        assert app.settings.model == "qwen-audio-3.0-tts-plus"
        assert app.settings.voice_id == "longanlingxin"
        assert app.synthesizer.settings.model == "qwen-audio-3.0-tts-plus"
        assert not app.ready
        await pilot.click("#choose-model")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ModelScreen)
        assert app.settings.model == "qwen-audio-3.0-tts-plus"
        await pilot.click("#load")
        await app.prepare_worker.wait()
        assert app.ready
