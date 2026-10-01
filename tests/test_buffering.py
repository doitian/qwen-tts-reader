import asyncio
import json
import time

import httpx
import pytest
from test_app import FakePlayer, until
from textual.widgets import Static

from qwen_reader.app import ReaderApp
from qwen_reader.config import Settings
from qwen_reader.playback_state import PlaybackState
from qwen_reader.player import MpvPlayer, find_mpv

PARAGRAPHS = [f"Paragraph {i}." for i in range(10)]
BODY = "---\ntitle: Opening\n---\n" + "\n\n".join(PARAGRAPHS)


def reader(tmp_path, tts_endpoint, sse_audio, requests, player):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, text=BODY)
        requests.append(json.loads(request.content)["input"]["text"])
        return sse_audio(duration=60)

    return ReaderApp(
        Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path),
        "https://example.test",
        player=player,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def landed(pilot, app, anchor, seconds=5.0):
    await until(pilot, lambda: app.timeline.anchor == anchor and app.seek is None, seconds)


async def test_buffers_three_minutes_ahead_and_follows_playback(tmp_path, tts_endpoint, sse_audio):
    requests, player = [], FakePlayer()
    app = reader(tmp_path, tts_endpoint, sse_audio, requests, player)
    async with app.run_test() as pilot:
        # Each paragraph is a minute: stop once three minutes are buffered (plus lookahead).
        await until(pilot, lambda: app.duration >= 240)
        await pilot.pause(0.6)
        assert requests == ["Opening.", "Paragraph 0.", "Paragraph 1.", "Paragraph 2."]
        assert app.preparing
        await app.refresh_playback()
        timeline = str(app.query_one("#timeline", Static).render())
        assert timeline.startswith("≈ 00:00 / ≈ ")
        assert "04:00 buffered ahead" in timeline

        player.position = 100
        await until(pilot, lambda: len(requests) == 5)
        assert requests[-1] == "Paragraph 3."

        # A double speed listener needs twice the audio for the same three minutes.
        app.action_speed(1)
        assert app.has_room(app.window)


async def test_seeking_far_ahead_restarts_there_and_reuses_cached_paragraphs(
    tmp_path, tts_endpoint, sse_audio
):
    requests, player = [], FakePlayer()
    app = reader(tmp_path, tts_endpoint, sse_audio, requests, player)
    async with app.run_test() as pilot:
        await until(pilot, lambda: len(requests) == 4 and app.duration >= 240)
        app.request_seek(8)
        assert app.query_one("#article-text").current == 8  # The UI shows the target at once.
        await landed(pilot, app, 8)
        # Skipped paragraphs are never generated; the target streams immediately.
        assert requests[4] == "Paragraph 7."
        assert "Paragraph 3." not in requests
        assert app.query_one("#article-text").current == 8
        assert app.position == app.timeline.start(8)
        app.checkpoint(force=True)
        bookmark = PlaybackState(tmp_path / "playback.json").get(app.current_url, app.narration_id)
        assert (bookmark.unit, bookmark.unit_offset) == (8, 0)

        # Cached paragraphs come back without new requests and stay in the playlist.
        before = len(requests)
        app.request_seek(2)
        await landed(pilot, app, 0)
        assert app.query_one("#article-text").current == 2
        assert app.position == 120
        assert player.position == 120
        cached = {"Opening.", "Paragraph 0.", "Paragraph 1.", "Paragraph 2."}
        assert not cached & set(requests[before:])
        assert app.timeline.exact(4)


async def test_rewinding_past_the_playlist_start_restarts_one_paragraph_earlier(
    tmp_path, tts_endpoint, sse_audio
):
    requests, player = [], FakePlayer()
    app = reader(tmp_path, tts_endpoint, sse_audio, requests, player)
    async with app.run_test() as pilot:
        await until(pilot, lambda: len(requests) == 4)
        app.request_seek(8)
        await landed(pilot, app, 8)
        target = app.position - 10
        app.action_seek(-10)
        assert app.position == target  # The UI is the source of truth while seeking.
        await landed(pilot, app, 7)
        assert "Paragraph 6." in requests
        assert "Paragraph 5." not in requests
        # Paragraph 7 is placed where the UI showed it, so speech lands at the target,
        # unless its real length is shorter than estimated and the target is past its end.
        assert app.query_one("#article-text").current == 7
        assert target - 5 < app.position <= target + 0.01


async def test_back_to_back_restarts_keep_playing_once_the_target_arrives(
    tmp_path, tts_endpoint, sse_audio
):
    requests, player = [], FakePlayer()
    app = reader(tmp_path, tts_endpoint, sse_audio, requests, player)
    async with app.run_test() as pilot:
        await until(pilot, lambda: len(requests) == 4 and app.seek is None)
        assert not player.paused
        app.request_seek(8)
        await until(pilot, lambda: app.window.anchor == 8)
        # Replaced while the first restart may still be silent, waiting for its target.
        app.request_seek(10)
        await landed(pilot, app, 10)
        assert not player.paused and not app.paused

        # While paused, a restart stays paused; play/pause during the wait changes that.
        app.action_toggle_pause()
        await until(pilot, lambda: player.paused)
        app.request_seek(5)
        app.action_toggle_pause()
        await landed(pilot, app, 5)
        assert app.query_one("#article-text").current == 5
        assert not player.paused


async def test_keys_and_quit_never_wait_for_a_stuck_player(tmp_path, tts_endpoint, sse_audio):
    class StuckPlayer(FakePlayer):
        """Every seek hangs, as if mpv stopped answering."""

        def __init__(self):
            super().__init__()
            self.stuck = asyncio.Event()

        async def seek(self, position):
            self.stuck.set()
            await asyncio.Event().wait()

    requests, player = [], StuckPlayer()
    app = reader(tmp_path, tts_endpoint, sse_audio, requests, player)
    async with app.run_test() as pilot:
        await until(pilot, lambda: len(requests) == 4 and app.seek is None)
        app.query_one("#article-view").focus()
        await pilot.press("j", "j")
        await asyncio.wait_for(player.stuck.wait(), 2)
        # The controller is blocked inside mpv, yet keys still update the UI at once.
        started = time.monotonic()
        await pilot.press("j", "k", "j")
        assert app.seek.unit == 3
        assert app.query_one("#article-text").current == 3
        await pilot.press("q")
        await until(pilot, lambda: not app.is_running)
        assert time.monotonic() - started < 3
    assert player.closed


@pytest.mark.skipif(not find_mpv(), reason="mpv is not installed")
async def test_real_mpv_restarts_reuse_cache_and_keep_time_continuous(
    tmp_path, tts_endpoint, sse_audio
):
    requests = []
    app = reader(tmp_path, tts_endpoint, sse_audio, requests, MpvPlayer(audio_output="null"))
    async with app.run_test() as pilot:
        # Cached-paragraph bursts and file switches must not surface as player errors.
        await until(pilot, lambda: app.ready and len(requests) >= 4 and app.seek is None, 10)
        app.request_seek(8)
        await landed(pilot, app, 8, 10)
        assert app.position == pytest.approx(app.timeline.start(8), abs=0.3)
        app.action_seek(-10)
        await landed(pilot, app, 7, 10)
        assert app.timeline.start(7) < app.position < app.timeline.start(8)
        app.request_seek(2)
        await landed(pilot, app, 0, 10)
        assert app.position == pytest.approx(120, abs=0.3)
        assert not app.generation_error
