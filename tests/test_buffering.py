import json

import httpx
import pytest
from test_app import FakePlayer
from textual.widgets import Static

from qwen_reader.app import ReaderApp
from qwen_reader.config import Settings
from qwen_reader.playback_state import PlaybackState
from qwen_reader.player import MpvPlayer, find_mpv

PARAGRAPHS = [f"Paragraph {i}." for i in range(10)]
BODY = "---\ntitle: Opening\n---\n" + "\n\n".join(PARAGRAPHS)


async def until(pilot, condition, seconds=5.0):
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition not reached")


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
        await app.refresh_playback()
        await until(pilot, lambda: len(requests) == 5)
        assert requests[-1] == "Paragraph 3."

        # A double speed listener needs twice the audio for the same three minutes.
        await app.action_speed(1)
        assert app.has_room()


async def test_seeking_far_ahead_restarts_there_and_reuses_cached_paragraphs(
    tmp_path, tts_endpoint, sse_audio
):
    requests, player = [], FakePlayer()
    app = reader(tmp_path, tts_endpoint, sse_audio, requests, player)
    async with app.run_test() as pilot:
        await until(pilot, lambda: len(requests) == 4 and app.duration >= 240)
        await app.seek_paragraph(8)
        await until(pilot, lambda: app.ready and app.timeline.anchor == 8)
        # Skipped paragraphs are never generated; the target streams immediately.
        assert requests[4] == "Paragraph 7."
        assert "Paragraph 3." not in requests
        assert app.query_one("#article-text").current == 8
        start = app.timeline.start(8)
        assert app.position == start
        await app.checkpoint(force=True)
        bookmark = PlaybackState(tmp_path / "playback.json").get(app.current_url, app.narration_id)
        assert (bookmark.unit, bookmark.unit_offset) == (8, 0)

        # Cached paragraphs come back without new requests and stay in the playlist.
        before = len(requests)
        await app.seek_paragraph(2)
        await until(pilot, lambda: app.ready and app.timeline.anchor == 0)
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
        await app.seek_paragraph(8)
        await until(pilot, lambda: app.ready and app.timeline.anchor == 8)
        target = app.position - 10
        await app.action_seek(-10)
        await until(pilot, lambda: app.ready and app.timeline.anchor == 7)
        assert "Paragraph 6." in requests
        assert "Paragraph 5." not in requests
        await until(pilot, lambda: app.pending_selection is None and 7 in app.timeline.known)
        await app.refresh_playback()
        # Times stay continuous across the restart: 10s before where the old playlist began.
        assert app.position == target
        assert app.query_one("#article-text").current == 7


@pytest.mark.skipif(not find_mpv(), reason="mpv is not installed")
async def test_real_mpv_restarts_reuse_cache_and_keep_time_continuous(
    tmp_path, tts_endpoint, sse_audio
):
    requests = []
    app = reader(tmp_path, tts_endpoint, sse_audio, requests, MpvPlayer(audio_output="null"))
    async with app.run_test() as pilot:
        # Cached-paragraph bursts and file switches must not surface as player errors.
        await until(pilot, lambda: app.ready and len(requests) == 5, 10)
        await app.seek_paragraph(8)
        await until(pilot, lambda: app.ready and app.timeline.anchor == 8, 10)
        await until(pilot, lambda: app.pending_selection is None, 10)
        await app.refresh_playback()
        assert app.position == pytest.approx(app.timeline.start(8), abs=0.3)
        await app.action_seek(-10)
        await until(pilot, lambda: app.ready and app.timeline.anchor == 7, 10)
        await until(pilot, lambda: app.pending_selection is None, 10)
        await app.refresh_playback()
        assert app.timeline.start(7) < app.position < app.timeline.start(8)
        await app.seek_paragraph(2)
        await until(pilot, lambda: app.ready and app.timeline.anchor == 0, 10)
        await until(pilot, lambda: app.pending_selection is None, 10)
        await app.refresh_playback()
        assert app.position == pytest.approx(120, abs=0.3)
        assert "Paragraph 4." not in requests[:7]
        assert not app.generation_error


async def test_back_to_back_restarts_keep_playing_once_the_target_arrives(
    tmp_path, tts_endpoint, sse_audio
):
    requests, player = [], FakePlayer()
    app = reader(tmp_path, tts_endpoint, sse_audio, requests, player)
    async with app.run_test() as pilot:
        await until(pilot, lambda: len(requests) == 4)
        assert not player.paused
        # The second restart starts while the first is still silent, waiting for its target.
        await app.seek_paragraph(8)
        await app.seek_paragraph(10)
        await until(pilot, lambda: app.ready and app.timeline.anchor == 10)
        await until(pilot, lambda: app.pending_selection is None)
        assert not player.paused and not app.paused

        # While paused, a restart stays paused; play/pause during the wait changes that.
        await app.action_toggle_pause()
        assert player.paused
        await app.seek_paragraph(5)
        await app.action_toggle_pause()
        await until(pilot, lambda: app.ready and not app.restarting)
        assert app.query_one("#article-text").current == 5
        assert not player.paused
