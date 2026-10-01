import asyncio
import json
import sys
from dataclasses import asdict, replace

import httpx
import pytest
from test_app import FakePlayer, mock_services
from test_streaming import STOP, ByteStream, pcm_event
from textual.widgets import Input, Select

from qwen_reader.app import ReaderApp
from qwen_reader.article import parse_article
from qwen_reader.article_view import ArticleText
from qwen_reader.config import MODEL, Settings
from qwen_reader.model_screen import CUSTOM_VOICE
from qwen_reader.playback_state import Bookmark, ModelChoice, PlaybackState
from qwen_reader.synthesis import Synthesizer


def test_progress_roundtrip_multiple_articles_and_private_atomic_file(tmp_path):
    path = tmp_path / "playback.json"
    state = PlaybackState(path)
    one = Bookmark("https://one.test/", "audio-one", 12.75, 1.4, True)
    two = Bookmark("https://two.test/", "audio-two", 8.5)
    state.save(one)
    state.save(two)
    restored = PlaybackState(path)
    assert restored.get(one.url, one.narration) == one
    assert restored.get(two.url, two.narration) == two
    assert restored.get(one.url, "changed-model-or-text") is None
    assert restored.last_url == two.url
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "content", ["{", "null", "[]", '{"version": 100}', '{"version": 1, "bookmarks": null}']
)
def test_corrupt_or_unknown_state_is_ignored(tmp_path, content):
    path = tmp_path / "playback.json"
    path.write_text(content)
    state = PlaybackState(path)
    assert state.last_url == ""
    assert state.bookmarks == {}


@pytest.mark.parametrize(
    "changes",
    [
        {"position": -1},
        {"position": float("nan")},
        {"speed": 9},
        {"paused": "false"},
        {"url": "file:///tmp/article"},
    ],
)
def test_invalid_bookmark_is_ignored(tmp_path, changes):
    path = tmp_path / "playback.json"
    entry = asdict(Bookmark("https://example.test/", "audio", 10)) | changes
    path.write_text(json.dumps({"version": 1, "last_url": entry["url"], "bookmarks": [entry]}))
    assert PlaybackState(path).bookmarks == {}


@pytest.mark.parametrize("paused", [False, True])
async def test_restart_restores_last_url_exact_position_speed_pause_and_highlight(
    tmp_path, tts_endpoint, sse_audio, paused
):
    settings = Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path)
    player = FakePlayer()
    app = ReaderApp(
        settings, "https://example.com/story", player=player, client=mock_services(sse_audio)
    )
    async with app.run_test() as pilot:
        await app.prepare_worker.wait()
        await app.action_speed(0.5)
        player.position, player.paused = 37.25, paused
        # Quit must fetch the final mpv position, even between periodic updates.
        await app.action_quit()
        await pilot.pause()
    stored = PlaybackState(tmp_path / "playback.json")
    bookmark = stored.get(app.current_url, app.narration_id)
    assert bookmark.position == 37.25
    assert bookmark.paused == paused
    assert player.closed

    restarted_player = FakePlayer()
    restarted = ReaderApp(settings, player=restarted_player, client=mock_services(sse_audio))
    async with restarted.run_test() as pilot:
        await restarted.prepare_worker.wait()
        await pilot.pause()
        assert restarted.query_one("#url", Input).value == "https://example.com/story"
        assert restarted_player.position == 37.25
        assert restarted_player.speed == 1.5
        assert restarted_player.paused == paused
        assert restarted.query_one(ArticleText).current == 1
        assert restarted.restoring is None


async def test_periodic_save_and_switching_articles_do_not_mix_positions(
    tmp_path, tts_endpoint, sse_audio
):
    settings = Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path)
    app = ReaderApp(
        settings, "https://one.test", player=FakePlayer(), client=mock_services(sse_audio)
    )
    async with app.run_test() as pilot:
        await app.prepare_worker.wait()
        narration = app.narration_id
        app.player.position = 13
        app.last_saved_at = 0
        await app.refresh_playback()
        assert (
            PlaybackState(tmp_path / "playback.json").get("https://one.test", narration).position
            == 13
        )
        app.player.position = 17
        app.query_one("#url", Input).value = "https://two.test"
        app.begin_load()
        await app.prepare_worker.wait()
        await pilot.pause()
        assert app.position == 0
        assert app.playback_state.get("https://one.test", narration).position == 17
        app.player.position = 22
        app.query_one("#url", Input).value = "https://one.test"
        app.begin_load()
        await app.prepare_worker.wait()
        await pilot.pause()
        assert app.player.position == 17
        assert app.playback_state.get("https://two.test", narration).position == 22


async def test_restore_waits_silently_for_target_without_overwriting_bookmark(
    tmp_path, tts_endpoint
):
    url = "https://example.test"
    body = "---\ntitle: First\n---\nFirst\n\nSecond."
    gate = asyncio.Event()
    waiting = asyncio.Event()
    settings = Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path)

    class Delayed(ByteStream):
        async def __aiter__(self):
            yield pcm_event(b"\0\0" * 24000)
            waiting.set()
            await gate.wait()
            yield pcm_event(b"\0\0" * 48000)
            yield STOP

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, text=body)
        requests.append(json.loads(request.content)["input"]["text"])
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Delayed([])
        )

    requests = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    narration = (
        Synthesizer(settings, client).cache_path(parse_article(url, body).text, "article").stem
    )
    # 1.25s into the second paragraph; the first has never been generated.
    original = Bookmark(url, narration, 4.25, 1.2, False, unit=1, unit_offset=1.25)
    PlaybackState(tmp_path / "playback.json").save(original)
    app = ReaderApp(settings, player=FakePlayer(), client=client)
    async with app.run_test() as pilot:
        await asyncio.wait_for(waiting.wait(), 2)
        await pilot.pause()
        assert app.ready and app.restoring
        assert app.player.paused
        assert app.player.position == 0
        await app.checkpoint(force=True, query_player=True)
        assert PlaybackState(tmp_path / "playback.json").get(url, narration) == original
        gate.set()
        await app.prepare_worker.wait()
        await pilot.pause()
        # The playlist starts at the restored paragraph; earlier ones are not generated.
        assert requests == ["Second."]
        assert app.player.position == 1.25
        assert not app.player.paused
        assert app.restoring is None


async def test_quitting_during_restore_keeps_saved_target(tmp_path, tts_endpoint):
    settings = Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path)
    original = Bookmark("https://example.test", "audio", 100, paused=True)
    PlaybackState(tmp_path / "playback.json").save(original)
    started = asyncio.Event()

    async def handler(request):
        started.set()
        await asyncio.Event().wait()

    app = ReaderApp(
        settings,
        player=FakePlayer(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    async with app.run_test() as pilot:
        await started.wait()
        await app.action_quit()
        await pilot.pause()
    assert PlaybackState(tmp_path / "playback.json").get(original.url, "audio") == original


async def test_finished_narration_stays_finished_and_can_replay(tmp_path, tts_endpoint, sse_audio):
    settings = Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path)
    app = ReaderApp(
        settings, "https://example.test", player=FakePlayer(), client=mock_services(sse_audio)
    )
    async with app.run_test():
        await app.prepare_worker.wait()
        app.player.ended = True
        await app.action_quit()
    restarted = ReaderApp(settings, player=FakePlayer(), client=mock_services(sse_audio))
    async with restarted.run_test() as pilot:
        await restarted.prepare_worker.wait()
        await pilot.pause()
        assert restarted.ended and restarted.player.paused
        assert restarted.player.position == pytest.approx(restarted.duration, abs=0.02)
        await restarted.action_toggle_pause()
        assert restarted.player.position == 0
        assert not restarted.player.paused and not restarted.ended


async def test_restart_remembers_tui_voice_but_respects_explicit_config_changes(
    tmp_path, tts_endpoint, sse_audio
):
    settings = Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path)
    app = ReaderApp(
        settings, "https://example.test", player=FakePlayer(), client=mock_services(sse_audio)
    )
    # Simulate a session choice before Read, retaining the original startup config.
    app.settings = replace(settings, voice="Emily_v3.1")
    app.synthesizer = Synthesizer(app.settings, app.client)
    async with app.run_test():
        await app.prepare_worker.wait()
        app.player.position = 5
        await app.action_quit()
    for configured, restore_choices, expected in [
        (settings, True, "Emily_v3.1"),
        (settings, False, settings.voice_id),
        (replace(settings, voice="Luna_v3.1"), True, "Luna_v3.1"),
    ]:
        reader = ReaderApp(configured, restore_choices=restore_choices)
        assert reader.settings.voice_id == expected
        await reader.client.aclose()


async def test_tui_model_choice_persists_for_new_sessions_unless_config_changes(tmp_path):
    settings = Settings(cache_dir=tmp_path)
    app = ReaderApp(settings, player=FakePlayer(), client=httpx.AsyncClient())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.click("#choose-model")
        await pilot.pause()
        picker = app.screen
        picker.query_one("#model-choice", Select).value = "qwen-audio-3.0-tts-plus"
        await pilot.pause()
        picker.query_one("#voice-choice", Select).value = CUSTOM_VOICE
        await pilot.pause()
        picker.query_one("#voice-custom", Input).value = "my-plus-clone"
        await pilot.click("#model-apply")
        await pilot.pause()
        assert app.settings.voice == "my-plus-clone"
    chosen = ("qwen-audio-3.0-tts-plus", "my-plus-clone")
    flash = replace(settings, model="qwen-audio-3.0-tts-flash")
    for configured, url, restore_choices, expected in [
        (settings, "", True, chosen),
        (settings, "https://other.test", True, chosen),
        (settings, "", False, (MODEL, "")),  # --model / --voice on the command line
        (flash, "", True, ("qwen-audio-3.0-tts-flash", "")),  # .env edited since the choice
    ]:
        reader = ReaderApp(configured, url, restore_choices=restore_choices)
        assert (reader.settings.model, reader.settings.voice) == expected
        await reader.client.aclose()

    path = tmp_path / "playback.json"
    assert PlaybackState(path).choice == ModelChoice(*chosen, app.startup_configuration)
    data = json.loads(path.read_text())
    data["choice"] = {"model": 5, "voice": "", "configuration": ""}
    path.write_text(json.dumps(data))
    assert PlaybackState(path).choice is None
