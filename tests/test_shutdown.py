import httpx
import pytest
from test_app import FakePlayer, mock_services
from textual import events
from textual.command import CommandInput

from qwen_reader.app import ReaderApp
from qwen_reader.config import Settings
from qwen_reader.model_screen import ModelScreen
from qwen_reader.playback_state import PlaybackState
from qwen_reader.screens import ReaderCommandPalette


async def test_palette_repeated_background_click_does_not_pop_root(tmp_path):
    app = ReaderApp(Settings(cache_dir=tmp_path), player=FakePlayer(), client=httpx.AsyncClient())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()
        palette = app.screen
        assert isinstance(palette, ReaderCommandPalette)
        click = events.Click(
            palette, 99, 29, 0, 0, 3, False, False, False, screen_x=99, screen_y=29
        )
        # Matches the user's traceback: two queued background clicks before unmount.
        palette._on_click(click)
        palette._on_click(click)
        assert len(app.screen_stack) == 1
        await pilot.pause()
        await pilot.click("#choose-model")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, ModelScreen)
        picker.action_cancel()
        picker.action_cancel()
        assert len(app.screen_stack) == 1


@pytest.mark.parametrize("surface", ["reader", "model", "palette", "palette-command"])
async def test_quit_from_every_surface_saves_progress_and_closes_player(
    tmp_path, sse_audio, tts_endpoint, surface
):
    settings = Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path)
    app = ReaderApp(
        settings, "https://example.test", player=FakePlayer(), client=mock_services(sse_audio)
    )
    async with app.run_test() as pilot:
        await app.prepare_worker.wait()
        app.player.position = 19.25
        if surface == "model":
            await pilot.click("#choose-model")
        elif surface.startswith("palette"):
            await pilot.press("ctrl+p")
        await pilot.pause()
        if surface == "palette-command":
            app.screen.query_one(CommandInput).value = "quit"
            await pilot.pause(0.6)
            await pilot.press("enter")
        else:
            await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.closing
    assert app.player.closed
    assert app.client.is_closed
    bookmark = PlaybackState(tmp_path / "playback.json").get(app.current_url, app.narration_id)
    assert bookmark.position == 19.25
