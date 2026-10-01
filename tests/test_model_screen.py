import httpx
import pytest
from test_app import FakePlayer
from textual.widgets import Input, Select

from qwen_reader.app import ReaderApp
from qwen_reader.config import MODEL_VOICES, Settings
from qwen_reader.model_screen import CUSTOM_VOICE, ModelScreen
from qwen_reader.voices import VOICES


@pytest.mark.parametrize("model", MODEL_VOICES)
async def test_voice_dropdown_is_filtered_and_applies_selection(tmp_path, model):
    app = ReaderApp(
        Settings(model=model, cache_dir=tmp_path), player=FakePlayer(), client=httpx.AsyncClient()
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.click("#choose-model")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, ModelScreen)
        voice = picker.query_one("#voice-choice", Select)
        assert voice.value == ""
        assert not picker.query_one("#voice-custom").display
        await pilot.click("#voice-choice")
        await pilot.press("down", "enter")
        assert voice.value == VOICES[model][0][1]
        await pilot.click("#model-apply")
        await pilot.pause()
        assert app.settings.voice == VOICES[model][0][1]
        await pilot.click("#choose-model")
        await pilot.pause()
        assert app.screen.query_one("#voice-choice", Select).value == app.settings.voice


async def test_custom_voice_validation_and_model_switch_reset(tmp_path):
    app = ReaderApp(
        Settings(voice="my-cloned-voice", cache_dir=tmp_path),
        player=FakePlayer(),
        client=httpx.AsyncClient(),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.click("#choose-model")
        await pilot.pause()
        picker = app.screen
        assert picker.query_one("#voice-choice", Select).value == CUSTOM_VOICE
        custom = picker.query_one("#voice-custom", Input)
        assert custom.display and custom.value == "my-cloned-voice"
        custom.value = " "
        await pilot.click("#model-apply")
        await pilot.pause()
        assert app.screen is picker
        assert picker.query_one("#voice-error").display
        assert picker.query_one("#model-apply").region.bottom <= 24
        picker.query_one("#model-choice", Select).value = "qwen-audio-3.0-tts-plus"
        await pilot.pause()
        assert picker.query_one("#voice-choice", Select).value == ""
        assert not custom.display
        assert "Emily_v3.1" not in {
            value for _, value in picker.voice_options("qwen-audio-3.0-tts-plus")
        }
        picker.query_one("#voice-choice", Select).value = CUSTOM_VOICE
        await pilot.pause()
        custom.value = "  my-plus-clone  "
        await pilot.click("#model-apply")
        await pilot.pause()
        assert app.settings.voice == "my-plus-clone"
        assert app.settings.model == "qwen-audio-3.0-tts-plus"
