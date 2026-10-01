import asyncio
import json

import httpx
from test_app import FakePlayer, until

from qwen_reader.app import ReaderApp, save_copy
from qwen_reader.config import Settings
from qwen_reader.synthesis import wav_duration

BODY = "---\ntitle: 'Opening: a / story?'\n---\n" + "\n\n".join(
    f"Paragraph {i}." for i in range(10)
)


async def test_ctrl_s_saves_the_whole_narration_once_per_paragraph(
    tmp_path, tts_endpoint, sse_audio
):
    requests = []

    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, text=BODY)
        requests.append(json.loads(request.content)["input"]["text"])
        await asyncio.sleep(0.05)  # Slow enough to press Ctrl+S again while saving.
        return sse_audio(duration=60)

    saved_dir = tmp_path / "saved"
    app = ReaderApp(
        Settings(
            api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path / "cache", save_dir=saved_dir
        ),
        "https://example.test",
        player=FakePlayer(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    notes = []
    async with app.run_test() as pilot:
        app.notify = lambda message, **options: notes.append(message)
        await until(pilot, lambda: app.ready and app.seek is None)
        # Playback buffers only three minutes ahead; saving generates everything.
        await pilot.press("ctrl+s", "ctrl+s")
        assert "Already saving this article's audio." in notes
        await until(pilot, lambda: list(saved_dir.glob("*.wav")), 10)
        saved = saved_dir / "Opening a story.wav"
        assert wav_duration(saved) == 11 * 60
        assert f"Saved {saved}" in notes
        # Playback and saving share paragraphs instead of requesting them twice.
        assert len(requests) == len(set(requests)) == 11
        assert app.timeline.exact()

        await pilot.press("ctrl+s")
        await until(pilot, lambda: (saved_dir / "Opening a story (2).wav").exists())
        assert len(requests) == 11


def test_saved_names_are_safe_on_every_platform(tmp_path):
    source = tmp_path / "narration.wav"
    source.write_bytes(b"RIFF")
    names = [save_copy(source, tmp_path / "out", title).name for title in ["NUL", "  ", "a\tb"]]
    assert names == ["NUL audio.wav", "article.wav", "a b.wav"]
