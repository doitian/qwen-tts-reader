import asyncio
import base64
import contextlib
import json

import httpx
import pytest
from test_app import FakePlayer
from textual.widgets import Static

from qwen_reader.app import ReaderApp
from qwen_reader.config import Settings
from qwen_reader.synthesis import Synthesizer, sse_data, wav_duration


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def pcm_event(pcm):
    data = {"output": {"audio": {"data": base64.b64encode(pcm).decode()}}}
    return f"data: {json.dumps(data)}\n\n".encode()


STOP = b'data: {"output": {"finish_reason": "stop"}}\n\n'


async def test_sse_network_fragments_comments_multiline_and_unterminated_record():
    body = b': heartbeat\r\nevent: result\r\ndata: {"output":\r\ndata: {}}\r\n\r\ndata: [DONE]'
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, stream=ByteStream([body[i : i + 7] for i in range(0, len(body), 7)])
                )
            )
        ) as client,
        client.stream("GET", "https://example.test") as response,
    ):
        assert [event async for event in sse_data(response)] == ['{"output":\n{}}', "[DONE]"]


async def test_stream_yields_before_completion_and_does_not_cache_cancelled_request(
    tmp_path, tts_endpoint
):
    gate = asyncio.Event()

    class GatedStream(ByteStream):
        async def __aiter__(self):
            yield pcm_event(b"\0\0" * 24000)
            await gate.wait()
            yield STOP

    response_stream = GatedStream([])
    settings = Settings(api_key="test", endpoint=tts_endpoint, cache_dir=tmp_path / "cache")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=response_stream
            )
        )
    ) as client:
        synth = Synthesizer(settings, client)
        stream = synth.stream("Hello", lambda *args: None, tmp_path / "spool")
        async with contextlib.aclosing(stream):
            part = await asyncio.wait_for(anext(stream), 1)
            assert part.duration == 0.75
            assert wav_duration(part.path) == 0.75
            assert not gate.is_set()
        assert response_stream.closed
        assert list(settings.cache_dir.iterdir()) == []


@pytest.mark.parametrize(
    "ending,expected",
    [
        (b"", "before completion"),
        (b'data: {"code":"Throttling", "message":"Please retry"}\n\n', "Throttling"),
        (b'data: {"output":{"audio":{"data":"not-base64!"}}}\n\n', "malformed"),
    ],
)
async def test_stream_failures_do_not_publish_partial_cache(
    tmp_path, tts_endpoint, ending, expected
):
    settings = Settings(api_key="test", endpoint=tts_endpoint, cache_dir=tmp_path / "cache")
    body = pcm_event(b"\0\0" * 24000) + ending
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body
            )
        )
    ) as client:
        with pytest.raises(ValueError, match=expected):
            await Synthesizer(settings, client).synthesize("Hello", lambda *args: None)
    assert list(settings.cache_dir.iterdir()) == []


async def test_odd_network_fragments_preserve_every_pcm_byte(tmp_path, tts_endpoint):
    pcm = bytes(range(256)) * 100
    body = b"".join(pcm_event(pcm[i : i + 101]) for i in range(0, len(pcm), 101)) + STOP
    settings = Settings(api_key="test", endpoint=tts_endpoint, cache_dir=tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body
            )
        )
    ) as client:
        path = await Synthesizer(settings, client).synthesize("Hello", lambda *args: None)
    import wave

    with wave.open(str(path), "rb") as audio:
        assert audio.readframes(audio.getnframes()) == pcm


@pytest.mark.parametrize("failed", [False, True])
async def test_ui_plays_while_streaming_preserves_pause_and_reports_stream_failure(
    tmp_path, tts_endpoint, failed
):
    gate = asyncio.Event()
    waiting = asyncio.Event()

    class GatedStream(ByteStream):
        async def __aiter__(self):
            yield pcm_event(b"\0\0" * 24000)
            waiting.set()
            await gate.wait()
            if not failed:
                yield pcm_event(b"\0\0" * 48000)
                yield STOP

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, text="---\ntitle: Hello\n---\nA story.")
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=GatedStream([])
        )

    player = FakePlayer()
    settings = Settings(api_key="test", endpoint=tts_endpoint, cache_dir=tmp_path)
    app = ReaderApp(
        settings,
        "https://example.test",
        player=player,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    async with app.run_test() as pilot:
        await asyncio.wait_for(waiting.wait(), 2)
        assert app.ready and app.preparing
        assert app.duration == 0.75
        player.ended = True  # Playback reached the live edge.
        await app.refresh_playback()
        assert "Buffering" in str(app.query_one("#status", Static).render())
        await pilot.press("space")
        assert player.paused
        gate.set()
        await app.prepare_worker.wait()
        await app.refresh_playback()
        assert player.paused
        assert not app.preparing
        if failed:
            assert "before completion" in str(app.query_one("#status", Static).render())
            assert not list(tmp_path.glob("article-*.wav"))
        else:
            assert app.duration == 6  # Title and body are separately timed paragraphs.
            assert player.appended
            assert app.ready
