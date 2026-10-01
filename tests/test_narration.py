import asyncio
import contextlib
import json

import httpx
from test_streaming import STOP, ByteStream, pcm_event

from qwen_reader.config import Settings
from qwen_reader.narration import Cue, narration_spans
from qwen_reader.synthesis import Synthesizer, wav_duration


def test_paragraph_units_preserve_offsets_repeated_text_and_unicode():
    text = (
        "  标题。\n\n重复的段落。 第二句话！\n \n重复的段落。 第二句话！\n\n"
        "English. Still the same paragraph.  "
    )
    spans = narration_spans(text)
    assert [text[s.start : s.end] for s in spans] == [
        "标题。",
        "重复的段落。 第二句话！",
        "重复的段落。 第二句话！",
        "English. Still the same paragraph.",
    ]
    assert spans[1].start != spans[2].start
    assert all(a.end < b.start for a, b in zip(spans, spans[1:], strict=False))


def test_long_paragraph_is_split_near_sentences_with_correct_offsets():
    text = "First sentence. Second sentence.\n\n第三句话" * 5
    spans = narration_spans(text, 20)
    assert all(s.end - s.start <= 20 for s in spans)
    assert "".join(text[s.start : s.end] for s in spans).replace(" ", "") == text.replace(
        "\n", ""
    ).replace(" ", "")
    assert text[spans[0].start : spans[0].end] == "First sentence."


async def test_actual_audio_timings_cache_and_corrupt_index_rebuild(
    tmp_path, tts_endpoint, sse_audio
):
    text = "短句。\n\nThis paragraph has two sentences. They are read together.\n\n短句。"
    spans = narration_spans(text)
    requests = []

    def handler(request):
        content = json.loads(request.content)["input"]["text"]
        requests.append(content)
        return sse_audio(duration=1 if content == "短句。" else 3.5)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        synth = Synthesizer(
            Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path / "cache"), client
        )
        cues = {}

        def receive(cue):
            cues[cue.start] = cue

        parts = [
            part
            async for part in synth.stream(text, lambda *args: None, tmp_path / "spool", receive)
        ]
        assert sum(p.duration for p in parts) == 5.5
        expected = [
            Cue(spans[0].start, spans[0].end, 0, 1),
            Cue(spans[1].start, spans[1].end, 1, 4.5),
            Cue(spans[2].start, spans[2].end, 4.5, 5.5),
        ]
        assert list(cues.values()) == expected
        assert len(requests) == 2  # Repeated text reuses the same audio.
        destination = synth.cache_path(text, "article")
        assert wav_duration(destination) == 5.5
        cues.clear()
        parts = [
            part
            async for part in synth.stream(text, lambda *args: None, tmp_path / "spool", receive)
        ]
        assert len(parts) == 1
        assert list(cues.values()) == expected
        # Do not play a full cached file if its seek index is missing or corrupt.
        destination.with_suffix(".json").write_text('{"version": 1, "cues": []}')
        cues.clear()
        await synth.synthesize(text, lambda *args: None)
        assert synth.cached_cues(destination, text) == expected
        assert len(requests) == 2


async def test_lookahead_is_bounded_ordered_and_cancelled(tmp_path, tts_endpoint):
    requests = []
    opened = asyncio.Event()
    closed = set()

    class Streaming(ByteStream):
        def __init__(self, content):
            super().__init__([])
            self.content = content

        async def __aiter__(self):
            # Enough audio to fill the next paragraph's bounded queue.
            for _ in range(20):
                yield pcm_event(b"\0\0" * 24000)
            yield STOP

        async def aclose(self):
            closed.add(self.content)

    def handler(request):
        content = json.loads(request.content)["input"]["text"]
        requests.append(content)
        if len(requests) == 2:
            opened.set()
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Streaming(content)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        synth = Synthesizer(
            Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path / "cache"), client
        )
        async with contextlib.aclosing(
            synth.stream("One.\n\nTwo.\n\nThree.", lambda *args: None, tmp_path / "spool")
        ) as stream:
            part = await anext(stream)
            await asyncio.wait_for(opened.wait(), 1)
            assert part.duration == 0.75
            assert requests == ["One.", "Two."]
        assert closed == {"One.", "Two."}
        assert not list((tmp_path / "cache").glob("*.partial"))
        assert not list((tmp_path / "cache").glob("article-*"))


async def test_duplicate_adjacent_paragraphs_share_one_inflight_request(
    tmp_path, tts_endpoint, sse_audio
):
    requests = []

    def handler(request):
        requests.append(request)
        return sse_audio(duration=2)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        synth = Synthesizer(
            Settings(api_key="key", endpoint=tts_endpoint, cache_dir=tmp_path), client
        )
        path = await synth.synthesize("Same.\n\nSame.", lambda *args: None)
        assert wav_duration(path) == 4
        assert len(requests) == 1
