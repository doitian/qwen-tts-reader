import pytest

from qwen_reader.timeline import CJK_RATE, LATIN_RATE, Timeline


def test_estimates_use_the_measured_rate_and_default_by_script():
    assert Timeline([10], text="English words").duration(0) == pytest.approx(10 * LATIN_RATE)
    assert Timeline([10], text="中文的句子。中文的句子。").duration(0) == pytest.approx(
        10 * CJK_RATE
    )
    timeline = Timeline([10, 20, 30], {0: 5.0})
    assert timeline.duration(1) == 10  # 0.5 seconds per character, as measured.
    assert timeline.total() == 30
    assert not timeline.exact() and timeline.exact(1)


def test_anchor_keeps_times_continuous_while_estimates_change():
    timeline = Timeline([10, 10, 10, 10], {0: 5.0})
    timeline.set_anchor(2)
    assert timeline.anchor_time == 10
    # A different measured rate changes estimates, but not the playlist start.
    timeline.known[3] = 10.0  # Now 0.75 seconds per character.
    assert timeline.anchor_time == 10
    assert timeline.start(3) == 10 + timeline.duration(2)
    # Earlier units are measured backwards from the anchor.
    assert timeline.start(1) == 10 - timeline.duration(1)
    unit, offset = timeline.locate(9)
    assert unit == 1 and offset == pytest.approx(9 - timeline.start(1))
    timeline.set_anchor(1)
    assert timeline.anchor_time == pytest.approx(10 - timeline.duration(1))


def test_locate_clamps_to_the_ends():
    timeline = Timeline([4, 4], {0: 2.0, 1: 2.0})
    assert timeline.locate(-1) == (0, 0.0)
    assert timeline.locate(3) == (1, 1.0)
    assert timeline.locate(10) == (1, 8.0)
