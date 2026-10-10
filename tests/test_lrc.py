from __future__ import annotations

from autosync.lyrics.lrc import parse_lrc_text


def test_basic_lines_and_order() -> None:
    text = """[ar:Artist]
[ti:Title]
[00:12.00]second line
[00:05.50]first line
[00:20.00]third line
"""
    lines, meta = parse_lrc_text(text)
    assert meta["ar"] == "Artist"
    assert [ln.text for ln in lines] == ["first line", "second line", "third line"]
    assert abs(lines[0].start - 5.5) < 1e-9
    assert abs(lines[1].start - 12.0) < 1e-9
    # end must NOT be fabricated from the next line start
    assert all(ln.end is None for ln in lines)


def test_multiple_timestamps_per_line() -> None:
    lines, _ = parse_lrc_text("[00:01.00][00:09.00]chorus")
    assert len(lines) == 2
    assert lines[0].text == "chorus"
    assert abs(lines[0].start - 1.0) < 1e-9
    assert abs(lines[1].start - 9.0) < 1e-9


def test_offset_metadata() -> None:
    lines, _ = parse_lrc_text("[offset:+500]\n[00:10.00]hi")
    assert abs(lines[0].start - 10.5) < 1e-9


def test_enhanced_segments() -> None:
    lines, _ = parse_lrc_text("[00:10.00]<00:10.00>Hel<00:10.50>lo")
    assert len(lines) == 1
    segs = lines[0].segments
    assert segs is not None
    assert [s.text for s in segs] == ["Hel", "lo"]
    assert abs(segs[1].start - 10.5) < 1e-9


def test_centiseconds_vs_milliseconds() -> None:
    lines, _ = parse_lrc_text("[00:01.50]a\n[00:02.500]b")
    assert abs(lines[0].start - 1.5) < 1e-9
    assert abs(lines[1].start - 2.5) < 1e-9
