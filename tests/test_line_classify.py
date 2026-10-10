"""Tests for non-singing LRC line classification (V0.9)."""

from __future__ import annotations

import numpy as np

from autosync.alignment.segment import SegmentOptions, align_lines
from autosync.lyrics.classify import LineKind, classify_line, classify_lines
from autosync.lyrics.lrc import parse_lrc_text
from autosync.types import AlignUnit

SR = 16_000


# --------------------------------------------------------------- classification
def test_plain_lyrics_are_singing() -> None:
    for text in ["君の名前を呼んでいる", "I will always love you", "我爱你中国", "夜空中最亮的星"]:
        assert classify_line(text).kind is LineKind.SINGING


def test_credit_lines_are_metadata() -> None:
    cases = [
        "作词：张三",
        "作詞：米津玄師",
        "作曲 : 李四",
        "编曲：王五",
        "作詞・作曲：Aimer",
        "制作人：某某",
        "混音：某某 / 母带：某某",
        "Lyrics by John Doe",
        "Music & Lyrics by A. Writer",
        "Produced by Someone",
        "Composer: J. S. Bach",
        "OP：某某音乐",
        "詞：林夕",
        "曲：陳輝陽",
        # real-world variants (from actual commercial LRC files)
        " 吉他演奏：Ivan Korop/Daniele Fabio",
        " 合声演唱：Aylin Abbasova/核野",
        " 人声录音师 : Simon Gooding",
        " 上海录音 : ChiliChill@Staff Only Studio",
        " 笛子 : Akshat Mehrotra",
        " 管弦乐编写 : 雷声/王铭基",
        " 和声编写 : 雷声",
        " 母带 : Joe LaPorta",
        "Recording Engineer: Jane Doe",
        "Chorus: The Choir",
    ]
    for text in cases:
        assert classify_line(text).kind is LineKind.METADATA, text


def test_instrumental_cues() -> None:
    for text in ["(間奏)", "【间奏】", "[Instrumental]", "Intro", "Outro", "前奏", "尾奏", "♪♪♪ ", "..."]:
        assert classify_line(text).kind is LineKind.INSTRUMENTAL, text
    assert classify_line("").kind is LineKind.INSTRUMENTAL


def test_lyric_containing_credit_keyword_is_not_metadata() -> None:
    # a real lyric line that merely contains/derives from a credit keyword
    for text in [
        "作詞は君のためじゃない",      # starts with 作詞 but is a sentence
        "我把我们的故事写成了歌词",
        "You wrote the lyrics on my heart",
        "作曲をしたのは僕だ",
    ]:
        assert classify_line(text).kind is LineKind.SINGING, text


def test_ambiguous_credit_list_is_unknown() -> None:
    assert classify_line("作詞作曲米津玄師").kind is LineKind.UNKNOWN


def test_classify_lines_preserves_order() -> None:
    lines, _ = parse_lrc_text("[00:01.00]作词：A\n[00:05.00]真正的歌词\n")
    kinds = [c.kind for c in classify_lines(lines)]
    assert kinds == [LineKind.METADATA, LineKind.SINGING]


# ------------------------------------------------------------------- pipeline
class FakeAligner:
    frame_seconds = 0.02

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.windows: list[tuple[float, float]] = []
        self.window_lens: list[float] = []
        self.last_quality = None

    def align_text(self, audio, sample_rate, text):
        self.calls.append(text)
        self.window_lens.append(len(audio) / sample_rate)
        return [AlignUnit(text=text, start=0.0, end=0.2, source="model")]


def _run(lrc: str, seconds: float = 30.0):
    lines, _ = parse_lrc_text(lrc)
    audio = np.zeros(int(SR * seconds), dtype=np.float32)
    fake = FakeAligner()
    aligned, _ = align_lines(audio, SR, lines, fake, SegmentOptions(padding=0.3), progress=False)
    return lines, aligned, fake


def test_metadata_is_not_aligned_and_keeps_original_text_and_time() -> None:
    lrc = "[00:01.00]作词：张三\n[00:05.00]真正的歌词\n[00:20.00]Produced by Someone\n"
    _lines, aligned, fake = _run(lrc)
    assert fake.calls == ["真正的歌词"]  # only the singing line hit the model
    meta = aligned[0]
    assert meta.source == "skipped" and meta.kind == "metadata"
    assert meta.text == "作词：张三" and meta.lrc_start == 1.0
    assert meta.units == [] and meta.skip_reason
    # tail credit skipped too, text preserved
    assert aligned[2].source == "skipped" and aligned[2].text == "Produced by Someone"
    assert aligned[2].lrc_start == 20.0


def test_skipped_metadata_does_not_truncate_window() -> None:
    # the singing line's window must extend to the NEXT SINGING line (20s),
    # not be cut short by the metadata line at 5s.
    lrc = "[00:02.00]song one\n[00:05.00]作词：A\n[00:20.00]song two\n"
    _, aligned, fake = _run(lrc)
    assert aligned[0].source == "model"
    # the audio window passed to the model runs to ~20s, well past the metadata
    assert fake.window_lens[0] > 15.0


def test_window_truncated_without_metadata_rule() -> None:
    # control: when the middle line IS singing, the first window ends near it
    lrc = "[00:02.00]song one\n[00:05.00]song mid\n[00:20.00]song two\n"
    _, _, fake = _run(lrc)
    assert fake.window_lens[0] < 6.0


def test_intro_and_tail_credits_skipped() -> None:
    lrc = (
        "[00:00.00]作词：A\n[00:01.00]作曲：B\n[00:02.00]混音：C\n"
        "[00:03.00]第一句歌词\n[00:10.00]第二句歌词\n"
        "[00:16.00]制作人：D\n[00:17.00]发行：E\n"
    )
    _, aligned, fake = _run(lrc)
    assert fake.calls == ["第一句歌词", "第二句歌词"]
    kinds = [ln.kind for ln in aligned]
    assert kinds[:3] == ["metadata"] * 3
    assert kinds[3:5] == ["singing", "singing"]
    assert kinds[5:] == ["metadata"] * 2
    # nothing was deleted: every original line is present with its time
    assert [ln.lrc_start for ln in aligned] == [0.0, 1.0, 2.0, 3.0, 10.0, 16.0, 17.0]


def test_all_metadata_lrc_produces_no_units() -> None:
    lrc = "[00:01.00]作词：A\n[00:02.00]作曲：B\n"
    _, aligned, fake = _run(lrc)
    assert fake.calls == []
    assert all(ln.units == [] and ln.source == "skipped" for ln in aligned)


def test_heuristic_path_also_skips_metadata() -> None:
    from autosync.cli import _heuristic_lines

    lines, _ = parse_lrc_text("[00:01.00]作词：A\n[00:05.00]真正的歌词\n")
    audio = np.zeros(SR * 20, dtype=np.float32)
    out = _heuristic_lines(audio, SR, lines, "zh")
    assert out[0].source == "skipped" and out[0].kind == "metadata"
    assert out[1].source == "heuristic" and out[1].units
