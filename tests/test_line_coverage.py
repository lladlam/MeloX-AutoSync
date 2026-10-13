"""Coverage accounting: no singable line may disappear silently (V1.3)."""

from __future__ import annotations

import line_coverage as lc


def _song(tmp_path, name, lines):
    (tmp_path / f"{name}.lrc").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / f"{name}.mp3").write_bytes(b"x")


def test_every_singing_line_has_an_outcome(tmp_path):
    _song(tmp_path, "mix", [
        "[00:00.000]作词 : 某人",                # metadata
        "[00:05.000]船 摇啊摇",                  # zh singing
        "[00:09.000](Bora sana, Bora sana)",    # pure Latin -> en (attempted)
        "[00:13.000](間奏)",                     # instrumental cue
        "[00:17.000]别害怕 we clap clap",        # latin-dominant mixed -> not routable
    ])
    res = lc.coverage("mix", "zh", tmp_path)
    assert res["unaccounted"] == 0
    assert res["denominator"] == res["accounted_for"]
    # the pure-Latin line must be routed to English, not dropped
    assert res["singing_routed"].get("en", 0) >= 1
    # the mixed line must be recorded as not routable, not silently removed
    assert res["singing_not_routable"] >= 1
    assert res["by_kind"].get("metadata", 0) >= 1
    assert res["by_kind"].get("instrumental", 0) >= 1


def test_non_singing_lines_are_excluded_from_the_singing_denominator(tmp_path):
    _song(tmp_path, "plain", [
        "[00:00.000]作词 : 某人",
        "[00:05.000]红透晚烟青",
        "[00:09.000]船 摇啊摇",
    ])
    res = lc.coverage("plain", "zh", tmp_path)
    assert res["singing_total"] == 2          # only the two lyric lines
    assert res["by_kind"]["metadata"] == 1
    assert res["unaccounted"] == 0
