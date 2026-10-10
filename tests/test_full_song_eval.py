"""Tests for the full-song harness and the human-annotation template tool."""

from __future__ import annotations

import pytest
from full_song_eval import discover, validate
from make_annotation_template import build_rows, score_csv, write_template


# ------------------------------------------------------------------- harness
def test_discover_pairs_by_exact_prefix(tmp_path) -> None:
    (tmp_path / "SongA.mp3").write_bytes(b"x")
    (tmp_path / "SongA-Artist.lrc").write_text("[00:01.00]hi\n", encoding="utf-8")
    (tmp_path / "SongB.wav").write_bytes(b"x")
    (tmp_path / "SongB.lrc").write_text("[00:01.00]hi\n", encoding="utf-8")
    songs = discover(tmp_path)
    assert {s["song"]: s["lrc"].name for s in songs} == {
        "SongA": "SongA-Artist.lrc", "SongB": "SongB.lrc"
    }


def test_discover_rejects_ambiguity(tmp_path) -> None:
    (tmp_path / "SongA.mp3").write_bytes(b"x")
    (tmp_path / "SongA-One.lrc").write_text("[00:01.00]a\n", encoding="utf-8")
    (tmp_path / "SongA-Two.lrc").write_text("[00:01.00]b\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        discover(tmp_path)


def _result() -> dict:
    return {
        "language": "en",
        "lines": [
            {"index": 0, "text": "one two", "kind": "singing", "source": "model", "lrc_start": 1.0,
             "units": [{"text": "one", "start": 0.9, "end": 1.2}, {"text": "two", "start": 1.2, "end": 1.6}]},
            {"index": 1, "text": "three", "kind": "singing", "source": "failed", "lrc_start": 5.0, "units": []},
        ],
    }


def test_validate_detects_out_of_range_and_crossing() -> None:
    data = _result()
    data["lines"][0]["units"].append({"text": "bad", "start": 1.6, "end": 99.0})
    stats = validate(data, dur=10.0)
    assert stats["out_of_range_units"] >= 1
    assert stats["lines"] == 2 and stats["failed_lines"] == 1


# ---------------------------------------------------------------- annotation
def test_template_selects_representative_rows() -> None:
    rows = build_rows(_result(), "song", max_units=5, seed=1)
    assert rows and all(r["ref_start_s"] == "" for r in rows)  # refs left empty
    assert any(r["notes"] == "edge line" for r in rows)


def test_annotation_roundtrip_and_scoring(tmp_path) -> None:
    data = _result()
    out = write_template({"song": data}, tmp_path / "t.csv", max_units=10, seed=1)
    filled = []
    for i, line in enumerate(out.read_text(encoding="utf-8").splitlines()):
        filled.append(line)
    # fill the first data row with its own prediction -> zero error
    lines = out.read_text(encoding="utf-8").splitlines()
    header, first = lines[0], lines[1].split(",")
    cols = header.split(",")
    first[cols.index("ref_start_s")] = first[cols.index("pred_start_s")]
    first[cols.index("ref_end_s")] = first[cols.index("pred_end_s")]
    out.write_text("\n".join([header, ",".join(first)] + lines[2:]) + "\n", encoding="utf-8")

    report = score_csv(out)
    assert report["unannotated_rows"] >= 1
    st = report["by_language"]["en"]
    assert st["n"] == 1 and st["start_mae"] == 0.0


def test_blank_template_yields_no_accuracy() -> None:
    report = score_csv(write_template({"song": _result()}, __import__("pathlib").Path("/tmp/_t.csv"), 5, 1))
    assert report["unannotated_rows"] >= 1
    assert report["by_language"] == {}
