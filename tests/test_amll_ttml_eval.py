"""Independent synthetic references; no third-party lyric fixtures."""

import pytest
from amll_ttml_eval import compare, match_lines, match_units, metrics, norm, parse_ttml, timestamp


@pytest.mark.parametrize(("value", "expected"), [
    ("01:02:03.45", 3723.45), ("02:03.4", 123.4), ("95.125", 95.125),
    ("2h", 7200), ("2m", 120), ("12.5s", 12.5), ("150ms", .15),
    ("30f", 1), ("15t", 15), ("00:00:01:15", 1.5),
])
def test_time_formats(value, expected):
    assert timestamp(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "bad", "1:70", "00:00:01:31", "-1s"])
def test_invalid_time(value):
    with pytest.raises(ValueError):
        timestamp(value)


def test_parser_roles_namespaces_beats_and_agents(tmp_path):
    path = tmp_path / "test.ttml"
    path.write_text('''<tt xmlns="http://www.w3.org/ns/ttml"
      xmlns:ttm="http://www.w3.org/ns/ttml#metadata" xmlns:amll="http://www.example.com/ns/amll">
      <head><metadata><span begin="0" end="1">not a lyric</span></metadata></head>
      <body><div><p begin="10" end="14" ttm:agent="v1">
        <span begin="10" end="11" amll:empty-beat="2">Ａ&amp;B</span>
        <span begin="11" end="12">世界</span>
        <span begin="12" end="12"> </span><amll:empty-beat/>
        <span ttm:role="x-translation"><span begin="10" end="11">excluded</span></span>
        <span ttm:role="x-bg" begin="10.5" end="13" ttm:agent="v2">
          <span begin="10.5" end="13">(back)</span>
          <span ttm:role="x-roman">excluded</span>
        </span>
      </p></div></body></tt>''')
    d = parse_ttml(path)
    assert len(d["lines"]) == 1
    line = d["lines"][0]
    assert len(line["lead"]) == 2
    assert line["lead"][0]["text"] == "Ａ&B"
    assert line["lead"][0]["start"] == 10  # no parent offset added
    assert line["lead"][0]["agent"] == "v1"
    assert len(line["background"]) == 1
    assert line["background"][0]["agent"] == "v2"
    assert d["counts"]["empty_beat_annotations"] == 1
    assert d["counts"]["empty_beat_elements"] == 1
    assert norm("Ａ & B，世界!") == "ab世界"


def units(*texts):
    return [{"text": t} for t in texts]


def test_aggregate_complete_reference_without_subdivision():
    assert match_units(units("甲乙", "丙"), units("甲", "乙", "丙")) == [(0, 1, 0, 2), (1, 2, 2, 3)]
    # Syllabic reference can only be compared at the common complete word bounds.
    assert match_units(units("hel", "lo", "world"), units("hello", "world")) == [(0, 2, 0, 1), (2, 3, 1, 2)]


def test_unit_ambiguity_and_missing_text():
    assert match_units(units("a", "a"), units("a")) == []
    assert match_units(units("甲", "乙", "丙"), units("甲", "丙")) == [(0, 1, 0, 1), (2, 3, 1, 2)]


def test_line_order_repetition_and_ambiguity():
    refs = units("start", "chorus", "bridge", "chorus", "end")
    preds = units("start", "chorus", "bridge", "chorus", "end")
    assert [k for k, _ in match_lines(refs, preds)] == [(i, i, i + 1, i + 1) for i in range(5)]
    assert match_lines(units("chorus", "chorus"), units("chorus")) == []


def test_exact_line_merge_is_allowed():
    assert match_lines(units("first second"), units("first", "second"))[0][0] == (0, 0, 1, 2)


def test_metrics_nonzero_and_both_boundary_rates():
    rows = [{"start_error": -.2, "end_error": .6}, {"start_error": .4, "end_error": -1.2}]
    result = metrics(rows)
    assert result["start"]["mae"] == pytest.approx(.3)
    assert result["start"]["signed_bias"] == pytest.approx(.1)
    assert result["end"]["mae"] == pytest.approx(.9)
    assert result["either_boundary_exceed"]["0.5"]["count"] == 2
    assert result["end"]["exceed"]["1.0"]["rate"] == .5


def test_failed_and_heuristic_rows_stay_in_denominator():
    refs = {"metadata": {}, "counts": {}, "lines": [{"index": i, "text": t, "background": [],
                       "lead": [{"text": t, "start": i, "end": i + .5,
                                 "line_index": i, "unit_index": 0}]} for i, t in enumerate(("hello", "world"))]}
    pred = {"language": "en", "model": {"name": "test"}, "lines": [], "player_timeline": []}
    for i, t in enumerate(("hello", "world")):
        source = "model" if i == 0 else "heuristic"
        pred["lines"].append({"index": i, "text": t, "kind": "singing", "source": source,
                              "units": [{"text": t, "start": i + .1, "end": i + .6, "source": source}]})
        pred["player_timeline"].append({"index": i, "mode": "word" if i == 0 else "line_lrc"})
    summary, detail = compare(refs, pred)
    assert summary["reference_units"] == 2
    assert summary["matched_reference_units"] == 1
    assert summary["production_fallback_lines"] == 1
    assert len(detail["unmatched"]) == 1
    assert summary["accepted_common_boundary_intervals"]["n"] == 1
    reserved, _ = compare(refs, pred, reserved=(0,))
    assert reserved["accepted_common_boundary_intervals"]["n"] == 0


def test_redacted_predictions_fail_closed():
    with pytest.raises(ValueError, match="redacted"):
        compare({}, {"text": "<redacted>"})
