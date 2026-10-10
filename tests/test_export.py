from __future__ import annotations

import json

from autosync.export.json_export import result_to_dict, write_json
from autosync.types import AlignedLine, AlignmentResult, AlignUnit, ModelInfo


def _sample_result() -> AlignmentResult:
    return AlignmentResult(
        language="en",
        engine="model",
        lines=[
            AlignedLine(
                index=0,
                text="hello world",
                start=1.0,
                end=2.0,
                units=[
                    AlignUnit(text="hello", start=1.0, end=1.5, source="model", score=-0.3),
                    AlignUnit(text="world", start=1.5, end=2.0, source="model", score=-0.4),
                ],
                source="model",
            )
        ],
        model=ModelInfo(name="test/model", version="abc", language="en", license="apache-2.0"),
    )


def test_result_to_dict_has_required_fields() -> None:
    d = result_to_dict(_sample_result())
    assert d["language"] == "en"
    line = d["lines"][0]
    assert line["text"] == "hello world"
    assert line["start"] == 1.0 and line["end"] == 2.0
    unit = line["units"][0]
    assert unit["text"] == "hello"
    assert unit["source"] == "model"
    assert d["model"]["name"] == "test/model"
    assert d["schema_version"]
    assert d["generator"] == "MeloX-AutoSync"


def test_write_json_roundtrip(tmp_path) -> None:
    p = write_json(_sample_result(), tmp_path / "out.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["lines"][0]["units"][1]["text"] == "world"
