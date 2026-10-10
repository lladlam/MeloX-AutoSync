"""CLI end-to-end control-flow tests (V0.8 P2).

These use a **test double** for the model (monkeypatched into the CLI): they
verify LRC -> engine selection -> inference -> quality gate -> JSON export ->
player_timeline wiring. Real-model inference is exercised separately (see
docs/CLI_END_TO_END.md); these tests do **not** claim real inference.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from autosync import cli
from autosync.models.base import ModelMeta
from autosync.types import AlignUnit


class FakeAligner:
    def __init__(self, quality: str | None = "reliable") -> None:
        self.meta = ModelMeta(
            name="fake/model", version="0", language="ja", license="test-only", status="experimental"
        )
        self.frame_seconds = 0.02
        self.infer_seconds = 0.0
        self.last_quality = None
        self._q = quality
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def align_text(self, audio, sample_rate, text):
        self.last_quality = {"state": self._q, "reasons": ["fake"]} if self._q else None
        return [AlignUnit(text=text, start=0.1, end=0.4, source="model")]


@pytest.fixture()
def media(tmp_path):
    wav = tmp_path / "song.wav"
    sf.write(str(wav), np.zeros(16000 * 8, dtype=np.float32), 16000)
    lrc = tmp_path / "song.lrc"
    lrc.write_text("[00:01.00]first line\n[00:04.00]second line\n", encoding="utf-8")
    return wav, lrc, tmp_path


def test_cli_model_control_flow(monkeypatch, media) -> None:
    wav, lrc, tmp = media
    fake = FakeAligner("reliable")
    monkeypatch.setattr(cli, "get_aligner", lambda *a, **k: fake)
    out = tmp / "out.json"
    rc = cli.main(["align", "--audio", str(wav), "--lyrics", str(lrc), "--language", "ja",
                   "--output", str(out), "-q"])
    assert rc == 0 and fake.loaded
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["model"]["name"] == "fake/model"
    assert data["model"]["status"] == "experimental"
    assert data["schema_version"] == "0.3"
    assert data["quality_summary"]["reliable"] == 2
    assert data["player_timeline"][0]["mode"] == "word"
    # original lyrics preserved end-to-end
    assert [e["text"] for e in data["player_timeline"]] == ["first line", "second line"]
    # original LRC times preserved independently
    assert data["player_timeline"][1]["lrc_start"] == 4.0
    assert data["player_timeline"][1]["lrc_end"] is None


def test_cli_uncertain_falls_back(monkeypatch, media) -> None:
    wav, lrc, tmp = media
    monkeypatch.setattr(cli, "get_aligner", lambda *a, **k: FakeAligner("uncertain"))
    out = tmp / "out.json"
    cli.main(["align", "--audio", str(wav), "--lyrics", str(lrc), "--language", "ja",
              "--output", str(out), "-q"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert all(e["mode"] == "line_lrc" for e in data["player_timeline"])
    assert data["quality_summary"]["uncertain"] == 2


def test_cli_heuristic_engine_marks_not_assessed(media) -> None:
    wav, lrc, tmp = media
    out = tmp / "out.json"
    cli.main(["align", "--audio", str(wav), "--lyrics", str(lrc), "--language", "en",
              "--engine", "heuristic", "--output", str(out), "-q"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["engine"] == "heuristic"
    assert data["quality_summary"]["not_assessed"] == 2
    # en is not gated -> heuristic timings stay in word mode
    assert all(e["mode"] == "word" for e in data["player_timeline"])


def test_cli_info_runs() -> None:
    assert cli.main(["info"]) == 0
