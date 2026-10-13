"""Guards for the Android contract vectors and the release-manifest provenance."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "android_contract" / "contract.json"
RELEASES = REPO / "release" / "android-models"

_HEX = re.compile(r"^[0-9a-f]{64}$")


def test_contract_vectors_are_our_own_and_shapes_match():
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert data["schema"] == "melox.autosync.contract.1"
    seen = {}
    for v in data["vectors"]:
        seen[v["language"]] = v
        wav = CONTRACT.parent / v["language"] / v["waveform"]
        assert wav.exists(), wav
        assert hashlib.sha256(wav.read_bytes()).hexdigest() == v["waveform_sha256"]
        assert _HEX.match(v["model_sha256"])
        # 3 s at 16 kHz, 20 ms hop -> T = floor((N-400)/320)+1 = 149
        T = (16000 * 3 - 400) // 320 + 1
        if v["language"] == "en":
            assert v["outputs"][0]["probe"]["shape"] == [1, T, 32]
        if v["language"] == "ja":
            assert v["outputs"][0]["probe"]["shape"] == [1, T, 48]
        if v["language"] == "zh":
            assert v["outputs"][0]["name"] == "us_alphas"
    assert set(seen) == {"en", "ja", "zh"}


def test_every_release_language_declares_accuracy_provenance():
    for pack in sorted(RELEASES.glob("android-models-v*")):
        idx = pack / "android-models.json"
        if not idx.exists():
            continue
        data = json.loads(idx.read_text(encoding="utf-8"))
        # provenance was introduced in beta.4; earlier historical packs are kept
        # as-is on purpose (we do not rewrite published release metadata).
        if data.get("pack_version", "0") < "0.1.0-beta.4":
            continue
        for lang, spec in data["languages"].items():
            acc = spec.get("accuracy", {})
            assert "provenance" in acc, f"{pack.name}/{lang} missing accuracy.provenance"
            assert acc["provenance"], f"{pack.name}/{lang} empty provenance"


def test_beta4_ja_manifest_describes_the_shipped_variant():
    pack = RELEASES / "android-models-v0.1.0-beta.4"
    if not (pack / "android-models.json").exists():
        return  # pack not present in this checkout
    data = json.loads((pack / "android-models.json").read_text(encoding="utf-8"))
    ja = data["languages"]["ja"]["accuracy"]
    recipe = ja.get("model_recipe", "")
    assert "layer 8" in recipe, recipe
    # the V1.2 stale cross-reference to the beta.2 report must be gone
    assert "int8_vs_fp32_char_start_mae_ms" not in ja
    # and the numbers must carry their corpus
    assert "real_music_int8_vs_fp32_start_mae_ms" in ja
    assert "gtsinger_int8_vs_fp32_start_mae_ms" in ja


def test_model_assets_have_valid_hashes_and_sizes():
    pack = RELEASES / "android-models-v0.1.0-beta.4"
    if not (pack / "android-models.json").exists():
        return
    data = json.loads((pack / "android-models.json").read_text(encoding="utf-8"))
    for spec in data["languages"].values():
        for f in spec["files"]:
            assert _HEX.match(f["sha256"]), f
            assert f["bytes"] > 0
            assert f["download_url"].endswith("/" + f["name"])
    for f in data.get("extra_assets", []):
        assert _HEX.match(f["sha256"])
