#!/usr/bin/env python
"""Build (and optionally publish) the Android model pack for MeloX-Android.

The pack contains INT8 ONNX models plus the tokenizer / preprocessing files the
Android backend needs.  Weights are NEVER committed to git: this script verifies
them against the build manifests, writes the small text artifacts into
``release/android-models/<tag>/`` and can upload everything as GitHub Release
assets.

Usage::

    .venv/bin/python tools/build_android_release.py --dry-run   # verify + print
    .venv/bin/python tools/build_android_release.py             # write pack files
    .venv/bin/python tools/build_android_release.py --upload    # create release

Accuracy numbers below are copied verbatim from the validation reports; this
script never recomputes or adjusts them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SLUG = "lladlam/MeloX-AutoSync"
TAG = "android-models-v0.1.0-beta.1"
PACK_VERSION = "0.1.0-beta.1"
MIN_BACKEND_PROTOCOL = 1
RELEASE_DOWNLOAD = f"https://github.com/{REPO_SLUG}/releases/download/{TAG}"
PACK_DIR = REPO_ROOT / "release" / "android-models" / TAG

LANGUAGES: dict[str, dict] = {
    "en": {
        "engine": "ctc-wav2vec2",
        "engine_notes": (
            "Android implements CTC Viterbi itself (blank=0), matching "
            "src/autosync/alignment/ctc.py::forced_align. No CTC code is shipped."
        ),
        "source_model": "facebook/wav2vec2-base-960h",
        "source_revision": "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
        "license": "Apache-2.0",
        "model": ("android_export/en/wav2vec2-base-960h-int8.onnx", "en-wav2vec2-base-960h-int8.onnx"),
        "extra": [
            (("models/hf/models--facebook--wav2vec2-base-960h/snapshots/"
              "22aad52d435eb6dbaf354bdad9b0da84ce7d6156/vocab.json"), "en-vocab.json", "tokenizer"),
        ],
        "accuracy": {
            "status": "NOT_ACCEPTED_FOR_WORD_LEVEL_ACCURACY",
            "gate_result": "4 of 17 checks failed (see android_export/en/REPORT.md)",
            "int8_vs_fp32_word_start_mae_ms": 121.19,
            "int8_vs_fp32_words": 168,
            "int8_vs_fp32_lines": 29,
            "fp32_vs_torch_argmax_agreement": 1.0,
            "int8_cosine_mean": 0.999108,
            "int8_cosine_min": 0.963666,
            "notes": (
                "MAE is INT8-vs-FP32 consistency, NOT accuracy against a human "
                "reference. INT8 shifts CTC boundaries by about 0.12 s on real "
                "songs; see android_export/en/QUANTIZATION_NOTES.md."
            ),
        },
    },
    "zh": {
        "engine": "monotonic-aligner-cif",
        "engine_notes": (
            "Non-CTC: the model takes audio + text and emits per-token us_alphas / "
            "us_peaks. Android implements ts_prediction_lfr6_standard (TIME_RATE "
            "0.02 s, force_time_shift -1.5, START_END_THRESHOLD 5, "
            "MAX_TOKEN_DURATION 12); reference: android_export/zh/postprocess_reference.py."
        ),
        "source_model": "iic/speech_timestamp_prediction-v1-16k-offline",
        "source_revision": "master (ModelScope; license verified 2026-10-11)",
        "license": "Apache-2.0",
        "model": ("android_export/zh/fa-zh-int8.onnx", "zh-fa-zh-int8.onnx"),
        "extra": [
            ("models/funasr/fa-zh/tokens.json", "zh-tokens.json", "tokenizer"),
            ("models/funasr/fa-zh/seg_dict", "zh-seg_dict", "tokenizer"),
            ("models/funasr/fa-zh/am.mvn", "zh-am.mvn", "audit-only"),
        ],
        "accuracy": {
            "status": "NOT_ACCEPTED_FOR_WORD_LEVEL_ACCURACY",
            "gate_result": "6 of 20 checks failed (see android_export/zh/REPORT.md)",
            "fp32_timestamps_exact_vs_engine": "693/693 tokens, 0 ms",
            "int8_vs_fp32_timestamp_mae_ms": 19.96,
            "int8_vs_fp32_timestamp_max_ms": 1340.0,
            "int8_cosine_mean_alphas": 0.9888,
            "int8_cosine_min_alphas": 0.9127,
            "int8_cosine_mean_peaks": 0.9597,
            "int8_cosine_min_peaks": 0.5492,
            "notes": (
                "fp32 timestamps are exact against the FunASR engine; INT8 is "
                "inside the 20 ms MAE gate but the worst single token is 1340 ms. "
                "us_peaks uses a closed-form cumsum/floor substitute for "
                "cif_wo_hidden (max|delta| 12) - an open contract deviation."
            ),
        },
    },
    "ja": {
        "engine": "ctc-hubert-phoneme",
        "engine_notes": (
            "Phoneme-level CTC (blank=PAD=0) over a 48-token phoneme inventory. "
            "Android must implement (a) G2P to phonemes, (b) the char<->phoneme "
            "mapper (ja_pronunciation.py / map_spans_to_char_units), (c) CTC "
            "Viterbi. The model itself has NO baked normalisation."
        ),
        "source_model": "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4",
        "source_revision": "f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99",
        "license": "Apache-2.0",
        "model": ("android_export/ja/hubert-ja-phoneme-ctc-int8.onnx", "ja-hubert-phoneme-ctc-int8.onnx"),
        "extra": [
            (("models/hf/models--prj-beatrice--japanese-hubert-base-phoneme-ctc-v4/"
              "snapshots/f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99/vocab.json"),
             "ja-vocab.json", "tokenizer"),
        ],
        "g2p_dependency": {
            "required_on_device": True,
            "recommended": "bundle OpenJTalk + NAIST dictionary and call it via JNI",
            "not_shipped_here": True,
            "openjtalk_code_license": "MIT (pyopenjtalk); modified BSD (NAIST) for the dictionary",
            "dictionary_name": "open_jtalk_dic_utf_8-1.11",
            "dictionary_bytes": 107300000,
            "dictionary_license_file": "COPYING (NAIST three-clause BSD)",
            "note": (
                "The dictionary is NOT part of this pack. If Android bundles it, it "
                "must reproduce the NAIST copyright notice and disclaimer. The HTS "
                "voice (mei_normal.htsvoice) is synthesis-only and must not be "
                "shipped for G2P."
            ),
        },
        "accuracy": {
            "status": "NOT_ACCEPTED_FOR_WORD_LEVEL_ACCURACY",
            "gate_result": "3 of 15 gate rows failed; one further reference row is non-gate (see android_export/ja/REPORT.md)",
            "int8_vs_fp32_char_start_mae_ms": 147.06,
            "int8_vs_fp32_chars": 1224,
            "int8_vs_fp32_lines": 120,
            "reference_test_vector_mae_ms": 1.90,
            "reference_real_singing_mae_ms": 173.99,
            "fp32_vs_torch_baseline_mae_ms": 0.0,
            "int8_cosine_mean": 0.997174,
            "int8_cosine_min": 0.972780,
            "notes": (
                "Same-methodology PyTorch-vs-FP32 baseline is 0.00 ms, so the "
                "147 ms is purely a quantisation effect. Japanese remains "
                "EXPERIMENTAL: the acoustic model is trained on read speech."
            ),
        },
    },
}

NORMALIZATION = {
    "en": "NFKC -> curly quotes to ' -> drop [^A-Za-z0-9' ] -> collapse spaces -> upper "
          "(src/autosync/lyrics/normalize.py::normalize_en)",
    "zh": "NFKC -> keep only CJK/kana/digits; Latin is dropped by design "
          "(src/autosync/lyrics/normalize.py::normalize_cjk)",
    "ja": "NFKC -> pyopenjtalk G2P to phonemes (Latin is read letter-by-letter); "
          "see g2p_dependency",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def asset(src: Path, pack_name: str, role: str, note: str = "") -> dict:
    return {
        "name": pack_name,
        "role": role,
        "download_url": f"{RELEASE_DOWNLOAD}/{pack_name}",
        "bytes": src.stat().st_size,
        "sha256": sha256_file(src),
        "note": note,
    }


def build() -> dict:
    index: dict = {
        "schema": "melox.autosync.model-index.1",
        "pack_version": PACK_VERSION,
        "release_tag": TAG,
        "release_url": f"https://github.com/{REPO_SLUG}/releases/tag/{TAG}",
        "prerelease": True,
        "min_backend_protocol": MIN_BACKEND_PROTOCOL,
        "protocol_schema": "melox.autosync.onnx.1",
        "accuracy_notice": (
            "BETA: the models are loadable and deterministic, but word/character "
            "level timing has NOT been accepted against a human reference. INT8 "
            "timelines differ from FP32 by about 0.12-0.17 s (en/ja) and about "
            "0.02 s (zh) on real songs. Do not advertise per-word precision."
        ),
        "languages": {},
    }
    binary_assets: list[dict] = []
    for lang, spec in LANGUAGES.items():
        build_manifest = json.loads(
            (REPO_ROOT / "android_export" / lang / "manifest.json").read_text(encoding="utf-8")
        )
        model_src = REPO_ROOT / spec["model"][0]
        expected = build_manifest["files"]["int8"]
        if (expected["bytes"], expected["sha256"]) != (model_src.stat().st_size, sha256_file(model_src)):
            raise SystemExit(f"{lang}: model does not match its build manifest; rebuild it first")

        files = [asset(model_src, spec["model"][1], "model")]
        for src_rel, pack_name, role in spec["extra"]:
            files.append(asset(REPO_ROOT / src_rel, pack_name, role))
        binary_assets.extend(files)

        manifest = {
            "schema": "melox.autosync.model-pack.1",
            "pack_version": PACK_VERSION,
            "min_backend_protocol": MIN_BACKEND_PROTOCOL,
            "language": lang,
            "engine": spec["engine"],
            "engine_notes": spec["engine_notes"],
            "source_model": spec["source_model"],
            "source_revision": spec["source_revision"],
            "license": spec["license"],
            "license_notice": (
                "Apache-2.0. This pack contains MODIFIED weights (dynamic INT8 "
                "quantisation with per-channel weight scales; for zh the feature "
                "frontend and for en the input normalisation are baked into the "
                "graph). Upstream copyright remains with the original authors; see "
                "NOTICE.md. No endorsement by the upstream authors is implied."
            ),
            "sample_rate": build_manifest.get("sample_rate", 16000),
            "frame_hop_samples": build_manifest.get("frame_hop_samples"),
            "frame_hop_seconds": build_manifest.get("frame_hop_seconds"),
            "input": build_manifest.get("input"),
            "aux_inputs": build_manifest.get("aux_inputs"),
            "outputs": build_manifest.get("outputs"),
            "blank_id": build_manifest.get("blank_id"),
            "text": {
                "tokens_count": build_manifest.get("tokens_count"),
                "tokens_file": next(f["name"] for f in files if f["role"] == "tokenizer"),
                "word_delimiter_id": build_manifest.get("word_delimiter_id"),
                "normalization": NORMALIZATION[lang],
            },
            "files": files,
            "accuracy": spec["accuracy"],
        }
        if "g2p_dependency" in spec:
            manifest["g2p_dependency"] = spec["g2p_dependency"]
        index["languages"][lang] = manifest

    index_text = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    rows: list[str] = []
    for lang, spec in LANGUAGES.items():
        for f in index["languages"][lang]["files"]:
            rows.append(
                f"| `{f['name']}` | {lang} | {f['role']} | {f['bytes']:,} | `{f['sha256']}` |"
            )
    readme = (REPO_ROOT / "release/android-models/PACK_README.md").read_text(encoding="utf-8")
    readme = readme.replace("{{ASSET_TABLE}}", "\n".join(rows))

    text_files = {"android-models.json": index_text, "README.md": readme}
    for lang in LANGUAGES:
        text_files[f"{lang}-manifest.json"] = (
            json.dumps(index["languages"][lang], indent=2, ensure_ascii=False) + "\n"
        )
    for doc, name in (
        ("release/android-models/NOTICE.md", "NOTICE.md"),
        ("release/android-models/LICENSE-APACHE-2.0.txt", "LICENSE-APACHE-2.0.txt"),
    ):
        text_files[name] = (REPO_ROOT / doc).read_text(encoding="utf-8")

    lines = [f'{f["sha256"]}  {f["name"]}' for f in sorted(binary_assets, key=lambda f: f["name"])]
    lines += [f"{sha256_text(body)}  {name}" for name, body in sorted(text_files.items())]
    text_files["checksums.sha256"] = "\n".join(lines) + "\n"
    return {"index": index, "binary_assets": binary_assets, "text_files": text_files}


def write_pack(result: dict) -> None:
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    for name, body in result["text_files"].items():
        (PACK_DIR / name).write_text(body, encoding="utf-8")
    for lang in LANGUAGES:
        (PACK_DIR / lang).mkdir(exist_ok=True)
        (PACK_DIR / lang / "manifest.json").write_text(
            result["text_files"][f"{lang}-manifest.json"], encoding="utf-8")


def upload(result: dict) -> None:
    import shutil
    import tempfile

    exists = subprocess.run(["gh", "release", "view", TAG, "--repo", REPO_SLUG],
                            capture_output=True, text=True, check=False).returncode == 0
    if not exists:
        subprocess.run(
            ["gh", "release", "create", TAG, "--repo", REPO_SLUG, "--prerelease",
             "--title", f"Android model pack {PACK_VERSION} (BETA - word-level accuracy not accepted)",
             "--notes-file", str(PACK_DIR / "README.md")],
            check=True,
        )
    else:
        print(f"release {TAG} already exists; updating notes and assets")
    # ``gh release upload <file>#<label>`` only sets a *label*; the asset name is
    # the file name. Stage every binary under its pack name so the download URLs
    # in android-models.json match the real asset names.
    stage = Path(tempfile.mkdtemp(prefix="melox_pack_"))
    try:
        planned = [(REPO_ROOT / spec["model"][0], spec["model"][1]) for spec in LANGUAGES.values()]
        planned += [(REPO_ROOT / src_rel, name)
                    for s in LANGUAGES.values() for src_rel, name, _role in s["extra"]]
        for src, name in planned:
            staged = stage / name
            shutil.copy2(src, staged)
            subprocess.run(["gh", "release", "upload", TAG, str(staged),
                            "--repo", REPO_SLUG, "--clobber"], check=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    for name in sorted(result["text_files"]):
        subprocess.run(["gh", "release", "upload", TAG, str(PACK_DIR / name),
                        "--repo", REPO_SLUG, "--clobber"], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="verify and print only")
    ap.add_argument("--upload", action="store_true", help="create the GitHub release and upload")
    args = ap.parse_args()

    result = build()
    print(f"pack {PACK_VERSION}  tag={TAG}  languages={list(result['index']['languages'])}")
    for f in result["binary_assets"]:
        print(f"  {f['bytes']:>12,}  {f['sha256']}  {f['name']}")
    print(f"  text files: {sorted(result['text_files'])}")
    if args.dry_run:
        print("dry-run: nothing written, nothing uploaded")
        return 0
    write_pack(result)
    print(f"pack files written to {PACK_DIR}")
    if args.upload:
        upload(result)
        print(f"release created: https://github.com/{REPO_SLUG}/releases/tag/{TAG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
