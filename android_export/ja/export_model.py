#!/usr/bin/env python
"""Export the Japanese HuBERT phoneme-CTC model to ONNX (fp32 + dynamic INT8).

Deliverable of `docs/ANDROID_ONNX_EXPORT_JA.md` (MeloX-AutoSync).

Model: ``prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`` (Apache-2.0),
loaded **offline** from the local HF snapshot.

Hard contract (task book §2.1):
  input  ``input_values`` float32 ``[1, N]``  raw 16k mono PCM in [-1, 1],
         **no external normalization** (``do_normalize=false``; HuBERT uses an
         internal group-norm frontend).
  output ``logits``     float32 ``[1, T, 48]``
         ``log_probs``  float32 ``[1, T, 48]`` = log_softmax(logits, -1)
  ``T = floor((N - 400) / 320) + 1`` (20 ms frames @ 16 kHz)

INT8: ``onnxruntime.quantization.quantize_dynamic`` (QInt8 weights, MatMul/Gemm
only; conv frontend and LayerNorm stay fp32; ``per_channel=True`` weight scales,
which measured uniformly better than per-tensor on this model).  Single file, no
external data, opset 17, standard ONNX ops only.

Usage::

    .venv/bin/python android_export/ja/export_model.py

Regenerates the two .onnx files and ``manifest.json`` in this directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Keep the export single-threaded and fully offline (shared machine / shared venv).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = (
    REPO_ROOT
    / "models"
    / "hf"
    / "models--prj-beatrice--japanese-hubert-base-phoneme-ctc-v4"
    / "snapshots"
    / "f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99"
)
SOURCE_MODEL = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"
FP32_NAME = "hubert-ja-phoneme-ctc-fp32.onnx"
INT8_NAME = "hubert-ja-phoneme-ctc-int8.onnx"

G2P_EVIDENCE = (
    "Option A (bundle OpenJTalk+dict via JNI). The reference chain "
    "(ja_pronunciation.py) needs MeCab morphological analysis plus the Open JTalk "
    "dictionary to read kanji; the same-source Open JTalk 1.11 frontend is what "
    "pyopenjtalk calls. Local evidence: open_jtalk_dic_utf_8-1.11 is 103.07 MB "
    "raw (sys.dic) / 21.67 MB gzip, Open JTalk+HTS Engine are BSD-family "
    "(see REPORT.md section 4.3). Option B (pure Kotlin port) was NOT delivered "
    "and cannot be accepted without a full dictionary+morphology re-implementation "
    "validated against the golden vectors; option C would make JA undeliverable."
)


class HubertCTCWrap:
    """Built at runtime to avoid importing torch at module import time."""

    @staticmethod
    def build(model):
        import torch

        class _Wrap(torch.nn.Module):
            """raw PCM -> (logits, log_probs); NO external normalization."""

            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_values):
                logits = self.m(input_values).logits
                return logits, torch.log_softmax(logits.float(), dim=-1)

        return _Wrap(model).eval()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_vocab(snapshot: Path) -> list[str]:
    with open(snapshot / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    n = max(vocab.values()) + 1
    tokens: list[str | None] = [None] * n
    for tok, idx in vocab.items():
        tokens[idx] = tok
    if any(t is None for t in tokens):
        raise RuntimeError("vocab.json has holes")
    return tokens  # type: ignore[return-value]


def export_fp32(snapshot: Path, out_path: Path) -> None:
    import torch
    from transformers import AutoModelForCTC

    torch.manual_seed(0)
    t0 = time.time()
    model = AutoModelForCTC.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    print(f"[fp32] model loaded in {time.time() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    wrapper = HubertCTCWrap.build(model)
    if out_path.exists():
        out_path.unlink()
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (torch.zeros(1, 16000, dtype=torch.float32),),
        str(out_path),
        input_names=["input_values"],
        output_names=["logits", "log_probs"],
        dynamic_axes={
            "input_values": {1: "n"},
            "logits": {1: "t"},
            "log_probs": {1: "t"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[fp32] exported in {time.time() - t0:.1f}s -> {out_path.name} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    del wrapper, model
    gc.collect()


def quantize(int8_path: Path, fp32_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    t0 = time.time()
    if int8_path.exists():
        int8_path.unlink()
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"[int8] dynamic quantization in {time.time() - t0:.1f}s -> "
          f"{int8_path.name} ({int8_path.stat().st_size / 1e6:.1f} MB)")


def smoke_test(path: Path) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(path))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = np.zeros((1, 8000), dtype=np.float32)
    logits, log_probs = sess.run(None, {"input_values": x})
    assert logits.shape == (1, 24, 48), logits.shape
    assert log_probs.shape == (1, 24, 48), log_probs.shape
    assert np.isfinite(logits).all() and np.isfinite(log_probs).all()
    del sess
    gc.collect()
    print(f"[smoke] {path.name}: checker + CPUExecutionProvider OK, T=24 for N=8000")


def write_manifest(out_dir: Path, snapshot: Path, fp32: Path, int8: Path) -> None:
    tokens = read_vocab(snapshot)
    manifest = {
        "schema": "melox.autosync.onnx.1",
        "language": "ja",
        "source_model": SOURCE_MODEL,
        "license": "Apache-2.0",
        "sample_rate": 16000,
        "frame_hop_samples": 320,
        "frame_hop_seconds": 0.02,
        "input": {
            "name": "input_values",
            "shape": [1, None],
            "dtype": "float32",
            "preprocessing": (
                "raw 16k mono PCM in [-1,1]; NO external normalization "
                "(preprocessor do_normalize=false; HuBERT uses internal group norm)"
            ),
        },
        "outputs": [
            {"name": "logits", "shape": [1, None, 48], "dtype": "float32"},
            {"name": "log_probs", "shape": [1, None, 48], "dtype": "float32"},
        ],
        "blank_id": 0,
        "phoneme_tokens": tokens,
        "g2p": {
            "reference": "pyopenjtalk (OpenJTalk) via ja_pronunciation.py",
            "wrap_pau": True,
            "device_option": "A",
            "device_option_evidence": G2P_EVIDENCE,
        },
        "files": {
            "fp32": {
                "name": fp32.name,
                "sha256": sha256_of(fp32),
                "bytes": fp32.stat().st_size,
            },
            "int8": {
                "name": int8.name,
                "sha256": sha256_of(int8),
                "bytes": int8.stat().st_size,
            },
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[manifest] wrote manifest.json (48 tokens, hashes)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--skip-quant", action="store_true")
    args = ap.parse_args(argv)

    snapshot = args.snapshot.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (snapshot / "model.safetensors").exists():
        print(f"snapshot not found: {snapshot}", file=sys.stderr)
        return 2

    fp32 = out_dir / FP32_NAME
    int8 = out_dir / INT8_NAME

    export_fp32(snapshot, fp32)
    smoke_test(fp32)
    if not args.skip_quant:
        quantize(int8, fp32)
        smoke_test(int8)
    write_manifest(out_dir, snapshot, fp32, int8)

    print("\nDone. Files:")
    for p in (fp32, int8):
        print(f"  {p}  {p.stat().st_size} bytes  sha256={sha256_of(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
