#!/usr/bin/env python
"""Export ``facebook/wav2vec2-base-960h`` to ONNX (fp32 + INT8) for MeloX-Android AutoSync.

Contract (authoritative): ``docs/ANDROID_ONNX_EXPORT_PROMPT.md``.

* input ``input_values`` float32 ``[1, N]``, raw 16 kHz mono PCM in [-1, 1];
* zero-mean / unit-variance normalization (population variance, eps 1e-7) is
  **baked into the graph**, matching ``Wav2Vec2FeatureExtractor(do_normalize=True)``;
* outputs ``logits`` and ``log_probs`` (``log_softmax(logits, -1)``), float32
  ``[1, T, 32]``;
* eval mode, float32, no attention mask, opset 17, standard ONNX ops only,
  single self-contained file (no external data);
* INT8 = ``onnxruntime.quantization.quantize_dynamic`` with
  ``weight_type=QuantType.QInt8`` restricted to MatMul/Gemm (the conv front-end
  stays fp32, as the task book requires; in ORT 1.31 the *default* op set also
  contains Conv/Gather, so the type list is pinned explicitly).

Stages are run as separate processes by the default ``all`` mode so that the
PyTorch model and the ONNX quantization step never coexist in memory (the
machine has little free RAM).

Usage
-----
    .venv/bin/python android_export/en/export_model.py            # all stages
    .venv/bin/python android_export/en/export_model.py --stage export
    .venv/bin/python android_export/en/export_model.py --stage quantize
    .venv/bin/python android_export/en/export_model.py --stage vectors
    .venv/bin/python android_export/en/export_model.py --stage manifest
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve()
OUT_DIR = SCRIPT.parent
REPO_ROOT = OUT_DIR.parents[1]

SNAPSHOT = (
    REPO_ROOT
    / "models/hf/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
)
FP32_NAME = "wav2vec2-base-960h-fp32.onnx"
INT8_NAME = "wav2vec2-base-960h-int8.onnx"
MANIFEST_NAME = "manifest.json"

NORM_EPS = 1e-7
OPSET = 17


# --------------------------------------------------------------------------- #
# test-vector audio                                                    (stage 3)
# --------------------------------------------------------------------------- #
# 16 kHz mono PCM16, 2-5 s, real alignable speech (see REPORT.md for sources).
# Sources are local, gitignored material only; `text` is a short ASCII excerpt of
# the public-domain LibriSpeech transcript / the repo's own synthetic clip.
VECTOR_SPECS = [
    {
        "file": "vector1.wav",
        "source": "experiments/audio/librispeech_sample1.flac",
        "start": 0.08,
        "end": 4.46,
        "text": "going along slushy country roads and speaking to damp audiences in draughty",
        "notes": (
            "LibriSpeech dev-clean read speech, 16 kHz; window 0.08-4.46 s; words 0-11 of the "
            "public-domain transcript (adjoining .lrc supplies it verbatim)"
        ),
    },
    {
        "file": "vector2.wav",
        "source": "experiments/audio/librispeech_sample1.flac",
        "start": 7.30,
        "end": 10.80,
        "text": "he'll have to put in an appearance at some place of worship on sunday",
        "notes": (
            "LibriSpeech dev-clean read speech, 16 kHz; window 7.30-10.80 s; words 19-32 of the "
            "public-domain transcript"
        ),
    },
    {
        "file": "vector3.wav",
        "source": "experiments/audio/librispeech_sample2.flac",
        "start": 5.90,
        "end": 10.16,
        "text": "say can i leave these here these were a small",
        "notes": (
            "LibriSpeech dev-clean read speech, 16 kHz; window 5.90-10.16 s; the clip has no "
            "external transcript in the repo, so `text` is the FP32 greedy CTC decode of the "
            "clip (the same decode was checked against the public-domain transcript of "
            "librispeech_sample1 and matched it verbatim)"
        ),
    },
]


def stage_vectors() -> None:
    """Write the 3 device self-check wavs (16 kHz mono PCM16) from local sources."""
    import numpy as np
    import soundfile as sf

    vec_dir = OUT_DIR / "test_vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)
    for spec in VECTOR_SPECS:
        src = REPO_ROOT / spec["source"]
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        audio = data.mean(axis=1)
        if sr != 16000:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(16000, sr)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
            sr = 16000
        seg = audio[int(round(spec["start"] * sr)) : int(round(spec["end"] * sr))]
        dur = len(seg) / sr
        assert 2.0 <= dur <= 5.0, f"{spec['file']}: duration {dur:.2f}s outside 2-5 s"
        out = vec_dir / spec["file"]
        sf.write(str(out), seg, sr, subtype="PCM_16")
        info = sf.info(str(out))
        print(
            f"[vectors] {out.name}: {info.frames} frames, {info.frames/sr:.3f}s, "
            f"{info.samplerate} Hz, {info.channels} ch, {info.subtype}"
        )
        print(f"[vectors]   text: {spec['text']!r}")


# --------------------------------------------------------------------------- #
# export / quantize                                                           #
# --------------------------------------------------------------------------- #
def _load_torch_model():
    import warnings

    warnings.filterwarnings("ignore")
    import torch
    from transformers import Wav2Vec2ForCTC

    torch.set_num_threads(1)
    model = Wav2Vec2ForCTC.from_pretrained(
        str(SNAPSHOT), local_files_only=True, low_cpu_mem_usage=True
    )
    model.eval()
    return torch, model


class Wav2Vec2CTCNorm:
    """placeholder name kept for readability; the real wrapper is built in
    ``_build_wrapper`` so that importing this module never imports torch."""


def _build_wrapper(torch, model):
    import torch as _torch

    eps = NORM_EPS

    class _Norm(_torch.nn.Module):
        """``(x - mean) / sqrt(var + eps)`` (population variance).

        Computed in float64 on purpose: ONNX Runtime's *float32* reduction over
        long inputs (200k+ samples) drifts ~1e-5 relative on the standard
        deviation, which pushes the normalised tensor beyond the 1e-5 parity
        budget vs ``Wav2Vec2FeatureExtractor``. float64 reductions are exact to
        ~1e-15, and the result is cast back to float32 for the model.
        """

        def forward(self, x):
            xd = x.double()
            mean = xd.mean(dim=1, keepdim=True)
            xc = xd - mean
            var = (xc * xc).mean(dim=1, keepdim=True)
            return (xc / _torch.sqrt(var + eps)).float()

    class Wrapper(_torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.normalize = _Norm()
            self.model = m

        def forward(self, input_values):  # [1, N] raw PCM
            x = self.normalize(input_values)
            logits = self.model(x).logits
            return logits, _torch.log_softmax(logits, dim=-1)

    return Wrapper(model).eval()


def stage_export(force: bool = False) -> None:
    fp32_path = OUT_DIR / FP32_NAME
    if fp32_path.exists() and not force:
        print(f"[export] {fp32_path.name} exists; use --force to re-export")
        return
    import numpy as np
    import onnx
    import onnxruntime as ort

    torch, model = _load_torch_model()
    wrapper = _build_wrapper(torch, model)

    dummy = torch.zeros(1, 16000)
    t0 = time.time()
    with torch.inference_mode():
        lg, lp = wrapper(dummy)
    print(f"[export] torch sanity forward: logits {tuple(lg.shape)}, log_probs {tuple(lp.shape)} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(fp32_path),
        input_names=["input_values"],
        output_names=["logits", "log_probs"],
        dynamic_axes={"input_values": {1: "n"}, "logits": {1: "t"}, "log_probs": {1: "t"}},
        opset_version=OPSET,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
    )
    print(f"[export] torch.onnx.export (opset {OPSET}) done in {time.time()-t0:.1f}s")

    del wrapper, model, lg, lp, dummy
    gc.collect()

    model_proto = onnx.load(str(fp32_path))
    opset_imports = [(io.domain, io.version) for io in model_proto.opset_import]
    domains = sorted({n.domain for n in model_proto.graph.node if n.domain})
    print(f"[export] opset_import {opset_imports}; node domains {domains}")
    assert all(d == "" for d in domains), f"non-standard ops: {domains}"
    assert all(v <= OPSET for _, v in opset_imports), "opset > 17"
    assert not any(onnx.external_data_helper.uses_external_data(t) for t in model_proto.graph.initializer), \
        "external data present: not a single self-contained file"
    onnx.checker.check_model(str(fp32_path))
    print(f"[export] onnx.checker: OK, {fp32_path.stat().st_size} bytes (no external data)")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(fp32_path), sess_options=so, providers=["CPUExecutionProvider"])
    print("[export] ORT I/O:", [(i.name, i.shape, i.type) for i in sess.get_inputs()],
          [(o.name, o.shape, o.type) for o in sess.get_outputs()])
    out = sess.run(None, {"input_values": np.zeros((1, 4000), dtype=np.float32)})
    for name, arr in zip(["logits", "log_probs"], out):
        assert np.isfinite(arr).all(), f"{name}: NaN/Inf on silence"
        print(f"[export]   {name} {arr.shape} finite")
    del sess, out, model_proto
    gc.collect()


def stage_quantize(force: bool = False) -> None:
    int8_path = OUT_DIR / INT8_NAME
    if int8_path.exists() and not force:
        print(f"[quantize] {int8_path.name} exists; use --force to re-quantize")
        return
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    t0 = time.time()
    quantize_dynamic(
        str(OUT_DIR / FP32_NAME),
        str(int8_path),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        # Per-channel weight scales. Measured on 29 real lyric lines / 168 words
        # (same validation harness): start MAE vs FP32 drops 219.05 ms -> 121.19 ms
        # (-45%; per song 77->35, 218->84, 354->207, 281->265) for +0.42 MB.
        # Excluding the CTC head or enabling ActivationSymmetric changed nothing
        # (217.74 ms / 121.19 ms respectively), so per-channel is the only lever
        # that mattered. Quantising Conv/Gather as well is much worse (425.36 ms).
        per_channel=True,
        use_external_data_format=False,
    )
    print(f"[quantize] quantize_dynamic done in {time.time()-t0:.1f}s, {int8_path.stat().st_size} bytes")

    m = onnx.load(str(int8_path))
    opset_imports = [(io.domain, io.version) for io in m.opset_import]
    domains = sorted({n.domain for n in m.graph.node if n.domain})
    q_ops = sorted({n.op_type for n in m.graph.node if n.op_type.endswith("Integer") or "Quantize" in n.op_type})
    print(f"[quantize] opset_import {opset_imports}; node domains {domains}")
    print(f"[quantize] quantized node ops: {q_ops}")
    assert all(d == "" for d in domains), f"non-standard ops: {domains}"
    assert all(v <= OPSET for _, v in opset_imports), "opset > 17"
    onnx.checker.check_model(str(int8_path))
    del m
    gc.collect()

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(int8_path), sess_options=so, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_values": __import__("numpy").zeros((1, 4000), dtype="float32")})
    for name, arr in zip(["logits", "log_probs"], out):
        assert __import__("numpy").isfinite(arr).all(), f"{name}: NaN/Inf on silence"
        print(f"[quantize]   {name} {arr.shape} finite")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_manifest() -> None:
    fp32_path, int8_path = OUT_DIR / FP32_NAME, OUT_DIR / INT8_NAME
    vocab = json.loads((SNAPSHOT / "vocab.json").read_text(encoding="utf-8"))
    tokens = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    assert len(tokens) == 32 and tokens[0] == "<pad>" and tokens[4] == "|"
    manifest = {
        "schema": "melox.autosync.onnx.1",
        "language": "en",
        "source_model": "facebook/wav2vec2-base-960h",
        "license": "Apache-2.0",
        "sample_rate": 16000,
        "frame_hop_samples": 320,
        "frame_hop_seconds": 0.02,
        "input": {
            "name": "input_values",
            "shape": [1, None],
            "dtype": "float32",
            "preprocessing": "raw 16k mono PCM in [-1,1]; zero-mean unit-var normalization is baked into the graph",
        },
        "outputs": [
            {"name": "logits", "shape": [1, None, 32], "dtype": "float32"},
            {"name": "log_probs", "shape": [1, None, 32], "dtype": "float32"},
        ],
        "blank_id": 0,
        "word_delimiter_id": 4,
        "tokens": tokens,
        "files": {
            "fp32": {"name": FP32_NAME, "sha256": _sha256(fp32_path), "bytes": fp32_path.stat().st_size},
            "int8": {"name": INT8_NAME, "sha256": _sha256(int8_path), "bytes": int8_path.stat().st_size},
        },
    }
    out = OUT_DIR / MANIFEST_NAME
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[manifest] wrote {out}")
    for key in ("fp32", "int8"):
        f = manifest["files"][key]
        print(f"[manifest]   {f['name']}: {f['bytes']} bytes sha256 {f['sha256']}")
    print(f"[manifest]   tokens: {tokens}")


STAGES = ("export", "quantize", "vectors", "manifest")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    ap.add_argument("--force", action="store_true", help="redo stages even if outputs exist")
    ap.add_argument("--only-vectors", action="store_true", help="alias for --stage vectors")
    args = ap.parse_args(argv)

    if args.only_vectors:
        args.stage = "vectors"

    if args.stage == "all":
        for stage in STAGES:
            # fresh subprocess per stage: keeps peak RSS to one heavyweight object at a time
            cmd = [sys.executable, str(SCRIPT), "--stage", stage]
            if args.force:
                cmd.append("--force")
            print(f"\n===== stage {stage} =====", flush=True)
            subprocess.run(cmd, check=True)
        return 0

    if args.stage == "export":
        stage_export(args.force)
    elif args.stage == "quantize":
        stage_quantize(args.force)
    elif args.stage == "vectors":
        stage_vectors()
    elif args.stage == "manifest":
        stage_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
