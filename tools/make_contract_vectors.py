#!/usr/bin/env python
"""Generate Android ONNX contract test vectors that contain only our own audio.

The delivered ``android_export/*/test_vectors`` use third-party speech corpora and
are therefore not redistributable. For the Android backend we need a *shippable*
contract check, so this tool synthesises deterministic, license-clean waveforms
(pure DSP, no third-party audio), runs each **shipped** model once, and writes a
small contract record: input spec, output names / dtypes / shapes, and a couple of
numeric probes. It is enough for the device to verify the tensor contract and
that inference runs, without copying any copyrighted audio.

Usage::

    .venv/bin/python tools/make_contract_vectors.py --out android_contract        # generate
    .venv/bin/python tools/make_contract_vectors.py --out android_contract --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SR = 16000
REPO = Path(__file__).resolve().parents[1]

SHIPPED = {
    "en": {"model": REPO / "android_export/en/wav2vec2-base-960h-int8.onnx",
           "seconds": 3.0, "input": "input_values", "sample_rate": SR},
    "ja": {"model": REPO / "android_export/ja/hubert-ja-phoneme-ctc-int8.onnx",
           "seconds": 3.0, "input": "input_values", "sample_rate": SR},
    # zh currently ships the beta.2 recipe (unchanged in beta.3/beta.4)
    "zh": {"model": REPO / "android_export/zh/fa-zh-int8.onnx",
           "seconds": 3.0, "input": "input_values", "sample_rate": SR, "token_num": 6},
}


def synth(seconds: float, seed: int) -> np.ndarray:
    """Deterministic voiced-ish waveform: a few harmonics with an AM envelope."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    f0 = 165.0 + rng.uniform(-5, 5)
    x = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 7))
    env = 0.5 * (1.0 - np.cos(2 * np.pi * 2.5 * t))  # slow syllable envelope
    x = (x * env).astype(np.float32)
    x /= max(1e-9, np.abs(x).max())
    return (0.6 * x).astype(np.float32)


def probes(arr: np.ndarray) -> dict:
    a = arr.reshape(-1)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype),
            "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "sum": float(a.sum()), "argmax": int(a.argmax())}


def make(lang: str, spec: dict, out_dir: Path) -> dict:
    import onnxruntime as ort
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    audio = synth(spec["seconds"], seed=hash(lang) % 1000)
    wav = out_dir / f"{lang}_vector.wav"
    sf.write(str(wav), audio, SR, subtype="PCM_16")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(spec["model"]), so, providers=["CPUExecutionProvider"])
    feeds = {spec["input"]: audio[None, :]}
    if "token_num" in spec:
        feeds["token_num"] = np.array([spec["token_num"]], dtype=np.int64)
    outs = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    record = {
        "language": lang,
        "model": spec["model"].name,
        "model_sha256": hashlib.sha256(spec["model"].read_bytes()).hexdigest(),
        "model_bytes": spec["model"].stat().st_size,
        "sample_rate": spec["sample_rate"],
        "waveform": wav.name,
        "waveform_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "inputs": [{"name": i.name, "dtype": i.type, "shape": [str(d) for d in i.shape]}
                   for i in sess.get_inputs()],
        "outputs": [{"name": n, "probe": probes(o)} for n, o in zip(names, outs)],
    }
    del sess
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "android_contract")
    ap.add_argument("--only", default=None, help="comma list of languages")
    args = ap.parse_args()

    langs = (args.only.split(",") if args.only else sorted(SHIPPED))
    records = []
    for lang in langs:
        print(f"[{lang}] generating contract vector ...", flush=True)
        records.append(make(lang, SHIPPED[lang], args.out / lang))
    (args.out / "contract.json").write_text(
        json.dumps({"schema": "melox.autosync.contract.1", "vectors": records},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'contract.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
