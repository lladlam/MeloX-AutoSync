#!/usr/bin/env python
"""Static QDQ quantisation with a real-singing calibration set.

The delivered INT8 models use *dynamic* quantisation.  The export report for
English already suggested trying **static QDQ + calibration** with real singing
audio.  This tool builds that alternative so it can be compared on a held-out
pool:

    fp32 ONNX  --quantize_static(calibration=real singing)--\u2192  QDQ INT8 ONNX

Only the MatMul/Gemm weights are quantised (the conv front-end and LayerNorm stay
float, matching the dynamic recipe), weight type QInt8, per-channel, QDQ
format.  Calibration audio comes from a directory tree of ``*.wav`` files that
the caller guarantees is disjoint from the validation pool.

Output goes to ``--out`` (a scratch path); nothing in the repository is
overwritten.

Usage::

    .venv/bin/python tools/quantize_static_calib.py \
        --fp32 android_export/ja/hubert-ja-phoneme-ctc-fp32.onnx \
        --calib-dir /mnt/.../gtsinger/Japanese/JA-Soprano-1 \
        --out /tmp/opencode/quant_eval/ja_qdq.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SR = 16000


def _read_audio(path: Path):
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(SR, sr)
        audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)
    return audio


class WaveReader:
    """Feed up to ``limit`` wav files as ``input_values`` calibration samples."""

    def __init__(self, wavs: list[Path], limit: int):
        self.wavs = wavs[:limit]
        self.i = 0

    def get_next(self):
        import numpy as np

        if self.i >= len(self.wavs):
            return None
        audio = _read_audio(self.wavs[self.i])
        self.i += 1
        print(f"  calib {self.i}/{len(self.wavs)}: {self.wavs[self.i-1].name} "
              f"({len(audio)/SR:.1f}s)", flush=True)
        return {"input_values": audio[None, :].astype(np.float32)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    args = ap.parse_args()

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    wavs = sorted(p for p in args.calib_dir.rglob("*.wav") if p.is_file())
    if not wavs:
        print(f"no calibration wavs under {args.calib_dir}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(wavs)} calibration wavs found; using {min(args.limit, len(wavs))}")

    reader = WaveReader(wavs, args.limit)
    quantize_static(
        model_input=str(args.fp32),
        model_output=str(args.out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=args.per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=False,
    )
    print(f"[done] {args.out}  {args.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
