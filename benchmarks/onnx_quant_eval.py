#!/usr/bin/env python
"""Absolute + INT8-vs-FP32 accuracy of the exported ONNX aligners on GTSinger.

This is the evaluation the export reports were missing: instead of only asking
"how far is INT8 from FP32", it asks "how far is each model from the human
word/character boundaries in a corpus we never trained on".

It runs the *exported* ONNX graphs (fp32 and int8) through the repository's own
forced-aligner, on GTSinger segments held out from any calibration pool, and
reports:

* absolute word/char start MAE (and P90) for fp32 and int8 against GTSinger;
* the INT8-vs-FP32 start MAE (same metric as the export reports);
* coverage / feasibility counts.

English and Japanese use CTC Viterbi (blank=0) on the model logits. Chinese uses
the CIF ``us_alphas`` + ``ts_prediction_lfr6_standard`` post-process.

GTSinger is CC BY-NC-SA 4.0: audio and annotations stay local and are never
redistributed. Only aggregate metrics are printed.

Usage::

    .venv/bin/python benchmarks/onnx_quant_eval.py --language en \
        --pool /mnt/.../gtsinger_val --limit 12 --out /tmp/opencode/en_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

EN_FP32 = REPO / "android_export/en/wav2vec2-base-960h-fp32.onnx"
EN_INT8 = REPO / "android_export/en/wav2vec2-base-960h-int8.onnx"
EN_VOCAB = (REPO / "models/hf/models--facebook--wav2vec2-base-960h/snapshots/"
            "22aad52d435eb6dbaf354bdad9b0da84ce7d6156/vocab.json")
JA_FP32 = REPO / "android_export/ja/hubert-ja-phoneme-ctc-fp32.onnx"
JA_INT8 = REPO / "android_export/ja/hubert-ja-phoneme-ctc-int8.onnx"
ZH_FP32 = REPO / "android_export/zh/fa-zh-fp32.onnx"
ZH_INT8 = REPO / "android_export/zh/fa-zh-int8.onnx"

FRAME_S = 0.02
EXCLUDE = {"<SP>", "<AP>", "<UNK>"}


# --------------------------------------------------------------------------- audio
def read_wav_mono16k(path: Path):
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    return audio


def load_units(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [u for u in data if isinstance(u, dict) and u.get("word") not in EXCLUDE
            and str(u.get("word", "")).strip()]


# --------------------------------------------------------------------------- english
def en_tokens(text: str, vocab: dict[str, int]) -> tuple[list[int], list[tuple[int, int]]]:
    norm = " ".join(text.upper().split())
    tokens: list[int] = []
    groups: list[tuple[int, int]] = []
    for word in norm.split():
        start = len(tokens)
        for ch in word:
            if ch in vocab:
                tokens.append(vocab[ch])
            elif ch == "'":
                continue
        groups.append((start, len(tokens) - 1))
    return tokens, groups


def eval_en(pool: Path, limit: int | None) -> dict:
    import numpy as np
    import onnxruntime as ort

    from autosync.alignment.ctc import forced_align

    vocab = json.loads(EN_VOCAB.read_text(encoding="utf-8"))
    records = json.loads((pool / "manifest.json").read_text())["records"]
    segs = [r for r in records if r["language"] == "en"]
    if limit:
        segs = segs[:limit]

    def sess(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.log_severity_level = 3
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    s_fp, s_i8 = sess(EN_FP32), sess(EN_INT8)
    rows = []
    for rec in segs:
        wav = pool / f"{rec['rel']}.wav"
        units = load_units(pool / f"{rec['rel']}.json")
        if not units:
            continue
        text = " ".join(u["word"] for u in units)
        tokens, groups = en_tokens(text, vocab)
        groups = [(a, b) for a, b in groups if b >= a]
        if len(groups) != len(units):
            rows.append({"rel": rec["rel"], "skipped": f"token groups {len(groups)} != units {len(units)}"})
            continue
        audio = read_wav_mono16k(wav)
        x = audio[None, :]
        lp_fp = s_fp.run(None, {"input_values": x})[1][0]
        lp_i8 = s_i8.run(None, {"input_values": x})[1][0]
        a_fp = forced_align(lp_fp, tokens, blank=0)
        a_i8 = forced_align(lp_i8, tokens, blank=0)
        ref_s = np.array([float(u["start_time"]) for u in units])

        def spans(a, groups=groups):
            out_s, out_e, ok = [], [], []
            for g0, g1 in groups:
                sp = a.spans[g0:g1 + 1]
                good = all(s is not None for s in sp)
                ok.append(good)
                if good:
                    out_s.append(sp[0].start_frame * FRAME_S)
                    out_e.append((sp[-1].end_frame + 1) * FRAME_S)
                else:
                    out_s.append(np.nan)
                    out_e.append(np.nan)
            return np.array(out_s), np.array(out_e), np.array(ok)

        ps_fp, _pe_fp, ok_fp = spans(a_fp)
        ps_i8, _pe_i8, ok_i8 = spans(a_i8)
        both = ok_fp & ok_i8
        d_fp = np.abs(ps_fp - ref_s)
        d_i8 = np.abs(ps_i8 - ref_s)
        rows.append({
            "rel": rec["rel"], "song": rec["song"], "singer": rec["singer"],
            "n_units": len(units),
            "abs_fp32_start_mae_ms": float(np.nanmean(d_fp) * 1000),
            "abs_int8_start_mae_ms": float(np.nanmean(d_i8) * 1000),
            "fp32_feasible": int(ok_fp.sum()), "int8_feasible": int(ok_i8.sum()),
            "int8_vs_fp32_start_mae_ms": float(np.nanmean(np.abs(ps_i8 - ps_fp)[both]) * 1000) if both.any() else None,
            "n_both": int(both.sum()),
            "ref_start_s": ref_s.tolist(),
            "fp32_start_s": np.where(ok_fp, ps_fp, np.nan).tolist(),
            "int8_start_s": np.where(ok_i8, ps_i8, np.nan).tolist(),
        })

    def agg(key):
        vals = [r[key] for r in rows if key in r and r[key] is not None and r[key] == r[key]]
        if not vals:
            return {"n": 0}
        arr = np.array(vals)
        return {"n": len(vals), "mean": float(arr.mean()), "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)), "max": float(arr.max())}

    return {"language": "en", "engine": "ctc-wav2vec2", "pool": str(pool),
            "n_segments": len(rows), "rows": rows,
            "summary": {k: agg(k) for k in
                        ("abs_fp32_start_mae_ms", "abs_int8_start_mae_ms", "int8_vs_fp32_start_mae_ms")}}


# --------------------------------------------------------------------------- japanese
JA_VOCAB = (REPO / "models/hf/models--prj-beatrice--japanese-hubert-base-phoneme-ctc-v4/"
            "snapshots/f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99/vocab.json")
# GTSinger uses some IPA symbols the model vocabulary spells differently.
JA_PH_FIX = {"ɾ": "r", "ɺ": "r", "β": "b", "ɡ": "g"}


def ja_tokens(units: list[dict], vocab: dict[str, int]) -> tuple[list[int], list[tuple[int, int] | None]]:
    tokens: list[int] = []
    groups: list[tuple[int, int] | None] = []
    for u in units:
        start = len(tokens)
        for ph in u.get("ph") or []:
            ph = JA_PH_FIX.get(ph, ph)
            if ph in vocab:
                tokens.append(vocab[ph])
            elif ph not in ("<AP>", "<SP>"):
                continue
        groups.append((start, len(tokens) - 1) if len(tokens) > start else None)
    return tokens, groups


def eval_ja(pool: Path, limit: int | None) -> dict:
    import numpy as np
    import onnxruntime as ort

    from autosync.alignment.ctc import forced_align

    vocab = json.loads(JA_VOCAB.read_text(encoding="utf-8"))
    records = json.loads((pool / "manifest.json").read_text())["records"]
    segs = [r for r in records if r["language"] == "ja"]
    if limit:
        segs = segs[:limit]

    def sess(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.log_severity_level = 3
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    s_fp, s_i8 = sess(JA_FP32), sess(JA_INT8)
    rows = []
    for rec in segs:
        units = load_units(pool / f"{rec['rel']}.json")
        if not units:
            continue
        tokens, groups = ja_tokens(units, vocab)
        if not tokens:
            rows.append({"rel": rec["rel"], "skipped": "no in-vocab phonemes"})
            continue
        keep = [(u, g) for u, g in zip(units, groups) if g is not None]
        if not keep:
            rows.append({"rel": rec["rel"], "skipped": "no mapped chars"})
            continue
        units_k = [u for u, _ in keep]
        groups_k = [g for _, g in keep]
        audio = read_wav_mono16k(pool / f"{rec['rel']}.wav")
        x = audio[None, :]
        lp_fp = s_fp.run(None, {"input_values": x})[1][0]
        lp_i8 = s_i8.run(None, {"input_values": x})[1][0]
        a_fp = forced_align(lp_fp, tokens, blank=0)
        a_i8 = forced_align(lp_i8, tokens, blank=0)
        ref_s = np.array([float(u["start_time"]) for u in units_k])

        def spans(a, groups=groups_k):
            out_s, ok = [], []
            for g0, g1 in groups:
                sp = a.spans[g0:g1 + 1]
                good = all(s is not None for s in sp)
                ok.append(good)
                out_s.append(sp[0].start_frame * FRAME_S if good else np.nan)
            return np.array(out_s), np.array(ok)

        ps_fp, ok_fp = spans(a_fp)
        ps_i8, ok_i8 = spans(a_i8)
        both = ok_fp & ok_i8
        rows.append({
            "rel": rec["rel"], "song": rec["song"], "singer": rec["singer"], "n_units": len(units_k),
            "abs_fp32_start_mae_ms": float(np.nanmean(np.abs(ps_fp - ref_s)) * 1000),
            "abs_int8_start_mae_ms": float(np.nanmean(np.abs(ps_i8 - ref_s)) * 1000),
            "fp32_feasible": int(ok_fp.sum()), "int8_feasible": int(ok_i8.sum()),
            "int8_vs_fp32_start_mae_ms": float(np.nanmean(np.abs(ps_i8 - ps_fp)[both]) * 1000) if both.any() else None,
            "n_both": int(both.sum()),
        })

    def agg(key):
        vals = [r[key] for r in rows if key in r and r[key] is not None and r[key] == r[key]]
        if not vals:
            return {"n": 0}
        arr = np.array(vals)
        return {"n": len(vals), "mean": float(arr.mean()), "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)), "max": float(arr.max())}

    return {"language": "ja", "engine": "ctc-hubert-phoneme", "pool": str(pool),
            "n_segments": len(rows), "rows": rows,
            "summary": {k: agg(k) for k in
                        ("abs_fp32_start_mae_ms", "abs_int8_start_mae_ms", "int8_vs_fp32_start_mae_ms")}}


# --------------------------------------------------------------------------- chinese
def eval_zh(pool: Path, limit: int | None) -> dict:
    import numpy as np
    import onnxruntime as ort
    import torch

    sys.path.insert(0, str(REPO / "android_export/zh"))
    import postprocess_reference as pp

    records = json.loads((pool / "manifest.json").read_text())["records"]
    segs = [r for r in records if r["language"] == "zh"]
    if limit:
        segs = segs[:limit]

    def sess(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.log_severity_level = 3
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    s_fp, s_i8 = sess(ZH_FP32), sess(ZH_INT8)

    def run_post(s, audio, chars):
        tn = np.array([len(chars) + 1], dtype=np.int64)
        us_a, us_p = s.run(None, {"input_values": audio[None, :], "token_num": tn})
        a = torch.from_numpy(np.ascontiguousarray(us_a))[0].clone()
        p = torch.from_numpy(np.ascontiguousarray(us_p))[0].clone()
        _, ts = pp.ts_prediction_lfr6_standard(a, p, list(chars))
        return ts

    rows = []
    for rec in segs:
        units = load_units(pool / f"{rec['rel']}.json")
        if not units:
            continue
        chars = [str(u["word"]) for u in units]
        audio = read_wav_mono16k(pool / f"{rec['rel']}.wav")
        ts_fp = run_post(s_fp, audio, chars)
        ts_i8 = run_post(s_i8, audio, chars)
        ref_s = np.array([float(u["start_time"]) for u in units])
        if len(ts_fp) != len(units) or len(ts_i8) != len(units):
            rows.append({"rel": rec["rel"], "skipped": f"ts len fp={len(ts_fp)} i8={len(ts_i8)} chars={len(units)}"})
            continue
        ps_fp = np.array([t[0] for t in ts_fp]) / 1000.0
        ps_i8 = np.array([t[0] for t in ts_i8]) / 1000.0
        rows.append({
            "rel": rec["rel"], "song": rec["song"], "singer": rec["singer"], "n_units": len(units),
            "abs_fp32_start_mae_ms": float(np.abs(ps_fp - ref_s).mean() * 1000),
            "abs_int8_start_mae_ms": float(np.abs(ps_i8 - ref_s).mean() * 1000),
            "int8_vs_fp32_start_mae_ms": float(np.abs(ps_i8 - ps_fp).mean() * 1000),
        })

    def agg(key):
        vals = [r[key] for r in rows if key in r and r[key] is not None and r[key] == r[key]]
        if not vals:
            return {"n": 0}
        arr = np.array(vals)
        return {"n": len(vals), "mean": float(arr.mean()), "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)), "max": float(arr.max())}

    return {"language": "zh", "engine": "monotonic-aligner-cif", "pool": str(pool),
            "n_segments": len(rows), "rows": rows,
            "summary": {k: agg(k) for k in
                        ("abs_fp32_start_mae_ms", "abs_int8_start_mae_ms", "int8_vs_fp32_start_mae_ms")}}


LANG_EVAL = {"en": eval_en, "ja": eval_ja, "zh": eval_zh}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=sorted(LANG_EVAL))
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--int8", type=Path, default=None, help="override INT8 model path")
    args = ap.parse_args()
    if args.int8 is not None:
        globals()[{"en": "EN_INT8", "ja": "JA_INT8", "zh": "ZH_INT8"}[args.language]] = args.int8
    result = LANG_EVAL[args.language](args.pool, args.limit)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    for r in result["rows"]:
        if "skipped" in r:
            print("  SKIP", r["rel"], r["skipped"])
        else:
            print(f"  {r['rel']}: units={r['n_units']} "
                  f"fp32={r['abs_fp32_start_mae_ms']:.1f}ms int8={r['abs_int8_start_mae_ms']:.1f}ms "
                  f"i8-vs-fp32={r['int8_vs_fp32_start_mae_ms']}ms")
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", args.out)
