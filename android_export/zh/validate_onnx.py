#!/usr/bin/env python3
"""One-command validation for the Chinese (`funasr/fa-zh`) Android ONNX export.

    .venv/bin/python android_export/zh/validate_onnx.py

Runs every acceptance item of docs/ANDROID_ONNX_EXPORT_ZH.md §5 against the
delivered ``fa-zh-fp32.onnx`` / ``fa-zh-int8.onnx`` plus the cached PyTorch
reference outputs in ``_work/`` (regenerated automatically, importing funasr,
when the cache is missing), then rewrites:

    test_vectors/vector{1,2,3}.wav + vector{1,2,3}.expected.json
    manifest.json
    REPORT.md

Nothing is written outside android_export/zh/.
"""

from __future__ import annotations

import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE / "_work"
VECTORS = HERE / "test_vectors"
ROOT = HERE.parents[1]
FP32 = HERE / "fa-zh-fp32.onnx"
INT8 = HERE / "fa-zh-int8.onnx"
MANIFEST = HERE / "manifest.json"
REPORT = HERE / "REPORT.md"

sys.path.insert(0, str(HERE))
import export_model as em  # noqa: E402  (no funasr import at module level)
import postprocess_reference as pp  # noqa: E402

PEAK_THRESHOLD = 1.0 - 1e-4
ACCEPTANCE = {
    "frontend_max_abs": 1e-3,
    "tensor_max_abs": 5e-4,
    "tensor_cos": 0.999,
    "ts_exact_rate": 0.95,
    "ts_max_abs_ms": 40,
    "int8_cos_mean": 0.995,
    "int8_cos_min": 0.97,
    "int8_ts_mae_ms": 20,
    "int8_ts_max_abs_ms": 60,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_session(path: Path, threads: int = 1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def run_onnx(sess, audio: np.ndarray, token_num: int):
    x = np.ascontiguousarray(audio, dtype=np.float32)[None, :]
    tn = np.array([int(token_num)], dtype=np.int64)
    return sess.run(None, {"input_values": x, "token_num": tn})


def scalar(x) -> int:
    """int() that also accepts a size-1 numpy array (numpy 2 raises on those)."""
    return int(np.asarray(x).reshape(-1)[0])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(a @ b / (na * nb))


def count_peaks(peaks: np.ndarray) -> int:
    return int((np.asarray(peaks) >= PEAK_THRESHOLD).sum())


def ts_compare(a: list, b: list) -> dict:
    """Compare two [[start_ms, end_ms], ...] lists."""
    out = {"len_a": len(a), "len_b": len(b)}
    n = min(len(a), len(b))
    if n == 0:
        out.update(dict(n=0, exact_pairs=0, max_abs_ms=0, mae_ms=0.0, exact=True))
        return out
    exact = sum(1 for i in range(n) if a[i][0] == b[i][0] and a[i][1] == b[i][1])
    deltas = [abs(float(a[i][j]) - float(b[i][j])) for i in range(n) for j in (0, 1)]
    out.update(
        dict(
            n=n,
            exact_pairs=exact,
            max_abs_ms=int(max(deltas)),
            mae_ms=float(sum(deltas) / len(deltas)),
            exact=(len(a) == len(b) and exact == n),
        )
    )
    return out


def postprocess(us_alphas: np.ndarray, us_peaks: np.ndarray, t3: int, tokens: list):
    a = torch.from_numpy(np.ascontiguousarray(us_alphas))[0, :t3].clone()
    p = torch.from_numpy(np.ascontiguousarray(us_peaks))[0, :t3].clone()
    _, ts = pp.ts_prediction_lfr6_standard(a, p, list(tokens))
    return ts


def _tolist(ts) -> list:
    return [[int(x[0]), int(x[1])] for x in ts]


# --------------------------------------------------------------------------- #
# step 1: loadability + contract
# --------------------------------------------------------------------------- #
# ONNX names IEEE-32 as "float" while manifest.json spells it "float32"; the
# comparison must not fail on spelling alone (the delivered graph is correct).
_DTYPE_ALIASES = {"float": "float32", "double": "float64", "float16": "float16"}


def dtype_label(elem_type: int) -> str:
    import onnx

    name = onnx.TensorProto.DataType.Name(elem_type).lower()
    return _DTYPE_ALIASES.get(name, name)


def inspect_model(path: Path) -> dict:
    import onnx

    out = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    onnx.checker.check_model(str(path))
    out["checker_ok"] = True
    m = onnx.load(str(path), load_external_data=False)
    opsets = {i.domain or "ai.onnx": i.version for i in m.opset_import}
    out["opsets"] = opsets
    out["opset_max"] = max(opsets.values())
    domains = sorted({n.domain for n in m.graph.node if n.domain not in ("", "ai.onnx")})
    out["non_standard_domains"] = domains
    out["has_com_microsoft"] = any("com.microsoft" in (n.domain or "") for n in m.graph.node)
    out["external_data"] = bool(
        any(i.data_location == onnx.TensorProto.EXTERNAL for i in m.graph.initializer)
    )
    total_init = 0
    for i in m.graph.initializer:
        total_init += int(np.prod(i.dims)) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(i.data_type)).itemsize
    out["initializer_bytes"] = total_init
    out["initializers"] = len(m.graph.initializer)
    out["nodes"] = len(m.graph.node)
    out["op_types"] = sorted({n.op_type for n in m.graph.node})

    def io_spec(vs):
        rows = []
        for v in vs:
            t = v.type.tensor_type
            dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or None)
                    for d in t.shape.dim]
            rows.append({"name": v.name, "dtype": dtype_label(t.elem_type), "shape": dims})
        return rows

    out["inputs"] = io_spec(m.graph.input)
    out["outputs"] = io_spec(m.graph.output)
    # quantised operators
    out["quant_ops"] = sorted(
        {n.op_type for n in m.graph.node if "Integer" in n.op_type or "Quantize" in n.op_type}
    )
    out["provider_ok"] = False
    sess = make_session(path)
    out["provider_ok"] = any("CPUExecutionProvider" in getattr(p, "name", str(p)) for p in sess.get_providers())
    del sess
    gc.collect()
    return out


# --------------------------------------------------------------------------- #
# step 2/3: frontend + encoder + FP32/INT8 outputs
# --------------------------------------------------------------------------- #
def find_lfr_tensor(path: Path):
    """Tensor name of the LFR+CMVN features inside a delivered graph."""
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    var = {i.name for i in m.graph.initializer if i.name.endswith("cmvn_var")}
    for n in m.graph.node:
        if n.op_type == "Mul" and any(i in var for i in n.input):
            return n.output[0]
    return None


def run_frontend(audios: dict, refs) -> dict:
    """Compare the frontend baked in the delivered graphs with extract_fbank(dither=0)."""
    out = {"per_model": {}, "per_audio": {}, "real_total_s": 0.0}
    models = {}
    if (WORK / "frontend-only.onnx").exists():
        models["fp32"] = WORK / "frontend-only.onnx"
    tensor = find_lfr_tensor(INT8)
    if tensor:
        try:
            em.extract_submodel(INT8, WORK / "frontend-int8.onnx", [tensor], ["input_values"])
            models["int8"] = WORK / "frontend-int8.onnx"
        except Exception as exc:  # noqa: BLE001
            out["int8_extract_error"] = repr(exc)
    for tag, path in models.items():
        sess = make_session(path)
        worst = 0.0
        per_audio = {}
        for name in audios:
            key = f"audio.{name}"
            if key not in refs:
                continue
            audio = refs[key]
            (feats,) = sess.run(None, {"input_values": audio[None, :]})
            ref_feats = refs[f"feats.{name}"]
            d = float(np.abs(feats.astype(np.float64) - ref_feats.astype(np.float64)).max())
            per_audio[name] = d
            worst = max(worst, d)
        out["per_model"][tag] = {"max_abs": worst, "n_audio": len(per_audio),
                                 "per_audio": per_audio}
        del sess
        gc.collect()
    if "fp32" in out["per_model"]:
        out["per_audio"] = out["per_model"]["fp32"]["per_audio"]
        out["max_abs"] = out["per_model"]["fp32"]["max_abs"]
    else:
        out["max_abs"] = None
    for name in audios:
        if audios[name]["kind"] in ("human speech", "singing", "tts", "test vector"):
            if f"audio.{name}" in refs:
                out["real_total_s"] += len(refs[f"audio.{name}"]) / 16000.0
    out["n_audio"] = len(out.get("per_audio", {}))
    return out


def run_encoder(audios: dict, refs) -> dict:
    out = {"per_audio": {}}
    if not (WORK / "encoder-only.onnx").exists():
        out["skipped"] = "encoder-only.onnx missing"
        return out
    sess = make_session(WORK / "encoder-only.onnx")
    worst = 0.0
    worst_cos = 1.0
    for name in audios:
        key = f"audio.{name}"
        if key not in refs:
            continue
        audio = refs[key]
        (enc,) = sess.run(
            None,
            {"input_values": audio[None, :], "token_num": np.array([8], dtype=np.int64)},
        )
        ref_enc = refs[f"enc.{name}"]
        d = float(np.abs(enc - ref_enc).max())
        c = cosine(enc, ref_enc)
        out["per_audio"][name] = {"max_abs": d, "cos": c}
        worst = max(worst, d)
        worst_cos = min(worst_cos, c)
    out["max_abs"] = worst
    out["min_cos"] = worst_cos
    del sess
    gc.collect()
    return out


def run_cases(onnx_path: Path, cases: list, refs, label: str) -> dict:
    """Run every case through one ONNX file; collect outputs + post-processing."""
    sess = make_session(onnx_path)
    res = {}
    t0 = time.time()
    for c in cases:
        name = c["name"]
        audio = refs[f"audio.{c['audio']}"]
        us_a, us_p = run_onnx(sess, audio, c["token_num"])
        t3_ref = scalar(refs[f"case.{name}.t3_out"])
        entry = {
            "t3_onnx": int(us_a.shape[1]),
            "t3_ref": t3_ref,
            "us_alphas": us_a,
            "us_peaks": us_p,
            "nan": bool(np.isnan(us_a).any() or np.isinf(us_a).any()
                        or np.isnan(us_p).any() or np.isinf(us_p).any()),
            "peak_count": count_peaks(us_p[0, : min(int(us_a.shape[1]), t3_ref)]),
        }
        if c.get("tokens") and int(us_a.shape[1]) >= t3_ref:
            entry["ts"] = postprocess(us_a, us_p, t3_ref, c["tokens"])
        res[name] = entry
    res["_seconds"] = time.time() - t0
    del sess
    gc.collect()
    return res


# --------------------------------------------------------------------------- #
# step 4/5: comparisons
# --------------------------------------------------------------------------- #
def compare_tensors(cases: list, refs, onnx_out: dict) -> dict:
    rows = []
    for c in cases:
        name = c["name"]
        t3 = scalar(refs[f"case.{name}.t3_out"])
        o = onnx_out[name]
        n = min(o["t3_onnx"], t3)
        ra = refs[f"case.{name}.us_alphas"][0, :n]
        rp = refs[f"case.{name}.us_peaks"][0, :n]
        oa = o["us_alphas"][0, :n]
        op = o["us_peaks"][0, :n]
        rows.append(
            {
                "name": name,
                "kind": c["kind"],
                "t3_match": o["t3_onnx"] == t3,
                "alpha_max_abs": float(np.abs(oa - ra).max()),
                "peak_max_abs": float(np.abs(op - rp).max()),
                "alpha_cos": cosine(oa, ra),
                "peak_cos": cosine(op, rp),
                "peak_count_onnx": o["peak_count"],
                "peak_count_ref": count_peaks(refs[f"case.{name}.us_peaks"][0, :t3]),
            }
        )
    return rows


def compare_timestamps(cases: list, refs, onnx_out: dict, ref_key: str = "official_ts") -> dict:
    rows = []
    for c in cases:
        name = c["name"]
        ref_ts_key = f"case.{name}.{ref_key}"
        if not c.get("tokens") or ref_ts_key not in refs:
            continue
        if "ts" not in onnx_out[name]:
            continue
        ref_ts = _tolist(refs[ref_ts_key])
        cmp = ts_compare(onnx_out[name]["ts"], ref_ts)
        t3 = scalar(refs[f"case.{name}.t3_out"])
        cmp.update(
            {
                "name": name,
                "kind": c["kind"],
                "n_tokens": len(c["tokens"]),
                "peak_count": onnx_out[name]["peak_count"],
                "peak_ref": count_peaks(refs[f"case.{name}.us_peaks"][0, :t3]),
                "peak_expected": len(c["tokens"]) + 1,
            }
        )
        rows.append(cmp)
    return aggregate_ts(rows)


def aggregate_ts(rows: list) -> dict:
    agg = {"rows": rows, "n_cases": len(rows)}
    if not rows:
        return agg
    total_pairs = sum(r["n"] for r in rows)
    exact_pairs = sum(r["exact_pairs"] for r in rows)
    agg["n_tokens_compared"] = total_pairs
    agg["exact_pairs"] = exact_pairs
    agg["exact_rate"] = exact_pairs / total_pairs if total_pairs else 0.0
    agg["max_abs_ms"] = max(r["max_abs_ms"] for r in rows)
    agg["mae_ms"] = float(np.mean([r["mae_ms"] for r in rows]))
    agg["length_mismatch_rows"] = [r["name"] for r in rows if r["len_a"] != r["len_b"]]
    agg["peak_mismatch_rows"] = [
        r["name"] for r in rows if r["peak_count"] != r["peak_ref"]
    ]
    agg["peak_expected_mismatch_rows"] = [
        r["name"] for r in rows if r["peak_count"] != r["peak_expected"]
    ]
    return agg


def compare_int8(cases: list, refs, fp32_out: dict, int8_out: dict) -> dict:
    rows = []
    for c in cases:
        name = c["name"]
        t3 = scalar(refs[f"case.{name}.t3_out"])
        n = min(fp32_out[name]["t3_onnx"], int8_out[name]["t3_onnx"], t3)
        fa, fp = fp32_out[name]["us_alphas"][0, :n], fp32_out[name]["us_peaks"][0, :n]
        qa, qp = int8_out[name]["us_alphas"][0, :n], int8_out[name]["us_peaks"][0, :n]
        row = {
            "name": name,
            "kind": c["kind"],
            "alpha_cos": cosine(qa, fa),
            "peak_cos": cosine(qp, fp),
            "alpha_max_abs": float(np.abs(qa - fa).max()),
            "peak_max_abs": float(np.abs(qp - fp).max()),
            "t3_match": fp32_out[name]["t3_onnx"] == int8_out[name]["t3_onnx"] == t3,
            "peak_count_fp32": fp32_out[name]["peak_count"],
            "peak_count_int8": int8_out[name]["peak_count"],
        }
        if "ts" in fp32_out[name] and "ts" in int8_out[name]:
            cmp = ts_compare(int8_out[name]["ts"], fp32_out[name]["ts"])
            row.update({"ts_mae_ms": cmp["mae_ms"], "ts_max_abs_ms": cmp["max_abs_ms"],
                        "ts_len_equal": cmp["len_a"] == cmp["len_b"]})
        rows.append(row)
    cos_alpha = [r["alpha_cos"] for r in rows]
    cos_peak = [r["peak_cos"] for r in rows]
    ts_rows = [r for r in rows if "ts_mae_ms" in r]
    agg = {
        "rows": rows,
        "n_cases": len(rows),
        "cos_alpha_mean": float(np.mean(cos_alpha)),
        "cos_alpha_min": float(np.min(cos_alpha)),
        "cos_peak_mean": float(np.mean(cos_peak)),
        "cos_peak_min": float(np.min(cos_peak)),
        "alpha_max_abs": max(r["alpha_max_abs"] for r in rows),
        "peak_max_abs": max(r["peak_max_abs"] for r in rows),
        "peak_count_mismatch_rows": [r["name"] for r in rows
                                     if r["peak_count_fp32"] != r["peak_count_int8"]],
        "ts_mae_ms": float(np.mean([r["ts_mae_ms"] for r in ts_rows])) if ts_rows else None,
        "ts_max_abs_ms": max(r["ts_max_abs_ms"] for r in ts_rows) if ts_rows else None,
        "ts_rows": len(ts_rows),
        "ts_len_mismatch_rows": [r["name"] for r in ts_rows if not r["ts_len_equal"]],
        "t3_mismatch_rows": [r["name"] for r in rows if not r["t3_match"]],
    }
    return agg


# --------------------------------------------------------------------------- #
# step 6/7: robustness, determinism, performance
# --------------------------------------------------------------------------- #
def robustness(cases: list, refs, int8_path: Path, fp32_out: dict, int8_out: dict) -> dict:
    out = {}
    silence_cases = [c for c in cases if c["audio"] == "silence4k"]
    out["silence_nan"] = {c["name"]: bool(int8_out[c["name"]]["nan"] or fp32_out[c["name"]]["nan"])
                          for c in silence_cases}
    out["short400"] = {c["name"]: {"shape": int8_out[c["name"]]["t3_onnx"], "nan": int8_out[c["name"]]["nan"]}
                       for c in cases if c["audio"] == "short400"}
    out["long30"] = {c["name"]: {"shape": int8_out[c["name"]]["t3_onnx"], "nan": int8_out[c["name"]]["nan"]}
                     for c in cases if c["audio"] == "long30"}
    out["token_num_boundary"] = {
        c["name"]: {"token_num": c["token_num"], "peak_count_int8": int8_out[c["name"]]["peak_count"],
                    "peak_count_fp32": fp32_out[c["name"]]["peak_count"],
                    "nan": int8_out[c["name"]]["nan"]}
        for c in cases if "_tn" in c["name"]
    }
    # bit-identical repeat
    sess = make_session(int8_path)
    c = next(c for c in cases if c["kind"] == "test vector")
    audio = refs[f"audio.{c['audio']}"]
    runs = [run_onnx(sess, audio, c["token_num"]) for _ in range(3)]
    out["int8_bit_identical_3x"] = all(
        np.array_equal(runs[0][j], run[j]) for run in runs[1:] for j in (0, 1)
    )
    out["int8_repeat_case"] = c["name"]
    del sess
    gc.collect()
    return out


PERF_SNIPPET = r"""
import json, resource, sys, time
import numpy as np, onnxruntime as ort, soundfile as sf
path, wav_path, window_s, runs = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
x, sr = sf.read(wav_path, dtype="float32")
buf = np.zeros((1, int(window_s * sr)), dtype=np.float32)
buf[0, : len(x)] = x[: buf.shape[1]]
tn = np.array([16], dtype=np.int64)
ts = []
for _ in range(runs):
    t = time.perf_counter()
    sess.run(None, {"input_values": buf, "token_num": tn})
    ts.append((time.perf_counter() - t) * 1000.0)
ts = np.array(ts)
print(json.dumps({"window_s": window_s, "runs": runs, "median_ms": float(np.median(ts)),
                  "p90_ms": float(np.percentile(ts, 90)), "min_ms": float(ts.min()),
                  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}))
"""


def performance() -> dict:
    out = {}
    wav = str(ROOT / "experiments" / "audio" / "fleurs_zh_sample.wav")
    for tag, path in (("fp32", FP32), ("int8", INT8)):
        r = subprocess.run(
            [sys.executable, "-c", PERF_SNIPPET, str(path), wav, "12.0", "10"],
            capture_output=True, text=True, check=True,
        )
        out[tag] = json.loads(r.stdout.strip().splitlines()[-1])
    return out


# --------------------------------------------------------------------------- #
# deliverables
# --------------------------------------------------------------------------- #
def write_vectors(cases: list, audios: dict, refs, fp32_out: dict, int8_out: dict) -> list:
    import soundfile as sf

    info = []
    for c in cases:
        if not c.get("vector"):
            continue
        name = c["name"]
        audio = refs[f"audio.{c['audio']}"]
        wav = VECTORS / f"{name}.wav"
        sf.write(str(wav), audio, 16000, subtype="PCM_16")
        vec = {
            "text_raw": c["text_raw"],
            "text_normalized": c["text_norm"],
            "tokens": list(c["tokens"]),
            "token_num": int(c["token_num"]),
            "int8": {
                "us_alphas_shape": list(int8_out[name]["us_alphas"].shape),
                "us_peaks_shape": list(int8_out[name]["us_peaks"].shape),
                "us_alphas_mean": float(int8_out[name]["us_alphas"].mean()),
                "us_alphas_std": float(int8_out[name]["us_alphas"].std()),
                "timestamps_ms": _tolist(int8_out[name].get("ts", [])),
                "peak_count": int(int8_out[name]["peak_count"]),
            },
            "fp32": {
                "us_alphas_shape": list(fp32_out[name]["us_alphas"].shape),
                "us_peaks_shape": list(fp32_out[name]["us_peaks"].shape),
                "us_alphas_mean": float(fp32_out[name]["us_alphas"].mean()),
                "us_alphas_std": float(fp32_out[name]["us_alphas"].std()),
                "timestamps_ms": _tolist(fp32_out[name].get("ts", [])),
                "peak_count": int(fp32_out[name]["peak_count"]),
            },
            "frontend_features_sha256": "",
            "encoder_shape": list(refs[f"enc.{c['audio']}"].shape),
            "source": c["source"],
            "source_path": audios[c["audio"]]["source"],
            "duration_s": round(len(audio) / 16000.0, 3),
            "notes": (
                "timestamps_ms = postprocess_reference.ts_prediction_lfr6_standard on the "
                "FP32 graph; INT8 uses the same post-processing. frontend features are baked "
                "into the graph (plan A) so no separate feature hash is shipped. "
                "Text is from repository-local FLEURS/TTS material."
            ),
        }
        (VECTORS / f"{name}.expected.json").write_text(
            json.dumps(vec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        info.append({"name": name, "wav": str(wav), "bytes": wav.stat().st_size,
                     "sha256": sha256_file(wav), "json": str(VECTORS / f"{name}.expected.json"),
                     "duration_s": vec["duration_s"]})
    return info


def write_manifest(insp: dict, perf: dict, frame_table: list) -> dict:
    manifest = {
        "schema": "melox.autosync.onnx.1",
        "language": "zh",
        "source_model": "funasr/fa-zh",
        "source_model_alias": "iic/speech_timestamp_prediction-v1-16k-offline",
        "license": "FunASR Model License (verify before redistribution)",
        "license_note": (
            "funasr/fa-zh is NOT Apache-2.0. The FunASR Model License must be re-checked "
            "before the exported weights are redistributed or shipped inside the app. "
            "Weights are not committed to git."
        ),
        "sample_rate": 16000,
        "frontend": {
            "baked_into_graph": True,
            "implementation": (
                "kaldi-fbank(hamming,80mel,25ms/10ms,snip_edges,remove_dc_offset,"
                "preemph 0.97,power,log) + LFR(7,6) + CMVN(am.mvn) in the graph; "
                "real FFT realised as a folded DFT matmul (aten::fft_rfft is not "
                "exportable at opset 17)"
            ),
            "dither": 0.0,
            "feature_dim": 560,
        },
        "frame_hop_samples": 160,
        "encoder_frame_seconds": 0.06,
        "upsample_rate": 3,
        "timestamp_frame_seconds": 0.02,
        "input": {"name": "input_values", "shape": [1, None], "dtype": "float32",
                  "preprocessing": "raw 16k mono PCM in [-1,1]; fbank+LFR+CMVN baked into the graph"},
        "aux_inputs": [
            {"name": "token_num", "shape": [1], "dtype": "int64",
             "note": "text token count + 1 (predictor_bias=1)"}
        ],
        "outputs": [
            {"name": "us_alphas", "shape": [1, None], "dtype": "float32", "note": "T3 = 3 * encoder frames"},
            {"name": "us_peaks", "shape": [1, None], "dtype": "float32"},
        ],
        "blank_id": 0,
        "tokens_file": "tokens.json",
        "tokens_count": 8404,
        "files": {
            "fp32": {"name": FP32.name, "sha256": insp["fp32"]["sha256"], "bytes": insp["fp32"]["bytes"]},
            "int8": {"name": INT8.name, "sha256": insp["int8"]["sha256"], "bytes": insp["int8"]["bytes"]},
        },
        "frames": {
            "fbank_frames_formula": "T_fbank = 1 + floor((N - 400) / 160), N >= 400, snip_edges=True",
            "lfr_frames_formula": "T_enc = ceil(T_fbank / 6)",
            "t3_formula": "T3 = 3 * T_enc",
            "measured": frame_table,
        },
        "quantization": {
            "method": "onnxruntime.quantization.quantize_dynamic, weight_type=QInt8, "
                      "op_types_to_quantize=['MatMul','Gemm'], per_channel=False",
            "excluded": "frontend DFT matmuls (constant weights) and every non-MatMul/Gemm op "
                        "(conv, LSTM, LayerNorm) stay fp32",
        },
        "int8_performance_desktop": {
            "window_s": perf["int8"]["window_s"],
            "median_ms": round(perf["int8"]["median_ms"], 1),
            "p90_ms": round(perf["int8"]["p90_ms"], 1),
            "peak_rss_mb": round(perf["int8"]["peak_rss_mb"], 1),
        },
        "postprocess": {
            "reference": "postprocess_reference.py",
            "TIME_RATE": 0.02,
            "force_time_shift": -1.5,
            "START_END_THRESHOLD": 5,
            "MAX_TOKEN_DURATION": 12,
            "peak_threshold": 1.0 - 1e-4,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def frame_table(cases: list, refs, audios: dict) -> list:
    rows = []
    seen = set()
    for c in cases:
        a = c["audio"]
        if a in seen:
            continue
        seen.add(a)
        n = len(refs[f"audio.{a}"])
        t_fbank = scalar(refs[f"feats_len.{a}"])
        t_enc = scalar(refs[f"enc_len.{a}"])
        rows.append({"n_samples": n, "seconds": round(n / 16000.0, 3),
                     "t_fbank": t_fbank, "t_enc": t_enc, "t3": t_enc * 3})
    rows.sort(key=lambda r: r["n_samples"])
    return rows


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def fmt(x, nd=6):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def write_report(M: dict, manifest: dict, insp: dict, perf: dict, vec_info: list):
    fp = M["frontend"]
    fp32t = M["fp32_tensors"]
    fp32ts = M["fp32_ts"]
    int8 = M["int8"]
    rob = M["robustness"]
    songs = [r for r in int8["rows"] if r["kind"] == "singing"]
    songs_zh = [r for r in songs if not r["name"].startswith("songC")]

    def passfail(ok):
        return "PASS" if ok else "**FAIL**"

    a = ACCEPTANCE
    fp_fail = [r["name"] for r in fp32t if not (
        r["alpha_max_abs"] <= a["tensor_max_abs"] and r["peak_max_abs"] <= a["tensor_max_abs"]
        and r["alpha_cos"] >= a["tensor_cos"] and r["peak_cos"] >= a["tensor_cos"]
        and r["t3_match"])]
    ts_fail = [r["name"] for r in fp32ts["rows"] if not (
        r["max_abs_ms"] <= a["ts_max_abs_ms"] and r["len_a"] == r["len_b"])]
    int8_cos_fail = [r["name"] for r in int8["rows"] if not (
        r["alpha_cos"] >= a["int8_cos_min"] and r["peak_cos"] >= a["int8_cos_min"])]
    int8_ts_fail = [r["name"] for r in int8["rows"] if "ts_max_abs_ms" in r and not (
        r["ts_max_abs_ms"] <= a["int8_ts_max_abs_ms"] and r["ts_len_equal"])]
    checks = {
        "frontend": all(pm["max_abs"] <= a["frontend_max_abs"] for pm in fp["per_model"].values())
        and fp["real_total_s"] >= 30,
        "fp32_tensors": not fp_fail,
        "fp32_ts_exact": fp32ts["exact_rate"] >= a["ts_exact_rate"],
        "fp32_ts_max": fp32ts["max_abs_ms"] <= a["ts_max_abs_ms"],
        "fp32_peaks": len(fp32ts["peak_mismatch_rows"]) == 0,
        "int8_cos_mean": int8["cos_alpha_mean"] >= a["int8_cos_mean"]
        and int8["cos_peak_mean"] >= a["int8_cos_mean"],
        "int8_cos_min": int8["cos_alpha_min"] >= a["int8_cos_min"]
        and int8["cos_peak_min"] >= a["int8_cos_min"],
        "int8_ts_mae": (int8["ts_mae_ms"] or 0) <= a["int8_ts_mae_ms"],
        "int8_ts_max": (int8["ts_max_abs_ms"] or 0) <= a["int8_ts_max_abs_ms"],
        "int8_peaks": not int8["peak_count_mismatch_rows"],
        "int8_lines": len(songs) >= 20 and len({r["name"].split("_")[0] for r in songs}) >= 3,
        "robust_silence": not any(rob["silence_nan"].values()),
        "robust_repeat": rob["int8_bit_identical_3x"],
        "robust_short_long": all(v["nan"] is False for v in
                                 list(rob["short400"].values()) + list(rob["long30"].values())),
        "robust_token_num": all(v["nan"] is False for v in rob["token_num_boundary"].values()),
        "loadable": insp["fp32"]["checker_ok"] and insp["int8"]["checker_ok"]
        and insp["fp32"]["provider_ok"] and insp["int8"]["provider_ok"],
        "manifest_io": manifest_io_ok(manifest, insp),
        "opset": insp["fp32"]["opset_max"] <= 17 and insp["int8"]["opset_max"] <= 17,
        "no_ms_ops": not insp["fp32"]["has_com_microsoft"] and not insp["int8"]["has_com_microsoft"],
        "single_file": not insp["fp32"]["external_data"] and not insp["int8"]["external_data"],
    }
    failed = [k for k, v in checks.items() if not v]

    dither = M["dither"]
    lines = []
    A = lines.append
    A("# AutoSync Android ONNX 导出报告（中文 / fa-zh）")
    A("")
    A("Generated by `android_export/zh/validate_onnx.py` "
      f"({time.strftime('%Y-%m-%d %H:%M:%S')} local, host `{M['host']}`).")
    A("")
    A("## 1. 结论（是否可交付 / 未达标项）")
    A("")
    if not failed:
        A("- **全部验收项通过**（§5.1–§5.7）。")
    else:
        A(f"- **未达标项：{len(failed)}** — " + ", ".join(f"`{x}`" for x in failed))
        for k in failed:
            A(f"  - `{k}`: `{checks[k]}`")
    A(f"- 交付物：`fa-zh-fp32.onnx`（{insp['fp32']['bytes'] / 1e6:.1f} MB）、"
      f"`fa-zh-int8.onnx`（{insp['int8']['bytes'] / 1e6:.1f} MB），"
      "单文件、opset 17、无 `com.microsoft` 算子。")
    A("- 许可：**FunASR Model License**（非 Apache-2.0）。导出物随 App 分发前必须复核，"
      "权重不得提交 Git。")
    A(f"- 契约偏差：见 §7（前端的真实 FFT 用 matmul-DFT 实现；`us_peaks` 用闭式 "
      f"`cumsum/floor` 等价式实现，逐值 max|Δ| = {fmt(M['peaks_closed_form']['max_abs'], 3)}，"
      f"峰值数不一致样本 {M['peaks_closed_form']['n_mismatch']}）。")
    A("")
    A("## 2. 交付文件（路径 / 字节 / sha256）")
    A("")
    A("| 文件 | 字节 | sha256 |")
    A("|---|---|---|")
    for tag in ("fp32", "int8"):
        A(f"| `{insp[tag]['path']}` | {insp[tag]['bytes']} | `{insp[tag]['sha256']}` |")
    A(f"| `{MANIFEST}` | {MANIFEST.stat().st_size} | `{sha256_file(MANIFEST)}` |")
    A(f"| `{REPORT}` | (written after this line) | — |")
    A(f"| `postprocess_reference.py` | {(HERE / 'postprocess_reference.py').stat().st_size} | "
      f"`{sha256_file(HERE / 'postprocess_reference.py')}` |")
    A(f"| `export_model.py` | {(HERE / 'export_model.py').stat().st_size} | "
      f"`{sha256_file(HERE / 'export_model.py')}` |")
    A(f"| `validate_onnx.py` | {(HERE / 'validate_onnx.py').stat().st_size} | "
      f"`{sha256_file(HERE / 'validate_onnx.py')}` |")
    A("")
    A("测试向量（设备端对拍用；`test_vectors/` 已在 .gitignore 中排除）：")
    A("")
    A("| 向量 | 时长 | 字节 | sha256 |")
    A("|---|---|---|---|")
    for v in vec_info:
        A(f"| `{v['name']}.wav` | {v['duration_s']} s | {v['bytes']} | `{v['sha256']}` |")
    A("")
    A("## 3. 契约核对（I/O 名、形状、opset、算子、external data、前端等价性、dither 固定证据）")
    A("")
    A("| 项 | fp32 | int8 |")
    A("|---|---|---|")
    A(f"| checker | {insp['fp32']['checker_ok']} | {insp['int8']['checker_ok']} |")
    A(f"| CPUExecutionProvider | {insp['fp32']['provider_ok']} | {insp['int8']['provider_ok']} |")
    A(f"| opset | {insp['fp32']['opset_max']} | {insp['int8']['opset_max']} |")
    A(f"| com.microsoft 算子 | {insp['fp32']['has_com_microsoft']} | {insp['int8']['has_com_microsoft']} |")
    A(f"| external data | {insp['fp32']['external_data']} | {insp['int8']['external_data']} |")
    A(f"| 节点数 | {insp['fp32']['nodes']} | {insp['int8']['nodes']} |")
    A(f"| initializer 字节 | {insp['fp32']['initializer_bytes']} | {insp['int8']['initializer_bytes']} |")
    A("")
    A("I/O（与 manifest.json 一致）：")
    A("")
    A("```")
    for tag in ("fp32", "int8"):
        A(f"{tag} inputs : {insp[tag]['inputs']}")
        A(f"{tag} outputs: {insp[tag]['outputs']}")
    A("```")
    A("")
    A(f"- int8 量化算子：{insp['int8']['quant_ops']}")
    A(f"- 帧数实测（N → T_fbank → T_enc → T3）：")
    A("")
    A("| N | 秒 | T_fbank | T_enc | T3 |")
    A("|---|---|---|---|---|")
    for r in manifest["frames"]["measured"][:8]:
        A(f"| {r['n_samples']} | {r['seconds']} | {r['t_fbank']} | {r['t_enc']} | {r['t3']} |")
    A(f"| … | | | | |")
    A("")
    A("### 3.1 前端等价性（ONNX 图内前端 vs `extract_fbank(dither=0)`）")
    A("")
    A(f"- 比对音频段数 {fp['n_audio']}，其中真实语音/歌声/向量共 {fp['real_total_s']:.1f} s；"
      f"逐段独立跑。")
    for tag, pm in fp["per_model"].items():
        A(f"- {tag} 图内前端：全局 `max|Δ| = {fmt(pm['max_abs'])}`"
          f"（目标 ≤ {a['frontend_max_abs']:g}）：{passfail(pm['max_abs'] <= a['frontend_max_abs'])}")
    if "int8_extract_error" in fp:
        A(f"- int8 前端抽取失败：{fp['int8_extract_error']}")
    worst_audio = sorted(fp["per_audio"].items(), key=lambda kv: -kv[1])[:5]
    A(f"- fp32 最差的 5 段： " + ", ".join(f"`{k}`={fmt(v)}" for k, v in worst_audio))
    A("")
    A("### 3.2 dither 固定证据")
    A("")
    A(f"- Python `dither=1.0`，同段音频 3 次 `extract_fbank`：两两 `max|Δ|` = "
      f"{[fmt(x, 3) for x in dither['fbank_dither1_pairwise_max_abs_diff']]}（特征本身随机抖动）")
    A(f"- Python `dither=0`，同段音频 3 次 `extract_fbank`：`max|Δ| = "
      f"{fmt(dither['fbank_dither0_max_abs_diff'], 3)}`（可复现）")
    A(f"- Python `dither=0`（基线快照已固定为 0），5 次 `AutoModel.generate` 时间戳 `max|Δ| = "
      f"{dither['generate_dither0_ts_max_abs_diff_ms']} ms`")
    A(f"- **FunASR 行为发现**：`AutoModel.generate` 每次调用都会执行 "
      f"`_reset_runtime_configs()`，用构造时的 `copy.deepcopy` 基线覆盖 `auto.kwargs`，"
      f"因此运行时设置 `frontend.dither = 0` 会被丢弃。直接跑 `generate`（引擎现状）时前端仍是 "
      f"配置默认 `dither=1.0`：5 次时间戳两两 `max|Δ| = "
      f"{dither['generate_dither1_ts_pairwise_max_abs_diff_ms']} ms`（非确定性）；"
      f"与固定 dither=0 对比 `max|Δ| = {dither['generate_dither1_vs_dither0_ts_max_abs_diff_ms']} ms`。"
      f"导出脚本通过改写 `auto._base_kwargs_map` 里的前端快照把整条链路固定为 dither=0。")
    A(f"- ONNX（dither 恒为 0）vs Python dither=0（端到端时间戳）：见 §4.2。")
    A("")
    A("## 4. 验证数字（按第 5 节 1–7 逐项列表，含素材来源与样本数）")
    A("")
    A("### 4.1 编码器 + 预测器正确性（FP32 vs PyTorch）")
    A("")
    A(f"- 样本：{len(fp32t)} 个 case（含真实语音、歌声行、静音/极短/长输入、token_num 边界）。")
    A(f"- `us_alphas`/`us_peaks` 逐值 `max|Δ|`：最差 {fmt(max(max(r['alpha_max_abs'], r['peak_max_abs']) for r in fp32t))}"
      f"（目标 ≤ {a['tensor_max_abs']:g}）→ {passfail(not fp_fail)}")
    A(f"- 余弦相似度最低：`us_alphas` {fmt(min(r['alpha_cos'] for r in fp32t), 8)}，"
      f"`us_peaks` {fmt(min(r['peak_cos'] for r in fp32t), 8)}（目标 ≥ {a['tensor_cos']}）")
    A(f"- T3 长度不一致样本：{len([r for r in fp32t if not r['t3_match']])}")
    if fp_fail:
        A(f"- 未达标样本：{fp_fail}")
    A("")
    A("### 4.2 端到端时间戳（FP32 ONNX + `postprocess_reference.py` vs Python 参考）")
    A("")
    A(f"- 参与比对：{fp32ts['n_cases']} 行 / {fp32ts['n_tokens_compared']} tokens。")
    A(f"- 完全一致（start 与 end 均 0 ms 差）：{fp32ts['exact_pairs']}/{fp32ts['n_tokens_compared']} = "
      f"{fp32ts['exact_rate'] * 100:.2f}%（目标 ≥ {a['ts_exact_rate'] * 100:g}%）→ "
      f"{passfail(fp32ts['exact_rate'] >= a['ts_exact_rate'])}")
    A(f"- 逐 token `max|Δ| = {fp32ts['max_abs_ms']} ms`（目标 ≤ {a['ts_max_abs_ms']}）→ "
      f"{passfail(fp32ts['max_abs_ms'] <= a['ts_max_abs_ms'])}")
    A(f"- token 数不一致的行：{fp32ts['length_mismatch_rows'] or '无'}")
    A(f"- 峰值数与 Python 不一致的样本：{fp32ts['peak_mismatch_rows'] or '无'}"
      f"（与 `len(tokens)+1` 不一致：{fp32ts['peak_expected_mismatch_rows'] or '无'}）")
    A(f"- `postprocess_reference.py` vs 官方 `ts_prediction_lfr6_standard`：逐 token "
      f"`max|Δ| = {M['postprocess_check']['max_abs_ms']} ms`（"
      f"{M['postprocess_check']['n_cases']} 行 / {M['postprocess_check']['n_tokens']} tokens）")
    A(f"- Python(alphas/peaks + §2.4) vs `AutoModel.generate`（引擎路径）时间戳：共同行 "
      f"{M['engine_vs_offline']['n_rows']}，完全一致 {M['engine_vs_offline']['exact_pairs']}/"
      f"{M['engine_vs_offline']['n_pairs']}，`max|Δ| = {M['engine_vs_offline']['max_abs_ms']} ms`")
    A("")
    A("### 4.3 INT8 精度（INT8 vs FP32）")
    A("")
    A(f"- 全部 {int8['n_cases']} 个 case，其中真实歌词行（歌声片段）{len(songs)} 行，"
      f"覆盖 {len({r['name'].split('_')[0] for r in songs})} 段素材"
      f"（其中纯中文歌曲 {len(songs_zh)} 行）；素材来源：本机 `Test/` 第三方歌曲 + "
      f"`experiments/audio/` FLEURS/TTS（歌词仅作输入，未写入任何交付文件）。")
    A(f"- 逐帧余弦：`us_alphas` mean {fmt(int8['cos_alpha_mean'], 8)} / min {fmt(int8['cos_alpha_min'], 8)}；"
      f"`us_peaks` mean {fmt(int8['cos_peak_mean'], 8)} / min {fmt(int8['cos_peak_min'], 8)}"
      f"（目标 mean ≥ {a['int8_cos_mean']}，min ≥ {a['int8_cos_min']}）→ "
      f"{passfail(checks['int8_cos_mean'] and checks['int8_cos_min'])}")
    A(f"- 逐值 `max|Δ|`：`us_alphas` {fmt(int8['alpha_max_abs'], 4)}，`us_peaks` {fmt(int8['peak_max_abs'], 4)}")
    A(f"- 端到端时间戳（{int8['ts_rows']} 行）：MAE {fmt(int8['ts_mae_ms'], 4)} ms，"
      f"`max|Δ| = {fmt(int8['ts_max_abs_ms'], 4)} ms`（目标 MAE ≤ {a['int8_ts_mae_ms']}，"
      f"max ≤ {a['int8_ts_max_abs_ms']}）→ "
      f"{passfail(checks['int8_ts_mae'] and checks['int8_ts_max'])}")
    A(f"- token 数不一致行：{int8['ts_len_mismatch_rows'] or '无'}；"
      f"峰值数不一致行：{int8['peak_count_mismatch_rows'] or '无'} → {passfail(checks['int8_peaks'])}")
    if int8_ts_fail:
        A(f"- 超差行（仍列出以便分析）：{int8_ts_fail[:10]}")
    A("")
    A("### 4.4 健壮性")
    A("")
    A(f"- 全 0 静音输入：NaN/Inf = {rob['silence_nan']}")
    A(f"- INT8 同一输入连跑 3 次逐位一致：{rob['int8_bit_identical_3x']}（case `{rob['int8_repeat_case']}`）")
    A(f"- 400 采样输入：{rob['short400']}")
    A(f"- ≥30 s 输入：{rob['long30']}")
    A("- `token_num` 边界（1/50/200 及相邻值）：")
    A("")
    A("| case | token_num | peaks fp32 | peaks int8 | NaN/Inf |")
    A("|---|---|---|---|---|")
    for k, v in rob["token_num_boundary"].items():
        A(f"| `{k}` | {v['token_num']} | {v['peak_count_fp32']} | {v['peak_count_int8']} | {v['nan']} |")
    A("")
    A("### 4.5 性能与内存（桌面参考值，单线程 CPU）")
    A("")
    A("| 模型 | 12 s 窗口中位数 | P90 | 峰值 RSS | 文件字节 | sha256 |")
    A("|---|---|---|---|---|---|")
    for tag in ("fp32", "int8"):
        p = perf[tag]
        A(f"| {tag} | {p['median_ms']:.0f} ms | {p['p90_ms']:.0f} ms | {p['peak_rss_mb']:.0f} MB | "
          f"{insp[tag]['bytes']} | `{insp[tag]['sha256'][:16]}…` |")
    A("")
    A(f"（INT8/FP32 体积比 {insp['int8']['bytes'] / insp['fp32']['bytes']:.3f}；"
      f"`model.pt` = 158,469,618 bytes。INT8 ≤ 60 MB 目标："
      f"{passfail(insp['int8']['bytes'] <= 60e6)}）")
    A("")
    A("### 4.6 可复跑")
    A("")
    A("```bash")
    A("cd /home/lladlam/项目/MeloX-AutoSync")
    A(".venv/bin/python android_export/zh/validate_onnx.py")
    A("# 完整重算（含 PyTorch 参考 + 重新导出量化）：")
    A(".venv/bin/python android_export/zh/export_model.py --force-refs")
    A("```")
    A("")
    A("## 5. 一键复跑命令")
    A("")
    A("```bash")
    A("cd /home/lladlam/项目/MeloX-AutoSync && .venv/bin/python android_export/zh/validate_onnx.py")
    A("```")
    A("")
    A("首次（或 `_work/` 被删除时）该命令会自动重建 PyTorch 参考再校验；"
      "`_work/` 存在时约 1 分钟。")
    A("")
    A("## 6. 已知限制与风险")
    A("")
    A("- **许可**：`funasr/fa-zh` 为 FunASR Model License（非 Apache-2.0），"
      "随 App 分发前必须复核；权重/ONNX 不提交 Git。")
    A("- **量化与歌声**：INT8 在真实歌声行上 `max|Δ|` 见 §4.3；"
      "歌声音高/混音与朗读差异大，量化误差高于朗读；KTV 场景建议端上保留 FP32 回退。")
    A("- **`token_num` 敏感**：CIF 上采样头用 `token_num / sum(alphas)` 归一化，"
      "token 数与音频不匹配时会重归一化（§2.4 步骤 2）；INT8 与 FP32 在该路径上的峰值数"
      "差异见 §4.4。")
    A("- **内存**：桌面进程峰值 RSS 见 §4.5（含 ORT runtime 与音频缓冲）；"
      "Android 端需用 ORT-Mobile + 动态 `input_values` 维度，12 s 窗口按实测预留。")
    A("- **桌面 ≠ 设备**：以上全部为 x86 CPU 单线程参考值；ARM 上 INT8 通常更快，"
      "但需在真机上复测 §4.4/§4.5。")
    A("- **前端**：本机无 torchaudio，FunASR 走 kaldi-native-fbank；若日后环境出现 "
      "torchaudio，需按任务书 §2.2 重新核对边界帧。")
    A("")
    A("## 7. 未做/做不了的部分（如实说明）")
    A("")
    A("- **真实 FFT 不可导出**：`aten::fft_rfft` 在 opset 17 无法导出"
      "（`torch.onnx.errors.UnsupportedOperatorError`），改用把 Hamming 窗折进去的 "
      "DFT 矩阵乘（256×400，float32）。与 kaldi-native-fbank 的逐值 "
      f"`max|Δ| = {fmt(fp['max_abs'])}`（§3.1）。")
    A("- **`cif_wo_hidden` 的顺序循环不可动态轨迹**：官方实现对时间轴做 Python 循环，"
      "追踪导出会把 T3 固定死。改为与渐增器完全等价的闭式（`cumsum`/`floor`），"
      f"逐值 `max|Δ| = {fmt(M['peaks_closed_form']['max_abs'], 3)}`，"
      f"峰值数不一致样本 {M['peaks_closed_form']['n_mismatch']}，"
      f"端到端时间戳不一致样本 {len(fp32ts['peak_mismatch_rows'])}。")
    A("- **SANM 编码器 mask 省略**：batch=1 无 padding 时 FunASR 传入的 mask 全为 1，"
      "在图里直接省略（数学上恒等）；因此图不支持 batch>1 或带 padding 的批量输入"
      "（契约本就是 `[1, N]`）。")
    A("- 未在真机（Android/ARM）上跑过；未做 XNNPACK/NNAPI EP 验证；"
      "未评估 ONNX Runtime Mobile 的二进制裁剪。")
    A("- `fleurs_zh_sample2.wav` 本地无转写文本，仅用于前端等价性（不参与时间戳比对）。")
    A("- 测试向量只用了仓库内可用的 FLEURS/TTS 素材；`Test/` 中第三方歌曲仅作评估输入，"
      "歌词文本未写入任何交付文件（见 `_work/cases.json`，该目录不提交）。")
    A("")
    A("## 8. 验收项汇总")
    A("")
    A("| # | 验收项 | 结果 |")
    A("|---|---|---|")
    order = [
        ("frontend", "前端等价性 max|Δ| ≤ 1e-3，≥3 段 ≥30 s 真实音频"),
        ("fp32_tensors", "FP32 vs PyTorch tensor max|Δ| ≤ 5e-4 / cos ≥ 0.999"),
        ("fp32_ts_exact", "端到端时间戳完全一致 ≥95%"),
        ("fp32_ts_max", "端到端 max|Δ| ≤ 40 ms"),
        ("fp32_peaks", "峰值数与 Python 一致（0 例外）"),
        ("int8_cos_mean", "INT8 余弦 mean ≥ 0.995"),
        ("int8_cos_min", "INT8 余弦 min ≥ 0.97"),
        ("int8_ts_mae", "INT8 时间戳 MAE ≤ 20 ms"),
        ("int8_ts_max", "INT8 时间戳 max|Δ| ≤ 60 ms"),
        ("int8_peaks", "INT8 token/峰值数不一致 = 0"),
        ("int8_lines", "≥20 真实歌词行 / ≥3 段素材"),
        ("robust_silence", "静音无 NaN/Inf"),
        ("robust_repeat", "INT8 连跑 3 次逐位一致"),
        ("robust_short_long", "400 采样与 ≥30 s 输入"),
        ("robust_token_num", "token_num 1/50/200"),
        ("loadable", "checker + CPU EP"),
        ("manifest_io", "I/O 名/形状/dtype 与 manifest.json 一致"),
        ("opset", "opset ≤ 17"),
        ("no_ms_ops", "无 com.microsoft 算子"),
        ("single_file", "单文件、无 external data"),
    ]
    for k, desc in order:
        A(f"| {k} | {desc} | {passfail(checks[k])} |")
    A("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return checks, failed


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def ensure_refs():
    if (WORK / "refs.npz").exists() and (WORK / "cases.json").exists():
        return
    print("reference cache missing: running export_model.py (imports funasr, slow) ...")
    em.main(["--no-validate"])


def check_postprocess_official(cases: list, refs) -> dict:
    cache = WORK / "postprocess_check.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    worst, n_cases, n_tokens = pp.check_official(cases, refs)
    out = {"max_abs_ms": worst, "n_cases": n_cases, "n_tokens": n_tokens}
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def engine_vs_offline(cases: list, refs) -> dict:
    n_rows = 0
    n_pairs = 0
    exact = 0
    worst = 0
    for c in cases:
        key = f"case.{c['name']}.generate_ts"
        if key not in refs or not c.get("tokens"):
            continue
        gen = _tolist(refs[key])
        off = _tolist(refs[f"case.{c['name']}.official_ts"])
        cmp = ts_compare(gen, off)
        n_rows += 1
        n_pairs += cmp["n"]
        exact += cmp["exact_pairs"]
        worst = max(worst, cmp["max_abs_ms"])
    return {"n_rows": n_rows, "n_pairs": n_pairs, "exact_pairs": exact, "max_abs_ms": worst}


def manifest_io_ok(manifest: dict, insp: dict) -> bool:
    """The delivered graphs must match manifest.json exactly."""
    ok = True
    for tag in ("fp32", "int8"):
        ins = insp[tag]["inputs"]
        outs = insp[tag]["outputs"]
        ok &= ins[0]["name"] == manifest["input"]["name"] == "input_values"
        ok &= ins[0]["dtype"] == manifest["input"]["dtype"] == "float32"
        ok &= ins[0]["shape"][0] == 1 and ins[0]["shape"][1] is not None
        ok &= ins[1]["name"] == manifest["aux_inputs"][0]["name"] == "token_num"
        ok &= ins[1]["dtype"] == manifest["aux_inputs"][0]["dtype"] == "int64"
        ok &= list(ins[1]["shape"]) == [1]
        ok &= [o["name"] for o in outs] == [o["name"] for o in manifest["outputs"]]
        ok &= all(o["dtype"] == "float32" for o in outs)
        ok &= all(o["shape"][0] == 1 and o["shape"][1] is not None for o in outs)
    return bool(ok)


def main(argv=None):
    torch.set_num_threads(1)
    t0 = time.time()
    ensure_refs()
    doc = json.loads((WORK / "cases.json").read_text(encoding="utf-8"))
    cases, audios = doc["cases"], doc["audios"]
    refs = np.load(WORK / "refs.npz")
    print(f"refs loaded ({len(cases)} cases)")

    insp = {"fp32": inspect_model(FP32), "int8": inspect_model(INT8)}
    print("loadability checked")

    fp32_out = run_cases(FP32, cases, refs, "fp32")
    int8_out = run_cases(INT8, cases, refs, "int8")
    print(f"onnx inference done ({time.time() - t0:.1f}s total)")

    frontend = run_frontend(audios, refs)
    encoder = run_encoder(audios, refs)
    fp32_tensors = compare_tensors(cases, refs, fp32_out)
    fp32_ts = compare_timestamps(cases, refs, fp32_out, "official_ts")
    int8 = compare_int8(cases, refs, fp32_out, int8_out)
    rob = robustness(cases, refs, INT8, fp32_out, int8_out)
    perf = performance()
    post_check = check_postprocess_official(cases, refs)
    evo = engine_vs_offline(cases, refs)
    peaks_cf = pp.check_peaks(cases, refs, verbose=False)
    peaks_check = {"max_abs": peaks_cf[0], "n_mismatch": peaks_cf[1]}
    dither = json.loads((WORK / "dither_evidence.json").read_text(encoding="utf-8"))

    ftab = frame_table(cases, refs, audios)
    manifest = write_manifest(insp, perf, ftab)
    vec_info = write_vectors(cases, audios, refs, fp32_out, int8_out)
    print("manifest.json + test_vectors written")

    import platform

    M = {
        "host": f"{platform.platform()} / python {platform.python_version()}",
        "frontend": frontend,
        "encoder": encoder,
        "fp32_tensors": fp32_tensors,
        "fp32_ts": fp32_ts,
        "int8": int8,
        "robustness": rob,
        "postprocess_check": post_check,
        "engine_vs_offline": evo,
        "peaks_closed_form": peaks_check,
        "dither": dither,
    }
    checks, failed = write_report(M, manifest, insp, perf, vec_info)
    (WORK / "validation.json").write_text(
        json.dumps({"checks": checks, "failed": failed,
                    "int8": {k: v for k, v in int8.items() if k != "rows"},
                    "frontend": frontend, "fp32_ts": {k: v for k, v in fp32_ts.items() if k != "rows"},
                    "peaks_closed_form": peaks_check},
                   indent=2, default=float),
        encoding="utf-8",
    )
    print(f"REPORT.md written ({time.time() - t0:.1f}s total)")
    print("failed checks:", failed or "none")
    return failed


if __name__ == "__main__":
    raise SystemExit(0 if not main() else 1)
