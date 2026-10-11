#!/usr/bin/env python
"""Selective-quantisation evaluation on the real-pop ``Test/`` playlist.

Compares a *candidate* INT8 ONNX against the FP32 ONNX on the same line windows
built from the playlist dataset (`tools/build_playlist_dataset.py`).  It reports
the full distribution of timing differences, not just the mean, so tail errors
cannot hide behind an average.

Runs on CPU only and loads **one** ONNX session per process; the fp32 and int8
phases are separate subprocess invocations that dump logits to ``.npz``, and the
compare phase reloads only that (no model in RAM). This keeps peak RSS low on a
4 GB machine.

Usage::

    # build tasks + fp32 pass (one language)
    .venv/bin/python benchmarks/quant_selective_eval.py --language en --phase fp32 \
        --scratch /tmp/opencode/q_eval_en
    # candidate int8 pass
    .venv/bin/python benchmarks/quant_selective_eval.py --language en --phase int8 \
        --model /tmp/cand.onnx --scratch /tmp/opencode/q_eval_en --tag cand
    # compare
    .venv/bin/python benchmarks/quant_selective_eval.py --language en --phase compare \
        --scratch /tmp/opencode/q_eval_en --tag cand --out /tmp/en_cand.json

Only aggregate metrics are printed/persisted; lyric text never leaves this
process (task tables store token ids / indices, not text).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "android_export/ja"))

SR = 16000
FRAME_S = 0.02
MODELS = {
    "en": {"fp32": REPO / "android_export/en/wav2vec2-base-960h-fp32.onnx"},
    "ja": {"fp32": REPO / "android_export/ja/hubert-ja-phoneme-ctc-fp32.onnx"},
    "zh": {"fp32": REPO / "android_export/zh/fa-zh-fp32.onnx"},
}
VOCAB = {
    "en": (REPO / "models/hf/models--facebook--wav2vec2-base-960h/snapshots/"
           "22aad52d435eb6dbaf354bdad9b0da84ce7d6156/vocab.json"),
}


# --------------------------------------------------------------------------- audio
def read_mono16k(path: Path):
    import numpy as np

    from autosync.audio.io import load_audio

    audio, sr = load_audio(str(path), SR)
    assert sr == SR
    return np.asarray(audio, dtype=np.float32)


# --------------------------------------------------------------------------- tasks
def build_tasks(root: Path, manifest: dict, language: str, splits: list[str],
                max_lines_per_song: int) -> list[dict]:
    from autosync.alignment.segment import _cap_window_end, _window_for_line
    from autosync.lyrics.classify import classify_lines
    from autosync.lyrics.lrc import parse_lrc
    recs = [r for r in manifest["records"]
            if r.get("language") == language and r["split"] in splits and r.get("lyrics")]
    tasks: list[dict] = []
    for rec in recs:
        audio_path = root / rec["audio"]
        lines, _meta = parse_lrc(root / rec["lyrics"])
        if not lines:
            continue
        audio = read_mono16k(audio_path)
        total = len(audio) / SR
        kinds = classify_lines(lines)
        nxt = None
        nxt_singing = [None] * len(lines)
        for i in range(len(lines) - 1, -1, -1):
            nxt_singing[i] = nxt
            if kinds[i].kind.alignable:
                nxt = lines[i]
        picked = 0
        for i, line in enumerate(lines):
            if not kinds[i].kind.alignable:
                continue
            w0, w1 = _window_for_line(line, nxt_singing[i], total, 0.30, 12.0)
            w1 = _cap_window_end(audio, SR, line.start, w1, 12.0, 30.0)
            if not (1.0 <= (w1 - w0) <= 32.0):
                continue
            tasks.append({
                "song": rec["song_id"], "split": rec["split"], "line_index": line.index,
                "w0": round(w0, 3), "w1": round(w1, 3), "text": line.text,
                "audio": rec["audio"],
            })
            picked += 1
            if picked >= max_lines_per_song:
                break
        del audio
    return tasks


def tokenize(language: str, text: str) -> dict | None:
    from autosync.lyrics.normalize import normalize_cjk, normalize_en

    if language == "en":
        vocab = json.loads(VOCAB["en"].read_text(encoding="utf-8"))
        norm = normalize_en(text)
        toks: list[int] = []
        groups: list[tuple[int, int]] = []
        for word in norm.split():
            s = len(toks)
            for ch in word:
                if ch in vocab:
                    toks.append(vocab[ch])
            if len(toks) > s:
                groups.append((s, len(toks) - 1))
        if not toks:
            return None
        return {"tokens": toks, "groups": groups, "units": len(groups)}
    if language == "ja":
        from g2p_reference import reference_map

        r = reference_map(text)
        ids = list(r["phoneme_ids"]) if r["phoneme_ids"] else []
        if not ids or r.get("no_attribution"):
            return None
        groups = []
        for sp in r["char_phoneme_spans"]:
            if sp.get("uncertain"):
                continue
            groups.append((sp["phoneme_index_start"], sp["phoneme_index_end"] - 1))
        groups = [g for g in groups if g[1] >= g[0]]
        if not groups:
            return None
        return {"tokens": ids, "groups": groups, "units": len(groups)}
    if language == "zh":
        chars = [c for c in normalize_cjk(text) if not c.isspace()]
        if not chars:
            return None
        return {"tokens": None, "groups": None, "units": len(chars), "chars": chars}
    raise ValueError(language)


# --------------------------------------------------------------------------- onnx
def make_session(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    so.enable_mem_pattern = False
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run_phase(language: str, phase: str, model: Path | None, scratch: Path, tag: str,
              root: Path, manifest: dict, splits: list[str], max_lines: int, threads: int) -> None:
    import numpy as np

    scratch.mkdir(parents=True, exist_ok=True)
    tasks_path = scratch / "tasks.json"
    if phase == "fp32" or not tasks_path.exists():
        tasks = build_tasks(root, manifest, language, splits, max_lines)
        if phase == "int8" and not tasks_path.exists():
            raise SystemExit("task list missing; run --phase fp32 first")
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
        tasks = [t for t in tasks if tokenize(language, t["text"])]
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    else:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    print(f"[{language}/{phase}] {len(tasks)} line tasks", flush=True)

    if phase == "fp32":
        tag = "fp32"  # fp32 pass is model/tag independent
    onnx_path = model if phase == "int8" else MODELS[language]["fp32"]
    sess = make_session(onnx_path, threads)
    audio_cache = {}
    store: dict[str, np.ndarray] = {}
    t0 = time.time()
    for n, task in enumerate(tasks):
        tok = tokenize(language, task["text"])
        if tok is None:
            continue
        key = f"{task['song']}|{task['line_index']}"
        a = audio_cache.get(task["audio"])
        if a is None:
            a = read_mono16k(root / task["audio"])
            audio_cache = {task["audio"]: a}  # keep a single song resident
        seg = a[round(task["w0"] * SR): round(task["w1"] * SR)]
        if seg.size < 400:
            continue
        feeds = {"input_values": seg[None, :].astype(np.float32)}
        if language == "zh":
            feeds["token_num"] = np.array([tok["units"] + 1], dtype=np.int64)
            ua, up = sess.run(None, feeds)
            store[f"{key}|a"] = ua
            store[f"{key}|p"] = up
        else:
            _, lp = sess.run(None, feeds)
            store[key] = lp
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    out = scratch / f"{phase}_{tag}.npz"
    np.savez(out, **store)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta = {"phase": phase, "model": str(onnx_path), "n": len(store), "seconds": round(time.time() - t0, 1),
            "peak_rss_mb": round(rss, 1), "bytes": Path(onnx_path).stat().st_size}
    (scratch / f"{phase}_{tag}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{language}/{phase}] wrote {out} peak_rss={rss:.0f}MB time={time.time() - t0:.0f}s", flush=True)
    del sess
    return meta


# --------------------------------------------------------------------------- compare
def _spans_ctc(a, groups, frame_s=FRAME_S):
    import numpy as np

    ss, ee, ok = [], [], []
    for g0, g1 in groups:
        sp = a.spans[g0:g1 + 1]
        good = all(x is not None for x in sp)
        ok.append(good)
        ss.append(sp[0].start_frame * frame_s if good else np.nan)
        ee.append((sp[-1].end_frame + 1) * frame_s if good else np.nan)
    return np.array(ss), np.array(ee), np.array(ok)


def compare(language: str, scratch: Path, tag: str, out: Path | None, manifest: dict) -> dict:
    import numpy as np

    from autosync.alignment.ctc import forced_align

    tasks = json.loads((scratch / "tasks.json").read_text(encoding="utf-8"))
    fp = np.load(scratch / "fp32_fp32.npz")
    i8 = np.load(scratch / f"int8_{tag}.npz")
    scene = {r["song_id"]: r.get("scene", {}) for r in manifest["records"]}

    per_line = []
    for task in tasks:
        tok = tokenize(language, task["text"])
        if tok is None:
            per_line.append({"song": task["song"], "line_index": task["line_index"], "skipped": "tokenize"})
            continue
        key = f"{task['song']}|{task['line_index']}"
        if language == "zh":
            have = f"{key}|a" in fp.files and f"{key}|a" in i8.files
        else:
            have = key in fp.files and key in i8.files
        if not have:
            per_line.append({"song": task["song"], "line_index": task["line_index"], "skipped": "no_outputs"})
            continue
        rec = {"song": task["song"], "split": task["split"], "line_index": task["line_index"],
               "n_units": tok["units"], "window_s": round(task["w1"] - task["w0"], 2)}
        if language == "zh":
            import torch

            sys.path.insert(0, str(REPO / "android_export/zh"))
            import postprocess_reference as pp

            def ts(arr_a, arr_p, chars=tok["chars"]):
                a = torch.from_numpy(np.ascontiguousarray(arr_a))[0].clone()
                p = torch.from_numpy(np.ascontiguousarray(arr_p))[0].clone()
                _, t = pp.ts_prediction_lfr6_standard(a, p, list(chars))
                return np.array([x[0] for x in t]) / 1000.0, np.array([x[1] for x in t]) / 1000.0

            s_fp, e_fp = ts(fp[f"{key}|a"], fp[f"{key}|p"])
            s_i8, e_i8 = ts(i8[f"{key}|a"], i8[f"{key}|p"])
            if not (len(s_fp) == len(s_i8) == tok["units"]):
                rec["skipped"] = f"len fp={len(s_fp)} i8={len(s_i8)} units={tok['units']}"
                per_line.append(rec)
                continue
            rec["fp32_feasible"] = rec["int8_feasible"] = tok["units"]
        else:
            a_fp = forced_align(fp[key][0], tok["tokens"], blank=0)
            a_i8 = forced_align(i8[key][0], tok["tokens"], blank=0)
            s_fp, e_fp, ok_fp = _spans_ctc(a_fp, tok["groups"])
            s_i8, e_i8, ok_i8 = _spans_ctc(a_i8, tok["groups"])
            both = ok_fp & ok_i8
            rec["fp32_feasible"] = int(ok_fp.sum())
            rec["int8_feasible"] = int(ok_i8.sum())
            rec["n_both"] = int(both.sum())
            rec["coverage_fp32"] = float(ok_fp.mean())
            rec["coverage_int8"] = float(ok_i8.mean())
            rec["model_failure"] = int((~ok_fp).sum() + (~ok_i8).sum())
            s_fp, e_fp, s_i8, e_i8 = s_fp[both], e_fp[both], s_i8[both], e_i8[both]
            if s_fp.size == 0:
                rec["skipped"] = "no comparable units"
                per_line.append(rec)
                continue
        rec["start_diffs_ms"] = (np.abs(s_i8 - s_fp) * 1000).tolist()
        rec["end_diffs_ms"] = (np.abs(e_i8 - e_fp) * 1000).tolist()
        rec["scene"] = scene.get(task["song"], {})
        per_line.append(rec)

    def q(vals, p):
        return float(np.percentile(vals, p)) if vals else None

    by_song: dict[str, dict] = {}
    by_scene: dict[str, list] = {}
    all_s, all_e = [], []
    fails = n_lines = 0
    for rec in per_line:
        if "start_diffs_ms" not in rec:
            continue
        n_lines += 1
        all_s += rec["start_diffs_ms"]
        all_e += rec["end_diffs_ms"]
        fails += rec.get("model_failure", 0)
        for name, vals in (("song", rec["song"]), (None, None)):
            if name:
                by_song.setdefault(vals, []).extend(rec["start_diffs_ms"])
        sc = rec.get("scene") or {}
        for tag_name in ("fast_singing", "dense_lyrics", "repeated_chorus"):
            if sc.get(tag_name):
                by_scene.setdefault(tag_name, []).extend(rec["start_diffs_ms"])

    def stats(vals):
        if not vals:
            return {"n": 0}
        a = np.array(vals)
        return {"n": len(vals), "mae": float(a.mean()), "median": float(np.median(a)),
                "p90": q(vals, 90), "p95": q(vals, 95), "max": float(a.max()),
                "gt100": float((a > 100).mean()), "gt200": float((a > 200).mean()),
                "gt300": float((a > 300).mean()), "gt500": float((a > 500).mean()),
                "gt1000": float((a > 1000).mean())}

    result = {
        "language": language, "tag": tag, "n_lines": n_lines,
        "start": stats(all_s), "end": stats(all_e),
        "model_failures": fails,
        "peak_rss_fp32_mb": json.loads((scratch / "fp32_fp32.meta.json").read_text())["peak_rss_mb"],
        "peak_rss_int8_mb": json.loads((scratch / f"int8_{tag}.meta.json").read_text())["peak_rss_mb"],
        "fp32_bytes": json.loads((scratch / "fp32_fp32.meta.json").read_text())["bytes"],
        "int8_bytes": json.loads((scratch / f"int8_{tag}.meta.json").read_text())["bytes"],
        "by_song": {k: stats(v) for k, v in sorted(by_song.items())},
        "by_scene": {k: stats(v) for k, v in sorted(by_scene.items())},
        "per_line": per_line,
    }
    if out:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", required=True, choices=["en", "ja", "zh"])
    ap.add_argument("--phase", required=True, choices=["fp32", "int8", "compare"])
    ap.add_argument("--model", type=Path, default=None, help="candidate int8 onnx (phase=int8)")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--tag", default="cand")
    ap.add_argument("--root", type=Path, default=REPO / "Test")
    ap.add_argument("--manifest", type=Path, default=REPO / "experiments/playlist/dataset_manifest.json")
    ap.add_argument("--splits", default="tune,val,blind",
                    help="comma list; blind should only be used after freezing")
    ap.add_argument("--max-lines", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.phase == "compare":
        res = compare(args.language, args.scratch, args.tag, args.out, manifest)
        print(json.dumps({k: res[k] for k in ("language", "tag", "n_lines", "start", "end",
                                              "model_failures", "int8_bytes")},
                         ensure_ascii=False, indent=2))
        return 0
    if args.phase == "int8" and args.model is None:
        raise SystemExit("--model is required for phase=int8")
    meta = run_phase(args.language, args.phase, args.model, args.scratch, args.tag,
                     args.root, manifest, args.splits.split(","), args.max_lines, args.threads)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
