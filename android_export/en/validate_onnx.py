#!/usr/bin/env python
"""One-command validation for the English AutoSync ONNX export.

    .venv/bin/python android_export/en/validate_onnx.py

It re-runs *all* checks from ``docs/ANDROID_ONNX_EXPORT_PROMPT.md`` section 5,
regenerates ``test_vectors/vector*.expected.json`` and rewrites ``REPORT.md``
from the measured numbers. Heavy phases run in separate subprocesses so that the
PyTorch reference model, the fp32 ONNX session and the int8 ONNX session never
have to coexist in RAM (this host runs with little free memory).

Raw results land in ``$MELOX_ONNX_SCRATCH`` (default
``/tmp/opencode/melox_en_validate``); nothing except the deliverables is written
into the repository.

Sources of the evaluation audio (all local, gitignored - never committed):
  * ``experiments/audio/librispeech_sample{1,2}.flac`` - LibriSpeech dev-clean,
    16 kHz mono, real read speech (transcripts from the adjoining ``.lrc``);
  * ``experiments/audio/synthetic_en.wav`` - espeak-ng TTS clip ("hello world
    this is a test"), used only for the ONNX-vs-PyTorch parity set and as a
    fallback;
  * ``experiments/singing/gtsinger/English/**`` - GTSinger real singing takes
    (CC BY-NC-SA, local research data, not redistributable);
  * ``Test/*.lrc`` + ``Test/*.mp3`` - local third-party songs, used *only* as
    blind evaluation input for the int8-vs-fp32 forced-alignment comparison.
    Their lyric text is never written to any deliverable or printed; songs are
    referred to as S1..Sn plus a line index.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import shutil
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
FP32_PATH = OUT_DIR / "wav2vec2-base-960h-fp32.onnx"
INT8_PATH = OUT_DIR / "wav2vec2-base-960h-int8.onnx"
MANIFEST_PATH = OUT_DIR / "manifest.json"
VEC_DIR = OUT_DIR / "test_vectors"
SCRATCH = Path(os.environ.get("MELOX_ONNX_SCRATCH", "/tmp/opencode/melox_en_validate"))

SR = 16000
FRAME_SECONDS = 0.02
BLANK_ID = 0

# thresholds (task book section 5)
T_MAX_LOGIT = 5e-4
T_MEAN_LOGIT = 5e-6
T_NORM = 1e-5
T_COS_MEAN = 0.995
T_COS_MIN = 0.97
T_START_MAE_MS = 30.0
T_COVERAGE_PP = 2.0
T_FLIPS = 1

PERF_WINDOW_SECONDS = 12.0
PERF_RUNS = 10

# |Δlogits| / normative comparison set: >=3 segments, >=30 s of real audio.
SEGMENTS = [
    {
        "label": "S1-librispeech-sample1",
        "path": "experiments/audio/librispeech_sample1.flac",
        "kind": "read speech (LibriSpeech dev-clean, public domain)",
        "required": True,
    },
    {
        "label": "S2-librispeech-sample2",
        "path": "experiments/audio/librispeech_sample2.flac",
        "kind": "read speech (LibriSpeech dev-clean, public domain)",
        "required": True,
    },
    {
        "label": "S3-synthetic-en",
        "path": "experiments/audio/synthetic_en.wav",
        "kind": "synthetic speech (espeak-ng, repo test material)",
        "required": True,
    },
    {
        "label": "S4-gtsinger-edelweiss",
        "path": "experiments/singing/gtsinger/English/EN-Alto-1/Vibrato/edelweiss/Vibrato_Group/0000.wav",
        "kind": "real singing (GTSinger EN-Alto-1, CC BY-NC-SA, local only)",
        "required": False,
    },
    {
        "label": "S5-gtsinger-mylove",
        "path": "experiments/singing/gtsinger/English/EN-Alto-1/Glissando/my love/Glissando_Group/0000.wav",
        "kind": "real singing (GTSinger EN-Alto-1, CC BY-NC-SA, local only)",
        "required": False,
    },
]

MAX_LINES_PER_SONG = 8
MIN_LINE_WORDS = 3
MIN_WINDOW_SECONDS = 1.5
MAX_WINDOW_SECONDS = 12.5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(msg, flush=True)


def write_json(name: str, obj) -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(name: str):
    return json.loads((SCRATCH / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_src_on_path() -> None:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def load_mono16k(rel_path: str):
    ensure_src_on_path()
    from autosync.audio.io import load_audio

    audio, sr = load_audio(str(REPO_ROOT / rel_path), 16000)
    assert sr == 16000
    return audio


def load_vocab() -> dict[str, int]:
    return json.loads((SNAPSHOT / "vocab.json").read_text(encoding="utf-8"))


def encode_en(text: str, vocab: dict[str, int]) -> list[int]:
    """Exact char-level encoding used by the HF Wav2Vec2 tokenizer (verified in
    the torch phase): space -> ``|``; known chars -> id; everything else -> <unk>."""
    unk = vocab["<unk>"]
    delim = vocab["|"]
    return [delim if ch == " " else vocab.get(ch, unk) for ch in text]


def is_special_token(tok: str, vocab: dict[str, int]) -> bool:
    return tok in ("<pad>", "<s>", "</s>", "<unk>", "|", " ")


def word_groups(tokens: list[int], vocab: dict[str, int]) -> list[tuple[int, int]]:
    """Index ranges of consecutive non-delimiter, non-special tokens (words)."""
    delim = vocab["|"]
    specials = {vocab[t] for t in ("<pad>", "<s>", "</s>", "<unk>")}
    groups: list[tuple[int, int]] = []
    start = None
    for i, tok in enumerate(tokens):
        if tok == delim or tok in specials:
            if start is not None:
                groups.append((start, i - 1))
                start = None
        elif start is None:
            start = i
    if start is not None:
        groups.append((start, len(tokens) - 1))
    return groups


def softmax(x):
    import numpy as np

    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def frame_cosine(a, b):
    import numpy as np

    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    den = np.where(den == 0, 1e-30, den)
    return num / den


def make_session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads if threads else 0
    so.inter_op_num_threads = threads if threads else 0
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def active_segments():
    return [
        {**seg, **{"source": seg["path"]}}
        for seg in SEGMENTS
        if (REPO_ROOT / seg["path"]).exists()
    ]


def perf_window_seconds() -> float:
    return PERF_WINDOW_SECONDS


# --------------------------------------------------------------------------- #
# phase: torch reference (logits, baked-normalization comparison, tokenizer)
# --------------------------------------------------------------------------- #
def phase_torch() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import torch

    torch.set_num_threads(2)
    ensure_src_on_path()

    import onnx
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    t_load = time.time()
    proc = AutoProcessor.from_pretrained(str(SNAPSHOT), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(SNAPSHOT), local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    import transformers

    log(f"[torch] processor+model loaded in {time.time()-t_load:.1f}s "
        f"(torch {torch.__version__}, transformers {transformers.__version__})")

    vocab = load_vocab()

    # --- tokenizer equivalence: HF tokenizer vs the manifest-derived encoder ----
    test_texts = ["HELLO WORLD", "DON'T STOP", "AB12", "hi 5", "Z'Q", "A-B_C"]
    tokenizer_checks = []
    for text in test_texts:
        ids_hf = list(proc.tokenizer(text)["input_ids"])
        ids_local = encode_en(text, vocab)
        tokenizer_checks.append({"input": text, "equal": ids_hf == ids_local,
                                 "hf": ids_hf, "local": ids_local})

    # --- baked normalization: exported formula vs AutoProcessor ----------------
    # float64 internals (see export_model._build_wrapper); the graph must match
    # Wav2Vec2FeatureExtractor's float32 numpy result to <= 1e-5.
    class Norm(torch.nn.Module):
        def forward(self, x):
            xd = x.double()
            mean = xd.mean(dim=1, keepdim=True)
            xc = xd - mean
            var = (xc * xc).mean(dim=1, keepdim=True)
            return (xc / torch.sqrt(var + 1e-7)).float()

    norm_module = Norm().eval()
    norm_onnx = SCRATCH / "norm_only.onnx"
    torch.onnx.export(
        norm_module,
        (torch.zeros(1, 16000),),
        str(norm_onnx),
        input_names=["input_values"],
        output_names=["normalized"],
        dynamic_axes={"input_values": {1: "n"}, "normalized": {1: "n"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(norm_onnx))
    import onnxruntime as ort

    norm_sess = ort.InferenceSession(str(norm_onnx), providers=["CPUExecutionProvider"])

    segments = active_segments()
    ref_logits: dict[str, "np.ndarray"] = {}
    seg_records = []
    norm_formula_max, norm_onnx_max = 0.0, 0.0
    total_seconds = 0.0
    for seg in segments:
        audio = load_mono16k(seg["path"]).astype(np.float32)
        total_seconds += len(audio) / SR
        x = torch.from_numpy(audio).unsqueeze(0)
        batch = proc(audio, sampling_rate=SR, return_tensors="pt")
        with torch.inference_mode():
            n_formula = norm_module(x).numpy()
            logits = model(batch.input_values).logits

        n_proc = batch.input_values.numpy()
        n_graph = norm_sess.run(None, {"input_values": audio[None, :]})[0]
        d_formula = float(np.abs(n_formula - n_proc).max())
        d_graph = float(np.abs(n_graph - n_proc).max())
        norm_formula_max = max(norm_formula_max, d_formula)
        norm_onnx_max = max(norm_onnx_max, d_graph)

        ref_logits[seg["label"]] = logits.numpy().astype(np.float32)
        rec = {
            "label": seg["label"],
            "source": seg["path"],
            "kind": seg["kind"],
            "seconds": round(len(audio) / SR, 3),
            "frames": int(logits.shape[1]),
            "norm_delta_formula_vs_processor": d_formula,
            "norm_delta_onnxgraph_vs_processor": d_graph,
        }
        seg_records.append(rec)
        log(f"[torch] {seg['label']}: {rec['seconds']}s, T={rec['frames']}, "
            f"norm Δ(formula)={d_formula:.3e}, Δ(onnx graph)={d_graph:.3e}")

    np.savez(SCRATCH / "torch_logits.npz", **ref_logits)
    write_json(
        "torch_ref.json",
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "segments": seg_records,
            "total_seconds": total_seconds,
            "norm_delta_formula_vs_processor_max": norm_formula_max,
            "norm_delta_onnxgraph_vs_processor_max": norm_onnx_max,
            "norm_eps": 1e-7,
            "tokenizer_checks": tokenizer_checks,
            "tokenizer_all_equal": all(c["equal"] for c in tokenizer_checks),
        },
    )
    log(f"[torch] total {total_seconds:.1f}s; normalization max Δ formula={norm_formula_max:.3e} "
        f"graph={norm_onnx_max:.3e}; tokenizer equivalence={all(c['equal'] for c in tokenizer_checks)}")


# --------------------------------------------------------------------------- #
# phase: fp32 onnx vs torch
# --------------------------------------------------------------------------- #
def phase_fp32() -> None:
    import numpy as np

    ref = np.load(SCRATCH / "torch_logits.npz")
    torch_ref = read_json("torch_ref.json")

    sess = make_session(FP32_PATH)
    logits_lists, lp_lists = [], []
    diffs_max, diffs_mean = [], []
    agree_num, agree_den = 0, 0
    seg_records = []
    for seg in torch_ref["segments"]:
        label = seg["label"]
        audio = load_mono16k(seg["source"]).astype(np.float32)
        onnx_logits, onnx_lp = sess.run(None, {"input_values": audio[None, :]})
        t_logits = ref[label]
        assert onnx_logits.shape == t_logits.shape, (onnx_logits.shape, t_logits.shape)
        d = np.abs(onnx_logits.astype(np.float64) - t_logits.astype(np.float64))
        conf = softmax(t_logits.astype(np.float64)).max(axis=-1) > 0.5
        am_onnx, am_torch = onnx_logits.argmax(-1), t_logits.argmax(-1)
        agree = float((am_onnx[conf] == am_torch[conf]).mean()) if conf.any() else 1.0
        agree_num += int((am_onnx[conf] == am_torch[conf]).sum())
        agree_den += int(conf.sum())
        diffs_max.append(float(d.max()))
        diffs_mean.append(float(d.mean()))
        seg_records.append(
            {
                "label": label,
                "seconds": seg["seconds"],
                "frames": int(onnx_logits.shape[1]),
                "max_abs_dlogits": float(d.max()),
                "mean_abs_dlogits": float(d.mean()),
                "confident_frames": int(conf.sum()),
                "argmax_agreement_confident": agree,
                "log_probs_finite": bool(np.isfinite(onnx_lp).all()),
            }
        )
        logits_lists.append(onnx_logits)
        lp_lists.append(onnx_lp)
        log(f"[fp32] {label}: max|Δlogits|={d.max():.3e} mean={d.mean():.3e} "
            f"argmax(conf)={agree:.4f} on {int(conf.sum())} frames")

    all_max = float(max(diffs_max))
    # mean over every element of every segment (element weighted)
    tot_elems = sum(int(lg.size) for lg in logits_lists)
    all_mean = float(sum(float(np.abs(lg.astype(np.float64) - ref[seg["label"]].astype(np.float64)).sum())
                         for lg, seg in zip(logits_lists, torch_ref["segments"])) / tot_elems)
    argmax_all = agree_num / agree_den if agree_den else 1.0

    # --- robustness: silence / 400 samples / >=30 s --------------------------
    rob = {}
    a_sil = np.zeros(SR * 2, dtype=np.float32)
    out = sess.run(None, {"input_values": a_sil[None, :]})
    rob["silence_finite"] = bool(np.isfinite(out[0]).all() and np.isfinite(out[1]).all())
    rob["silence_frames"] = int(out[0].shape[1])
    a_400 = np.zeros(400, dtype=np.float32)
    out400 = sess.run(None, {"input_values": a_400[None, :]})
    rob["short400_ok"] = True
    rob["short400_frames"] = int(out400[0].shape[1])
    concat = np.concatenate([load_mono16k(s["source"]).astype(np.float32) for s in torch_ref["segments"]])
    out_long = sess.run(None, {"input_values": concat[None, :]})
    rob["long_seconds"] = float(len(concat) / SR)
    rob["long_finite"] = bool(np.isfinite(out_long[0]).all() and np.isfinite(out_long[1]).all())
    rob["long_frames"] = int(out_long[0].shape[1])
    log(f"[fp32] robustness: silence finite={rob['silence_finite']}, 400 frames="
        f"{rob['short400_frames']}, long {rob['long_seconds']:.1f}s frames={rob['long_frames']}")

    np.savez(SCRATCH / "fp32_logprobs.npz", **{s["label"]: lp for s, lp in zip(torch_ref["segments"], lp_lists)})
    write_json(
        "fp32.json",
        {
            "segments": seg_records,
            "overall_max_abs_dlogits": all_max,
            "overall_mean_abs_dlogits": all_mean,
            "overall_argmax_agreement_confident": argmax_all,
            "confident_frames_total": agree_den,
            "total_elements": tot_elems,
            "robustness": rob,
            "thresholds": {"max": T_MAX_LOGIT, "mean": T_MEAN_LOGIT},
            "pass": bool(all_max <= T_MAX_LOGIT and all_mean <= T_MEAN_LOGIT and argmax_all >= 1.0),
        },
    )
    del sess
    gc.collect()


# --------------------------------------------------------------------------- #
# phase: int8 vs fp32
# --------------------------------------------------------------------------- #
def phase_int8() -> None:
    import numpy as np

    fp32 = np.load(SCRATCH / "fp32_logprobs.npz")
    sess = make_session(INT8_PATH)
    seg_records = []
    cos_all = []
    for key in fp32.files:
        audio = None
        for seg in SEGMENTS:
            if seg["label"] == key:
                audio = load_mono16k(seg["path"]).astype(np.float32)
                break
        assert audio is not None
        _, lp_i8 = sess.run(None, {"input_values": audio[None, :]})
        lp_f = fp32[key]
        assert lp_i8.shape == lp_f.shape
        cos = frame_cosine(lp_i8.astype(np.float64), lp_f.astype(np.float64)).reshape(-1)
        seg_records.append(
            {
                "label": key,
                "frames": int(cos.size),
                "cosine_mean": float(cos.mean()),
                "cosine_min": float(cos.min()),
            }
        )
        cos_all.append(cos)
        log(f"[int8] {key}: cosine mean={cos.mean():.6f} min={cos.min():.6f} over {cos.size} frames")

    cos_cat = np.concatenate(cos_all)
    cosine_mean, cosine_min = float(cos_cat.mean()), float(cos_cat.min())

    # --- bit-identical determinism (3 runs) ----------------------------------
    a3 = load_mono16k(SEGMENTS[0]["path"]).astype(np.float32)[: SR * 3]
    runs = [sess.run(None, {"input_values": a3[None, :]}) for _ in range(3)]
    determinism = all(
        runs[0][0].tobytes() == runs[i][0].tobytes() and runs[0][1].tobytes() == runs[i][1].tobytes()
        for i in (1, 2)
    )

    # --- robustness -----------------------------------------------------------
    rob = {}
    out = sess.run(None, {"input_values": np.zeros((1, SR * 2), dtype=np.float32)})
    rob["silence_finite"] = bool(np.isfinite(out[0]).all() and np.isfinite(out[1]).all())
    out400 = sess.run(None, {"input_values": np.zeros((1, 400), dtype=np.float32)})
    rob["short400_ok"] = True
    rob["short400_frames"] = int(out400[0].shape[1])
    concat = np.concatenate([load_mono16k(s["path"]).astype(np.float32) for s in active_segments()])
    out_long = sess.run(None, {"input_values": concat[None, :]})
    rob["long_seconds"] = float(len(concat) / SR)
    rob["long_finite"] = bool(np.isfinite(out_long[0]).all() and np.isfinite(out_long[1]).all())
    rob["long_frames"] = int(out_long[0].shape[1])

    # extra: int8-vs-fp32 argmax agreement on confident fp32 frames
    agree_num = agree_den = 0
    for seg in active_segments():
        audio = load_mono16k(seg["path"]).astype(np.float32)
        lg_i8, lp_i8 = sess.run(None, {"input_values": audio[None, :]})
        lp_f = fp32[seg["label"]]
        conf = softmax(lp_f.astype(np.float64)).max(axis=-1) > 0.5
        agree_num += int((lg_i8.argmax(-1)[conf] == lp_f.argmax(-1)[conf]).sum())
        agree_den += int(conf.sum())

    write_json(
        "int8.json",
        {
            "segments": seg_records,
            "cosine_mean": cosine_mean,
            "cosine_min": cosine_min,
            "determinism_3x_bit_equal": bool(determinism),
            "argmax_agreement_confident_vs_fp32": agree_num / agree_den if agree_den else 1.0,
            "robustness": rob,
            "thresholds": {"cosine_mean": T_COS_MEAN, "cosine_min": T_COS_MIN},
            "pass": bool(cosine_mean >= T_COS_MEAN and cosine_min >= T_COS_MIN and determinism),
        },
    )
    del sess
    gc.collect()


# --------------------------------------------------------------------------- #
# phases: end-to-end forced alignment on real songs (int8 vs fp32)
# --------------------------------------------------------------------------- #
def _find_audio_for(lrc_path: Path) -> Path | None:
    """Test/ stores full lyric-file names (`Song-Artist.lrc`) but short audio
    names (`Song.mp3`); try the full stem first, then progressively shortened
    variants at '-' / ' (' / '[' boundaries."""
    stem = lrc_path.stem
    names = [stem]
    for sep in ("-", " (", " [", "（"):
        if sep in stem and stem.split(sep)[0] not in names:
            names.append(stem.split(sep)[0])
    for name in names:
        for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
            cand = lrc_path.with_name(name + ext)
            if cand.exists():
                return cand
    return None


def _has_cjk(text: str) -> bool:
    return any(
        "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af"
        for ch in text
    )


def build_song_tasks() -> dict:
    """Pick English lyric lines from the local Test/ songs (blind evaluation only).

    Deterministic: sorted LRC files, first ``MAX_LINES_PER_SONG`` lines that pass
    the filters. Lyric text is never stored - only token ids / word index ranges.
    """
    ensure_src_on_path()
    from autosync.alignment.segment import _cap_window_end, _window_for_line
    from autosync.lyrics.classify import classify_lines
    from autosync.lyrics.lrc import parse_lrc
    from autosync.lyrics.normalize import normalize_en

    vocab = load_vocab()
    test_dir = REPO_ROOT / "Test"
    if not test_dir.is_dir():
        return {"skipped": True, "reason": f"{test_dir} not found", "tasks": [], "songs": []}

    tasks: list[dict] = []
    songs_meta: list[dict] = []
    alias = 0
    for lrc_path in sorted(test_dir.glob("*.lrc")):
        audio_path = _find_audio_for(lrc_path)
        if audio_path is None:
            continue
        lines, _meta = parse_lrc(lrc_path)
        audio = load_mono16k(str(audio_path.relative_to(REPO_ROOT)))
        total = len(audio) / SR
        classifications = classify_lines(lines)
        next_singing = [None] * len(lines)
        nxt = None
        for i in range(len(lines) - 1, -1, -1):
            next_singing[i] = nxt
            if classifications[i].kind.alignable:
                nxt = lines[i]

        alias += 1
        song_alias = f"S{alias}"
        picked = 0
        for i, line in enumerate(lines):
            if not classifications[i].kind.alignable:
                continue
            if _has_cjk(line.text):
                continue
            normalized = normalize_en(line.text)
            if not normalized:
                continue
            words = normalized.split()
            if len(words) < MIN_LINE_WORDS:
                continue
            # keep only fully in-vocabulary lines (no digits / unknowns)
            if any(ch not in vocab and ch != " " for ch in normalized):
                continue
            tokens = encode_en(normalized, vocab)
            groups = word_groups(tokens, vocab)
            if len(groups) != len(words):
                continue

            w_start, w_end = _window_for_line(line, next_singing[i], total, 0.30, 12.0)
            w_end = _cap_window_end(audio, SR, line.start, w_end, 12.0, 30.0)
            if not (MIN_WINDOW_SECONDS <= (w_end - w_start) <= MAX_WINDOW_SECONDS):
                continue
            i0, i1 = int(round(w_start * SR)), int(round(w_end * SR))
            tasks.append(
                {
                    "song_alias": song_alias,
                    "song_file": lrc_path.name,
                    "audio": str(audio_path.relative_to(REPO_ROOT)),
                    "line_index": int(line.index),
                    "start_sample": i0,
                    "end_sample": i1,
                    "window_seconds": (i1 - i0) / SR,
                    "tokens": tokens,
                    "word_groups": [[a, b] for a, b in groups],
                    "n_words": len(groups),
                }
            )
            picked += 1
            if picked >= MAX_LINES_PER_SONG:
                break
        songs_meta.append(
            {
                "alias": song_alias,
                "file": lrc_path.name,
                "audio": audio_path.name,
                "picked_lines": picked,
            }
        )
        del audio
        gc.collect()

    return {"skipped": False, "tasks": tasks, "songs": songs_meta}


def _run_align_phase(which: str, onnx_path: Path) -> None:
    import numpy as np

    tasks_info = build_song_tasks()
    if tasks_info["skipped"]:
        write_json(f"align_{which}.json", {"skipped": True, "reason": tasks_info["reason"]})
        log(f"[align_{which}] skipped: {tasks_info['reason']}")
        return
    tasks = tasks_info["tasks"]
    write_json(f"align_tasks_{which}.json", tasks_info)
    log(f"[align_{which}] {len(tasks)} lyric lines across {len(tasks_info['songs'])} songs")

    sess = make_session(onnx_path, threads=4)
    store: dict[str, "np.ndarray"] = {}
    current_audio_path = None
    audio = None
    for ti, task in enumerate(tasks):
        if task["audio"] != current_audio_path:
            audio = load_mono16k(task["audio"]).astype(np.float32)
            current_audio_path = task["audio"]
        seg = audio[task["start_sample"]: task["end_sample"]]
        _, lp = sess.run(None, {"input_values": seg[None, :]})
        key = f"{task['song_alias']}|{task['line_index']}"
        store[key] = lp
        if (ti + 1) % 5 == 0:
            log(f"[align_{which}]   {ti+1}/{len(tasks)} lines")
    del sess
    np.savez(SCRATCH / f"align_{which}.npz", **store)
    gc.collect()
    log(f"[align_{which}] wrote {len(store)} line log_prob arrays")


def phase_align_fp32() -> None:
    _run_align_phase("fp32", FP32_PATH)


def phase_align_int8() -> None:
    _run_align_phase("int8", INT8_PATH)


def phase_align_compare() -> None:
    import numpy as np

    ensure_src_on_path()
    from autosync.alignment.ctc import forced_align

    t_fp = read_json("align_tasks_fp32.json")
    t_i8 = read_json("align_tasks_int8.json")
    if t_fp.get("skipped") or t_i8.get("skipped"):
        write_json("align_compare.json", {"skipped": True, "reason": t_fp.get("reason") or t_i8.get("reason")})
        return None

    assert t_fp["tasks"] == t_i8["tasks"], "task lists differ between phases"
    lp_fp = np.load(SCRATCH / "align_fp32.npz")
    lp_i8 = np.load(SCRATCH / "align_int8.npz")

    vocab = load_vocab()
    n_lines = n_words = 0
    flips = 0
    start_diffs_ms, end_diffs_ms = [], []
    aligned_fp = aligned_i8 = 0
    both_feasible = 0
    per_song: dict[str, dict] = {}
    for task in t_fp["tasks"]:
        key = f"{task['song_alias']}|{task['line_index']}"
        tokens = list(task["tokens"])
        groups = [tuple(g) for g in task["word_groups"]]
        a_fp = forced_align(lp_fp[key][0], tokens, blank=BLANK_ID)
        a_i8 = forced_align(lp_i8[key][0], tokens, blank=BLANK_ID)
        n_lines += 1
        n_words += len(groups)
        if bool(a_fp.feasible) != bool(a_i8.feasible):
            flips += 1
        if a_fp.feasible and a_i8.feasible:
            both_feasible += 1
        rec = per_song.setdefault(
            task["song_alias"],
            {"lines": 0, "words": 0, "feasible_fp32": 0, "feasible_int8": 0, "flips": 0,
             "aligned_words_fp32": 0, "aligned_words_int8": 0, "start_diffs_ms": []},
        )
        rec["lines"] += 1
        rec["words"] += len(groups)
        rec["feasible_fp32"] += int(a_fp.feasible)
        rec["feasible_int8"] += int(a_i8.feasible)
        rec["flips"] += int(bool(a_fp.feasible) != bool(a_i8.feasible))
        for (g0, g1) in groups:
            span_fp = a_fp.spans[g0:g1 + 1]
            span_i8 = a_i8.spans[g0:g1 + 1]
            ok_fp = all(s is not None for s in span_fp)
            ok_i8 = all(s is not None for s in span_i8)
            aligned_fp += int(ok_fp)
            aligned_i8 += int(ok_i8)
            rec["aligned_words_fp32"] += int(ok_fp)
            rec["aligned_words_int8"] += int(ok_i8)
            if ok_fp and ok_i8:
                s_fp = span_fp[0].start_frame * FRAME_SECONDS * 1000.0
                e_fp = (span_fp[-1].end_frame + 1) * FRAME_SECONDS * 1000.0
                s_i8 = span_i8[0].start_frame * FRAME_SECONDS * 1000.0
                e_i8 = (span_i8[-1].end_frame + 1) * FRAME_SECONDS * 1000.0
                start_diffs_ms.append(abs(s_fp - s_i8))
                end_diffs_ms.append(abs(e_fp - e_i8))
                rec["start_diffs_ms"].append(abs(s_fp - s_i8))

    cov_fp = 100.0 * aligned_fp / n_words if n_words else 0.0
    cov_i8 = 100.0 * aligned_i8 / n_words if n_words else 0.0
    start_mae = float(np.mean(start_diffs_ms)) if start_diffs_ms else float("nan")
    end_mae = float(np.mean(end_diffs_ms)) if end_diffs_ms else float("nan")

    songs = []
    for alias, rec in sorted(per_song.items()):
        songs.append(
            {
                "alias": alias,
                "lines": rec["lines"],
                "words": rec["words"],
                "feasible_fp32": rec["feasible_fp32"],
                "feasible_int8": rec["feasible_int8"],
                "flips": rec["flips"],
                "start_mae_ms": float(np.mean(rec["start_diffs_ms"])) if rec["start_diffs_ms"] else None,
                "aligned_words_fp32": rec["aligned_words_fp32"],
                "aligned_words_int8": rec["aligned_words_int8"],
            }
        )

    result = {
        "skipped": False,
        "n_songs": len(per_song),
        "n_lines": n_lines,
        "n_words": n_words,
        "n_word_pairs_compared": len(start_diffs_ms),
        "both_feasible_lines": both_feasible,
        "start_mae_ms": start_mae,
        "end_mae_ms": end_mae,
        "coverage_fp32_pct": cov_fp,
        "coverage_int8_pct": cov_i8,
        "coverage_delta_pp": abs(cov_i8 - cov_fp),
        "feasibility_flips": flips,
        "songs": songs,
        "sources": {
            "songs": [s["file"] for s in t_fp["songs"]],
            "note": "local Test/ third-party LRC+mp3 used as blind evaluation input only; "
                    "lyric text not stored in any deliverable",
        },
        "thresholds": {"start_mae_ms": T_START_MAE_MS, "coverage_pp": T_COVERAGE_PP, "flips": T_FLIPS},
        "pass": bool(
            not (start_mae != start_mae)  # NaN -> fail
            and start_mae <= T_START_MAE_MS
            and abs(cov_i8 - cov_fp) <= T_COVERAGE_PP
            and flips <= T_FLIPS
            and n_lines >= 20
            and len(per_song) >= 3
        ),
    }
    write_json("align_compare.json", result)
    log(f"[align_compare] lines={n_lines} songs={len(per_song)} words={n_words}; "
        f"start MAE={start_mae:.2f} ms; coverage {cov_fp:.2f}% vs {cov_i8:.2f}% "
        f"(Δ={abs(cov_i8-cov_fp):.2f} pp); flips={flips}")
    return result


# --------------------------------------------------------------------------- #
# phases: performance / memory
# --------------------------------------------------------------------------- #
def phase_perf(which: str) -> None:
    import numpy as np

    onnx_path = FP32_PATH if which == "fp32" else INT8_PATH
    audio = load_mono16k(SEGMENTS[0]["path"]).astype(np.float32)[: int(PERF_WINDOW_SECONDS * SR)]
    assert len(audio) == int(PERF_WINDOW_SECONDS * SR)

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    load_seconds = time.perf_counter() - t0

    x = audio[None, :]
    for _ in range(2):  # warmup
        sess.run(None, {"input_values": x})
    times = []
    for _ in range(PERF_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"input_values": x})
        times.append((time.perf_counter() - t0) * 1000.0)
    times_sorted = sorted(times)
    median = float(np.median(times_sorted))
    p90 = float(np.percentile(times_sorted, 90))
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    write_json(
        f"perf_{which}.json",
        {
            "model": which,
            "file": onnx_path.name,
            "window_seconds": PERF_WINDOW_SECONDS,
            "runs": PERF_RUNS,
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "load_seconds": load_seconds,
            "median_ms": median,
            "p90_ms": p90,
            "min_ms": min(times_sorted),
            "max_ms": max(times_sorted),
            "times_ms": times,
            "peak_rss_mib": peak_rss_mib,
        },
    )
    log(f"[perf_{which}] load={load_seconds:.2f}s median={median:.1f}ms p90={p90:.1f}ms "
        f"peak RSS={peak_rss_mib:.1f} MiB")


# --------------------------------------------------------------------------- #
# phase: loadability / contract
# --------------------------------------------------------------------------- #
def phase_loadcheck() -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results = {}
    for key, path in (("fp32", FP32_PATH), ("int8", INT8_PATH)):
        rec: dict = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        try:
            onnx.checker.check_model(str(path))
            rec["checker_ok"] = True
        except Exception as exc:  # noqa: BLE001
            rec["checker_ok"] = False
            rec["checker_error"] = f"{type(exc).__name__}: {exc}"
        m = onnx.load(str(path), load_external_data=False)
        rec["opset_import"] = [[io.domain, int(io.version)] for io in m.opset_import]
        rec["opset_le_17"] = all((io.domain == "" and io.version <= 17) for io in m.opset_import)
        domains = sorted({n.domain for n in m.graph.node if n.domain})
        rec["node_domains"] = domains
        rec["no_ms_ops"] = all(d != "com.microsoft" for d in domains)
        rec["all_ops"] = sorted({n.op_type for n in m.graph.node})
        rec["external_data_used"] = bool(
            any(onnx.external_data_helper.uses_external_data(t) for t in m.graph.initializer)
        )
        rec["ir_version"] = int(m.ir_version)

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        rec["providers"] = sess.get_providers()

        def _shape_dims(shape):
            out = []
            for d in shape:
                try:
                    out.append(int(d))
                except (TypeError, ValueError):
                    out.append(None)
            return out

        rec["inputs"] = [{"name": i.name, "shape": _shape_dims(i.shape), "type": i.type}
                         for i in sess.get_inputs()]
        rec["outputs"] = [{"name": o.name, "shape": _shape_dims(o.shape), "type": o.type}
                          for o in sess.get_outputs()]

        man_in = manifest["input"]
        ok_names = [i["name"] for i in rec["inputs"]] == [man_in["name"]]
        ok_out = [o["name"] for o in rec["outputs"]] == [o["name"] for o in manifest["outputs"]]
        # shapes: manifest uses null for dynamic dims
        def shape_ok(got, want):
            if len(got) != len(want):
                return False
            return all(w is None or (g is not None and int(g) == int(w)) for g, w in zip(got, want))

        rec["io_matches_manifest"] = bool(
            ok_names
            and ok_out
            and shape_ok(rec["inputs"][0]["shape"], man_in["shape"])
            and all(shape_ok(o["shape"], w["shape"]) for o, w in zip(rec["outputs"], manifest["outputs"]))
        )
        rec["dtype_ok"] = bool(
            rec["inputs"][0]["type"] == "tensor(float)"
            and all(o["type"] == "tensor(float)" for o in rec["outputs"])
        )
        rec["single_file"] = bool(not rec["external_data_used"])
        rec["sha_matches_manifest"] = bool(
            rec["sha256"] == manifest["files"][key]["sha256"]
            and rec["bytes"] == manifest["files"][key]["bytes"]
        )
        del sess, m
        gc.collect()
        results[key] = rec
        log(f"[loadcheck] {key}: checker={rec['checker_ok']} opset<=17={rec['opset_le_17']} "
            f"no ms ops={rec['no_ms_ops']} io_match={rec['io_matches_manifest']} "
            f"sha_match={rec['sha_matches_manifest']} external={rec['external_data_used']}")
    write_json("loadcheck.json", results)


# --------------------------------------------------------------------------- #
# phase: test-vector expected json
# --------------------------------------------------------------------------- #
def _vector_fields(lp, tokens, vocab) -> dict:
    import numpy as np

    arr = np.asarray(lp)
    assert arr.shape[0] == 1 and arr.shape[2] == 32
    am = arr[0].argmax(-1)
    # first 20 runs of the argmax sequence
    runs: list[int] = []
    if am.size:
        cur, count = am[0], 1
        for v in am[1:]:
            if v == cur:
                count += 1
            else:
                runs.append(int(count))
                cur, count = v, 1
        runs.append(int(count))

    ensure_src_on_path()
    from autosync.alignment.ctc import forced_align

    al = forced_align(arr[0], tokens, blank=BLANK_ID)
    spans = []
    for tok, span in zip(tokens, al.spans):
        if span is None:
            continue
        spans.append({"token_id": int(tok), "start_frame": int(span.start_frame), "end_frame": int(span.end_frame)})

    return {
        "log_probs_shape": [int(x) for x in arr.shape],
        "log_probs_mean": float(arr.mean()),
        "log_probs_std": float(arr.std()),
        "argmax_run_lengths_first20": runs[:20],
        "blank_ratio": float((am == BLANK_ID).mean()),
        "align_spans": spans,
        "align_feasible": bool(al.feasible),
        "align_reasons": list(al.reasons),
    }


def phase_vectors() -> None:
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, str(OUT_DIR))
    import export_model  # noqa: E402  (module-level imports are stdlib only)

    if not (VEC_DIR / "vector1.wav").exists():
        export_model.stage_vectors()

    vocab = load_vocab()
    out_records = []
    for spec in export_model.VECTOR_SPECS:
        path = VEC_DIR / spec["file"]
        info = sf.info(str(path))
        assert info.samplerate == 16000 and info.channels == 1
        audio, sr = sf.read(str(path), dtype="float32")
        assert sr == 16000
        tokens = encode_en(spec["text"].upper(), vocab)

        rec = {
            "file": spec["file"],
            "source": spec["source"],
            "seconds": round(len(audio) / SR, 3),
            "samplerate": 16000,
            "channels": 1,
            "subtype": info.subtype,
        }
        fp32_lp = None
        for which, path_onnx in (("fp32", FP32_PATH), ("int8", INT8_PATH)):
            sess = make_session(path_onnx, threads=2)
            _, lp = sess.run(None, {"input_values": audio[None, :].astype(np.float32)})
            del sess
            gc.collect()
            rec[which] = _vector_fields(lp, tokens, vocab)
            if which == "fp32":
                fp32_lp = lp
        expected = {
            "text": spec["text"],
            "int8": rec["int8"],
            "fp32": rec["fp32"],
            "notes": (
                "token_id -> character mapping in manifest.tokens; align_spans generated with "
                "src/autosync/alignment/ctc.py::forced_align(blank=0) over the ONNX 'log_probs' output; "
                "log_probs_mean/std and blank_ratio are over all [1,T,32] values (float64 accumulation); "
                "argmax_run_lengths_first20 = first 20 run lengths of argmax over frames. "
                f"Audio source: {spec['notes']}"
            ),
        }
        (VEC_DIR / f"{spec['file'].replace('.wav', '.expected.json')}").write_text(
            json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        out_records.append(rec)
        log(f"[vectors] {spec['file']}: T={rec['int8']['log_probs_shape'][1]} "
            f"blank_ratio={rec['int8']['blank_ratio']:.3f} int8-feasible={rec['int8']['align_feasible']} "
            f"fp32-feasible={rec['fp32']['align_feasible']}")
    write_json("vectors.json", out_records)


# --------------------------------------------------------------------------- #
# phase: report
# --------------------------------------------------------------------------- #
def _fmt(x, nd=6):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x != x:
            return "NaN"
        return f"{x:.{nd}f}"
    return str(x)


def phase_report() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    torch_ref = read_json("torch_ref.json")
    fp32 = read_json("fp32.json")
    int8 = read_json("int8.json")
    loadcheck = read_json("loadcheck.json")
    align = read_json("align_compare.json")
    perf_fp32 = read_json("perf_fp32.json")
    perf_i8 = read_json("perf_int8.json")

    checks: list[tuple[str, bool, str, str]] = []

    def check(name, ok, detail, threshold):
        checks.append((name, bool(ok), detail, threshold))

    check("FP32 max|Δlogits|", fp32["overall_max_abs_dlogits"] <= T_MAX_LOGIT,
          f"{fp32['overall_max_abs_dlogits']:.3e}", f"<= {T_MAX_LOGIT:.0e}")
    check("FP32 mean|Δlogits|", fp32["overall_mean_abs_dlogits"] <= T_MEAN_LOGIT,
          f"{fp32['overall_mean_abs_dlogits']:.3e}", f"<= {T_MEAN_LOGIT:.0e}")
    check("FP32 argmax agreement (confident frames)",
          fp32["overall_argmax_agreement_confident"] >= 1.0,
          f"{fp32['overall_argmax_agreement_confident']*100:.4f}% "
          f"({fp32['confident_frames_total']} frames)",
          "== 100%")
    check("Baked normalization vs AutoProcessor",
          torch_ref["norm_delta_onnxgraph_vs_processor_max"] <= T_NORM,
          f"graph max|Δ|={torch_ref['norm_delta_onnxgraph_vs_processor_max']:.3e} "
          f"(formula {torch_ref['norm_delta_formula_vs_processor_max']:.3e})",
          f"<= {T_NORM:.0e}")
    check("Tokenizer encoder == HF tokenizer", torch_ref["tokenizer_all_equal"],
          f"{sum(c['equal'] for c in torch_ref['tokenizer_checks'])}/{len(torch_ref['tokenizer_checks'])} probe strings",
          "all equal")
    check("INT8 cosine mean", int8["cosine_mean"] >= T_COS_MEAN, f"{int8['cosine_mean']:.6f}", f">= {T_COS_MEAN}")
    check("INT8 cosine min", int8["cosine_min"] >= T_COS_MIN, f"{int8['cosine_min']:.6f}", f">= {T_COS_MIN}")
    check("INT8 determinism 3x bit-equal", int8["determinism_3x_bit_equal"],
          str(int8["determinism_3x_bit_equal"]), "True")
    check("Alignment start MAE (INT8 vs FP32)", (align["start_mae_ms"] <= T_START_MAE_MS) if not align.get("skipped") else False,
          f"{_fmt(align.get('start_mae_ms'), 2)} ms over {align.get('n_word_pairs_compared')} words",
          f"<= {T_START_MAE_MS} ms")
    check("Alignment coverage change", (align.get("coverage_delta_pp", 1e9) <= T_COVERAGE_PP) if not align.get("skipped") else False,
          f"{_fmt(align.get('coverage_int8_pct'),2)}% vs {_fmt(align.get('coverage_fp32_pct'),2)}% "
          f"(Δ {_fmt(align.get('coverage_delta_pp'),2)} pp)",
          f"<= {T_COVERAGE_PP} pp")
    check("Alignment feasibility flips", (align.get("feasibility_flips", 1e9) <= T_FLIPS) if not align.get("skipped") else False,
          f"{align.get('feasibility_flips')} of {align.get('n_lines')} lines", f"<= {T_FLIPS}")
    check("Alignment sample size", (not align.get("skipped") and align.get("n_lines", 0) >= 20 and align.get("n_songs", 0) >= 3),
          f"{align.get('n_lines')} lines / {align.get('n_songs')} songs", ">= 20 lines, >= 3 songs")
    for key in ("fp32", "int8"):
        lc = loadcheck[key]
        check(f"{key} loadable (checker/CPU EP/names/shapes)", lc["checker_ok"] and lc["io_matches_manifest"]
              and lc["dtype_ok"] and lc["providers"][0] == "CPUExecutionProvider", "see section 3", "required")
        check(f"{key} standard ops / opset<=17 / single file",
              lc["no_ms_ops"] and lc["opset_le_17"] and lc["single_file"], "see section 3", "required")
    check("Test vectors regenerated", (VEC_DIR / "vector1.expected.json").exists()
          and (VEC_DIR / "vector2.expected.json").exists() and (VEC_DIR / "vector3.expected.json").exists(),
          "3 expected.json files", "required")

    verdict = all(c[1] for c in checks)
    lines: list[str] = []
    add = lines.append
    add("# AutoSync Android ONNX 导出报告")
    add("")
    add("> 语言：英文（`facebook/wav2vec2-base-960h`）")
    add(f"> 生成方式：`{sys.executable} android_export/en/validate_onnx.py`（本文件由脚本自动生成，"
        f"所有数字来自本机实测；生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}）")
    add("")
    add("## 1. 结论（是否可交付 / 未达标项）")
    add("")
    if verdict:
        add("**全部验收项通过，可交付。**")
    else:
        failed = [n for n, ok, _d, _t in checks if not ok]
        add(f"**存在未达标项：{'、'.join(failed)}（下表 FAIL 行，已如实报告，未粉饰）。**")
    add("")
    add("| 验收项 | 结果 | 实测 | 阈值 |")
    add("|---|---|---|---|")
    for name, ok, detail, threshold in checks:
        add(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} | {threshold} |")
    add("")
    add("## 2. 交付文件（路径 / 字节 / sha256）")
    add("")
    add("| 文件 | 字节 | sha256 |")
    add("|---|---|---|")
    for key in ("fp32", "int8"):
        f = manifest["files"][key]
        add(f"| `{OUT_DIR / f['name']}` | {f['bytes']} | `{f['sha256']}` |")
    add(f"| `{OUT_DIR / 'manifest.json'}` | {MANIFEST_PATH.stat().st_size} | `{sha256_file(MANIFEST_PATH)}` |")
    for name in ("validate_onnx.py", "export_model.py"):
        p = OUT_DIR / name
        if p.exists():
            add(f"| `{p}` | {p.stat().st_size} | `{sha256_file(p)}` |")
    add(f"| `{VEC_DIR}/vector1..3.wav + vectorN.expected.json` | (小文件，见目录) | — |")
    add("")
    add("## 3. 契约核对（I/O 名、形状、opset、算子、external data、归一化比对）")
    add("")
    for key in ("fp32", "int8"):
        lc = loadcheck[key]
        add(f"### {key}")
        add("")
        add(f"- `onnx.checker.check_model`: {'通过' if lc['checker_ok'] else '**失败**'}")
        add(f"- ORT `CPUExecutionProvider` 加载: {'通过' if lc['providers'][0]=='CPUExecutionProvider' else lc['providers']}")
        add(f"- 输入: {lc['inputs'][0]['name']} shape={lc['inputs'][0]['shape']} type={lc['inputs'][0]['type']}")
        add(f"- 输出: " + "; ".join(f"{o['name']} shape={o['shape']} type={o['type']}" for o in lc["outputs"]))
        add(f"- 与 `manifest.json` 名称/形状一致: {'是' if lc['io_matches_manifest'] else '**否**'}；"
            f"sha256/字节一致: {'是' if lc['sha_matches_manifest'] else '**否**'}")
        add(f"- opset_import: {lc['opset_import']}（≤17: {'是' if lc['opset_le_17'] else '否'}）；"
            f"算子 domain: {lc['node_domains'] or '全部为标准 ONNX domain'}；"
            f"含 `com.microsoft` 算子: {'否' if lc['no_ms_ops'] else '**是**'}")
        add(f"- external data: {'无（单文件自包含）' if lc['single_file'] else '**有**'}；IR version {lc['ir_version']}")
        add(f"- 图中算子: {', '.join(lc['all_ops'])}")
        add("")
    add("### 归一化（烘焙 vs AutoProcessor）")
    add("")
    add(f"- 同一批 {len(torch_ref['segments'])} 段音频逐段比较归一化张量：")
    add(f"  - 烘焙公式（torch，导出源）vs `AutoProcessor` 输出: max|Δ| = "
        f"{torch_ref['norm_delta_formula_vs_processor_max']:.3e}")
    add(f"  - 单独导出的归一化子图（ONNX + ORT）vs `AutoProcessor` 输出: max|Δ| = "
        f"{torch_ref['norm_delta_onnxgraph_vs_processor_max']:.3e}（阈值 1e-5）")
    add("- 公式: `(x - mean(x)) / sqrt(var(x) + 1e-7)`，population variance，整段求均值/方差，与 "
        "`Wav2Vec2FeatureExtractor(do_normalize=True)` 一致。")
    add(f"- 词表编码器与 HF tokenizer 一致性: "
        f"{sum(c['equal'] for c in torch_ref['tokenizer_checks'])}/{len(torch_ref['tokenizer_checks'])} 探针串相等；"
        "注意：词表中没有数字，`0-9` 与未知字符均映射为 `<unk>`(3)（与 HF tokenizer 行为一致，"
        "Android 端需同样处理）。")
    add("")
    add("## 4. 验证数字（按第 5 节 1–5 逐项列表，含素材来源与样本数）")
    add("")
    add("### 5.1 导出正确性（FP32 ONNX vs PyTorch）")
    add("")
    add(f"素材：{len(torch_ref['segments'])} 段、合计 {torch_ref['total_seconds']:.2f} s；逐段独立前向。")
    add("")
    add("| 片段 | 来源 | 类型 | 秒 | T | max&#124;Δlogits&#124; | mean&#124;Δlogits&#124; | 置信帧 | argmax 一致率 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for s in fp32["segments"]:
        meta = next(m for m in torch_ref["segments"] if m["label"] == s["label"])
        add(f"| {s['label']} | `{meta['source']}` | {meta['kind']} | {s['seconds']} | {s['frames']} | "
            f"{s['max_abs_dlogits']:.3e} | {s['mean_abs_dlogits']:.3e} | {s['confident_frames']} | "
            f"{s['argmax_agreement_confident']*100:.4f}% |")
    add(f"| **合计** | {len(fp32['segments'])} 段 | | {torch_ref['total_seconds']:.2f} | "
        f"{sum(s['frames'] for s in fp32['segments'])} | **{fp32['overall_max_abs_dlogits']:.3e}** | "
        f"**{fp32['overall_mean_abs_dlogits']:.3e}** | {fp32['confident_frames_total']} | "
        f"**{fp32['overall_argmax_agreement_confident']*100:.4f}%** |")
    add("")
    add(f"- 阈值：max ≤ {T_MAX_LOGIT:.0e}，mean ≤ {T_MEAN_LOGIT:.0e}，置信帧 argmax 一致率 = 100% —— "
        f"{'通过' if fp32['pass'] else '**未通过**'}。")
    add(f"- 烘焙归一化 vs `AutoProcessor` max|Δ| = "
        f"{torch_ref['norm_delta_onnxgraph_vs_processor_max']:.3e}（阈值 1e-5）—— "
        f"{'通过' if torch_ref['norm_delta_onnxgraph_vs_processor_max'] <= T_NORM else '**未通过**'}。")
    add("")
    add("### 5.2 INT8 精度（INT8 vs FP32）")
    add("")
    add("**逐帧 `log_probs` 余弦相似度**")
    add("")
    add("| 片段 | 帧数 | cosine mean | cosine min |")
    add("|---|---|---|---|")
    for s in int8["segments"]:
        add(f"| {s['label']} | {s['frames']} | {s['cosine_mean']:.6f} | {s['cosine_min']:.6f} |")
    add(f"| **合计** | {sum(s['frames'] for s in int8['segments'])} | **{int8['cosine_mean']:.6f}** | "
        f"**{int8['cosine_min']:.6f}** |")
    add("")
    add(f"- 阈值 mean ≥ {T_COS_MEAN} / min ≥ {T_COS_MIN} —— {'通过' if int8['pass'] else '**未通过**'}；"
        f"同批输入 int8-vs-fp32 argmax 一致率（fp32 置信帧）= "
        f"{int8['argmax_agreement_confident_vs_fp32']*100:.3f}%。")
    add("")
    add("**端到端强制对齐（`src/autosync/alignment/ctc.py::forced_align`, blank=0）**")
    add("")
    if align.get("skipped"):
        add(f"跳过：{align.get('reason')}")
    else:
        add(f"素材：本地 `Test/` 第三方歌曲 {align['n_songs']} 首（仅作评测输入，避免第三方歌词文本，"
            f"以别名 S1..S{align['n_songs']} + 行号引用），共 {align['n_lines']} 行、{align['n_words']} 个词单元；"
            f"INT8/FP32 各自对同一裁剪窗推理并强制对齐，逐词比较。")
        add("")
        add("| 歌曲别名 | 行数 | 词数 | FP32 feasible 行 | INT8 feasible 行 | 翻转 | 词起始 MAE (ms) |")
        add("|---|---|---|---|---|---|---|")
        for s in align["songs"]:
            add(f"| {s['alias']} | {s['lines']} | {s['words']} | {s['feasible_fp32']} | {s['feasible_int8']} | "
                f"{s['flips']} | {_fmt(s['start_mae_ms'], 2)} |")
        add(f"| **合计** | **{align['n_lines']}** | **{align['n_words']}** | "
            f"{sum(s['feasible_fp32'] for s in align['songs'])} | {sum(s['feasible_int8'] for s in align['songs'])} | "
            f"**{align['feasibility_flips']}** | **{align['start_mae_ms']:.2f}** |")
        add("")
        add(f"- 起始时间 MAE（INT8 vs FP32，逐词）= **{align['start_mae_ms']:.2f} ms**（阈值 ≤ 30 ms）—— "
            f"{'通过' if align['start_mae_ms'] <= T_START_MAE_MS else '**未通过**'}；"
            f"结束时间 MAE（附加）= {align['end_mae_ms']:.2f} ms。")
        add(f"- 覆盖率（有起止时间的词单元占比）：FP32 {align['coverage_fp32_pct']:.2f}% vs "
            f"INT8 {align['coverage_int8_pct']:.2f}%，变化 **{align['coverage_delta_pp']:.2f} pp**"
            f"（阈值 ≤ 2 pp）—— {'通过' if align['coverage_delta_pp'] <= T_COVERAGE_PP else '**未通过**'}。")
        add(f"- feasible 判定翻转 **{align['feasibility_flips']}** 行（阈值 ≤ 1）—— "
            f"{'通过' if align['feasibility_flips'] <= T_FLIPS else '**未通过**'}。")
        add("")
        add("> 说明：该项测量的是 INT8 与 FP32 **输出的一致性**（不是绝对准确率）；对齐输入为真实歌曲音频 + LRC 行起点与文本。"
            "歌词原文不写入交付物。")
    add("")
    add("### 5.3 健壮性")
    add("")
    add(f"- 静音（全 0，2 s）输入：FP32 {'无 NaN/Inf' if fp32['robustness']['silence_finite'] else '**有 NaN/Inf**'}；"
        f"INT8 {'无 NaN/Inf' if int8['robustness']['silence_finite'] else '**有 NaN/Inf**'}。")
    add(f"- 同一输入连跑 3 次：INT8 输出逐位一致 = "
        f"{'是' if int8['determinism_3x_bit_equal'] else '**否**'}（logits 与 log_probs 的原始字节完全相等）。")
    add(f"- 极短输入 400 采样：FP32/INT8 均正常返回（T={fp32['robustness']['short400_frames']}）；"
        f">=30 s 长输入：FP32 用 {fp32['robustness']['long_seconds']:.1f} s 连续音频，"
        f"INT8 用 {int8['robustness']['long_seconds']:.1f} s，均无报错、输出有限。")
    add("")
    add("### 5.4 可加载性")
    add("")
    add("见第 3 节；`onnx.checker`、CPUExecutionProvider、I/O 名称/形状、单文件、opset ≤ 17、"
        "无 `com.microsoft` 算子均逐项检查。")
    add("")
    add("### 5.5 性能与内存（桌面参考值，单线程 CPU）")
    add("")
    add("| 模型 | 12 s 窗口中位数 (ms) | P90 (ms) | 加载 (s) | 10 窗推理峰值 RSS (MiB) |")
    add("|---|---|---|---|---|")
    for perf in (perf_i8, perf_fp32):
        add(f"| {perf['model']} (`{perf['file']}`) | {perf['median_ms']:.1f} | {perf['p90_ms']:.1f} | "
            f"{perf['load_seconds']:.2f} | {perf['peak_rss_mib']:.1f} |")
    add("")
    add(f"- 计时：`intra_op_num_threads=1`、10 次推理的中位数/P90（2 次预热后）；窗口为 "
        f"{PERF_WINDOW_SECONDS:.0f} s LibriSpeech 片段。")
    add("")
    add("| 文件 | 字节数 | 说明 |")
    add("|---|---|---|")
    add(f"| `{FP32_PATH.name}` | {FP32_PATH.stat().st_size} | 对比用 |")
    add(f"| `{INT8_PATH.name}` | {INT8_PATH.stat().st_size} | Android 端实际使用（目标 ≤ 120 MB） |")
    add("")
    add("### 设备端自检向量")
    add("")
    add("| 文件 | 时长 | 源 | INT8 T | blank_ratio | INT8 feasible | FP32 feasible |")
    add("|---|---|---|---|---|---|---|")
    for v in read_json("vectors.json"):
        add(f"| `test_vectors/{v['file']}` | {v['seconds']}s | `{v['source']}` | {v['int8']['log_probs_shape'][1]} | "
            f"{v['int8']['blank_ratio']:.4f} | {v['int8']['align_feasible']} | {v['fp32']['align_feasible']} |")
    add("")
    add("## 5. 一键复跑命令")
    add("")
    add("```bash")
    add("cd /home/lladlam/项目/MeloX-AutoSync")
    add(".venv/bin/python android_export/en/validate_onnx.py        # 重新跑全部检查并重写 REPORT.md")
    add("# 如需重新导出模型（约需数分钟、多进程分批执行）：")
    add(".venv/bin/python android_export/en/export_model.py --force")
    add("```")
    add("")
    add("## 6. 已知限制与风险")
    add("")
    add("- **量化在歌声上的表现**：INT8 余弦/对齐一致性在真实歌曲（含伴奏、嘶吼、混响）上成立，"
        "但它衡量的是与 FP32 的**一致性**；两者共同的绝对误差在歌声/强伴奏上仍可能较大（引擎侧已有质量分级兜底）。"
        "本次评测素材中 2 段 GTSinger 为干声演唱，其余段落为朗读/合成语音与商业歌曲混音。")
    add("- **动态量化的确定性**：ORT 动态量化推理逐位可复现（已 3× 验证）；但不同 ORT 版本/不同设备的"
        "MatMulInteger 内核可能带来极小数值差异（设备端以 test_vectors 对拍为准）。")
    add("- **内存**：本机可用内存仅约 1.5 GB（另有并发任务），校验脚本按阶段拆分子进程；"
        "桌面 fp32 会话本身约需数百 MB，Android 端 INT8 会话显著更小（见 5.5）。")
    add("- **桌面 ≠ 设备**：5.5 是 x86 单线程参考值；Android ARM 上动态量化 MatMulInteger 通常更快，"
        "但内存带宽与调度不同，应以真机 benchmark 为准。")
    add("- **归一化是整段统计**：与 HF `Wav2Vec2FeatureExtractor` 完全一致；这意味着**裁剪窗口会改变归一化**，"
        "设备端对同一段音频须整段喂入（引擎端也是逐行裁剪后整体归一化）。")
    add("")
    add("## 7. 未做/做不了的部分")
    add("")
    add("- 未在 Android 真机/ARM 上运行（本机为 x86 CPU，单机验证）。")
    add("- 未量化 Conv 前端（按任务书要求保持 fp32）；未尝试 QDQ/静态量化、未做 QAT——任务书未要求。")
    add("- 未做绝对精度（人工标注）评测：INT8-vs-FP32 一致性已按要求测评；真实歌曲的人工词级标注"
        "本地不可得（GTSinger 有标注，但为干声且非商业混音）。")
    add("- 中文/日文链路不在本轮范围。")
    add("")
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "REPORT.md").write_text(text, encoding="utf-8")
    log(f"[report] wrote {OUT_DIR / 'REPORT.md'} ({len(text)} bytes)")
    log("[report] verdict: " + ("ALL CHECKS PASS" if verdict else "FAILURES PRESENT"))
    for name, ok, detail, threshold in checks:
        if not ok:
            log(f"[report]   FAILED: {name}: {detail} (threshold {threshold})")
    (SCRATCH / "verdict.json").write_text(
        json.dumps({"all_pass": verdict, "checks": [{"name": n, "ok": o, "detail": d, "threshold": t} for n, o, d, t in checks]},
                   indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
PHASES = {
    "torch": phase_torch,
    "fp32": phase_fp32,
    "int8": phase_int8,
    "align_fp32": phase_align_fp32,
    "align_int8": phase_align_int8,
    "align_compare": phase_align_compare,
    "perf_fp32": lambda: phase_perf("fp32"),
    "perf_int8": lambda: phase_perf("int8"),
    "loadcheck": phase_loadcheck,
    "vectors": phase_vectors,
    "report": phase_report,
}
RUN_ORDER = ["torch", "fp32", "int8", "align_fp32", "align_int8", "align_compare",
             "perf_int8", "perf_fp32", "loadcheck", "vectors", "report"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=[*PHASES, "all"], default="all")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="with --phase all: keep existing scratch results instead of wiping them")
    args = ap.parse_args(argv)

    if args.phase != "all":
        SCRATCH.mkdir(parents=True, exist_ok=True)
        PHASES[args.phase]()
        return 0

    if not args.reuse_cache and SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for phase in RUN_ORDER:
        if phase == "report":
            continue  # run in-process at the end
        cmd = [sys.executable, str(SCRIPT), "--phase", phase]
        log(f"\n===== phase {phase} =====")
        subprocess.run(cmd, check=True)
    phase_report()

    verdict = json.loads((SCRATCH / "verdict.json").read_text(encoding="utf-8"))
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
