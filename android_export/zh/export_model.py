#!/usr/bin/env python3
"""Export ``funasr/fa-zh`` (MonotonicAligner) to ONNX fp32 + INT8 for Android.

Job book: docs/ANDROID_ONNX_EXPORT_ZH.md

  * frontend (kaldi fbank + LFR(7,6) + CMVN) is baked into the graph
  * ``input_values`` float32 [1, N] raw 16 kHz mono PCM in [-1, 1], N >= 400
  * ``token_num``    int64   [1]  = text token count + 1
  * outputs ``us_alphas`` / ``us_peaks`` float32 [1, 3 * T_enc],
    T_enc = ceil(T_fbank / 6), T_fbank = 1 + (N - 400) // 160

The graph is exported with the legacy (TorchScript) exporter, opset 17,
single file, standard operators only.  ``us_peaks`` is the closed form of
FunASR's sequential ``cif_wo_hidden`` integrator (see postprocess_reference.py);
the sequential loop itself cannot be traced with a dynamic time axis.

Usage (from the repository root):

    .venv/bin/python android_export/zh/export_model.py            # refs + export
    .venv/bin/python android_export/zh/validate_onnx.py           # one-command rerun

This script never runs uv/pip and never writes outside android_export/zh/.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL_DIR = ROOT / "models" / "funasr" / "fa-zh"
WORK = HERE / "_work"
VECTORS = HERE / "test_vectors"
AUDIO_DIR = ROOT / "experiments" / "audio"

FLT_EPSILON = float(np.finfo(np.float32).eps)  # 1.1920929e-07, kaldi log floor
SAMPLE_RATE = 16000
FRAME_SHIFT_SAMPLES = 160
FRAME_LENGTH_SAMPLES = 400


# --------------------------------------------------------------------------- #
# minimal kaldi fbank, exportable (no torchaudio / kaldi-native-fbank in graph)
# --------------------------------------------------------------------------- #
def mel_scale(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def kaldi_mel_banks(
    num_bins: int = 80,
    window_size: int = 512,
    sample_freq: float = 16000.0,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> np.ndarray:
    """Kaldi ``MelBanks`` weights [num_bins, window_size // 2], float32.

    Mirrors kaldi/src/feat/mel-computations.cc (vtln_warp = 1.0).
    """
    num_fft_bins = window_size // 2
    fft_bin_width = sample_freq / window_size
    mel_low = mel_scale(low_freq)
    mel_high = mel_scale(high_freq if high_freq > 0 else sample_freq / 2)
    delta = (mel_high - mel_low) / (num_bins + 1)
    banks = np.zeros((num_bins, num_fft_bins), dtype=np.float64)
    for i in range(num_bins):
        left = mel_low + i * delta
        center = mel_low + (i + 1) * delta
        right = mel_low + (i + 2) * delta
        for j in range(num_fft_bins):
            mel = mel_scale(fft_bin_width * j)
            if left < mel < right:
                if mel <= center:
                    banks[i, j] = (mel - left) / (center - left)
                else:
                    banks[i, j] = (right - mel) / (right - center)
    return banks.astype(np.float32)


class KaldiFbankTorch(nn.Module):
    """Kaldi fbank (hamming, 80 mel, 25/10 ms, snip_edges, dither=0).

    Matches ``funasr.utils.fbank._fbank_knf`` (kaldi-native-fbank backend) to
    ~2e-4 per value; the real FFT is done with a folded DFT matrix because
    ``aten::fft_rfft`` is not ONNX-exportable at opset 17.
    """

    def __init__(
        self,
        sample_frequency: float = 16000.0,
        num_mel_bins: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        dither: float = 0.0,
        preemph: float = 0.97,
        remove_dc_offset: bool = True,
        window_type: str = "hamming",
        upsacle_samples: bool = True,
        energy_floor: float = 0.0,
        use_log: bool = True,
        use_power: bool = True,
    ):
        super().__init__()
        if dither != 0.0:
            raise ValueError("dither must be fixed to 0 for a deterministic graph")
        if window_type != "hamming":
            raise ValueError(f"unsupported window {window_type!r}")
        self.sample_frequency = float(sample_frequency)
        self.num_mel_bins = int(num_mel_bins)
        self.frame_length = int(sample_frequency * 0.001 * frame_length_ms)
        self.frame_shift = int(sample_frequency * 0.001 * frame_shift_ms)
        self.preemph = float(preemph)
        self.remove_dc_offset = bool(remove_dc_offset)
        self.upsacle_samples = bool(upsacle_samples)
        self.energy_floor = float(energy_floor) if energy_floor > 0 else FLT_EPSILON
        self.use_log = bool(use_log)
        self.use_power = bool(use_power)
        self.dither = 0.0

        fft_size = 1
        while fft_size < self.frame_length:
            fft_size *= 2
        self.fft_size = fft_size
        if fft_size != 512:
            raise ValueError("unexpected fft size")

        i = np.arange(self.frame_length, dtype=np.float64)
        win = 0.54 - 0.46 * np.cos(2.0 * math.pi / (self.frame_length - 1) * i)

        n = np.arange(self.frame_length, dtype=np.float64)[None, :]
        k = np.arange(fft_size // 2, dtype=np.float64)[:, None]
        ang = 2.0 * math.pi * k * n / fft_size
        # parameters (not buffers) -> ONNX initializers, easy to audit/skip.
        # stored pre-transposed so the exporter keeps the initializer name
        # (a ``.t()`` call would be constant-folded into an anonymous tensor).
        self.dft_re = nn.Parameter(
            torch.from_numpy((win[None, :] * np.cos(ang)).T.astype(np.float32)), requires_grad=False
        )
        self.dft_im = nn.Parameter(
            torch.from_numpy((win[None, :] * -np.sin(ang)).T.astype(np.float32)), requires_grad=False
        )
        self.mel_banks = nn.Parameter(
            torch.from_numpy(
                kaldi_mel_banks(num_mel_bins, fft_size, float(sample_frequency)).T
            ),
            requires_grad=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform[0] if waveform.dim() == 2 else waveform
        if self.upsacle_samples:
            x = x * float(1 << 15)
        n_samples = x.shape[0]
        num_frames = (n_samples - self.frame_length) // self.frame_shift + 1
        starts = torch.arange(num_frames, dtype=torch.int64, device=x.device) * self.frame_shift
        idx = starts[:, None] + torch.arange(
            self.frame_length, dtype=torch.int64, device=x.device
        )[None, :]
        frames = x[idx]
        if self.remove_dc_offset:
            frames = frames - frames.mean(dim=-1, keepdim=True)
        if self.preemph != 0.0:
            frames = frames - self.preemph * torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
        re = frames @ self.dft_re
        im = frames @ self.dft_im
        power = re * re + im * im if self.use_power else torch.sqrt(re * re + im * im)
        mel = power @ self.mel_banks
        if self.use_log:
            mel = torch.log(torch.clamp(mel, min=self.energy_floor))
        return mel


class ZhFrontend(nn.Module):
    """fbank + LFR(7,6) + CMVN(am.mvn): the frontend baked into the ONNX graph.

    ``apply_lfr`` semantics: left pad (m-1)//2 copies of frame 0, stride n,
    repeat the last frame for the tail -> equivalent to clamping the gather
    indices to [0, T-1].
    """

    def __init__(self, fbank: KaldiFbankTorch, cmvn: torch.Tensor, lfr_m: int, lfr_n: int):
        super().__init__()
        self.fbank = fbank
        self.lfr_m = int(lfr_m)
        self.lfr_n = int(lfr_n)
        cmvn = cmvn.to(torch.float32)
        self.cmvn_mean = nn.Parameter(cmvn[0:1].clone(), requires_grad=False)  # <AddShift>
        self.cmvn_var = nn.Parameter(cmvn[1:2].clone(), requires_grad=False)  # <Rescale>

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        fbank = self.fbank(waveform)  # [T, 80]
        t = fbank.shape[0]
        t_lfr = (t + self.lfr_n - 1) // self.lfr_n
        j = torch.arange(t_lfr, dtype=torch.int64, device=fbank.device)[:, None]
        m = torch.arange(self.lfr_m, dtype=torch.int64, device=fbank.device)[None, :]
        idx = j * self.lfr_n + m - (self.lfr_m - 1) // 2
        idx = torch.clamp(idx, 0, t - 1)
        lfr = fbank[idx.reshape(-1)].reshape(t_lfr, self.lfr_m * fbank.shape[-1])
        return (lfr + self.cmvn_mean) * self.cmvn_var


# --------------------------------------------------------------------------- #
# ONNX wrapper
# --------------------------------------------------------------------------- #
class FaZhOnnx(nn.Module):
    """raw PCM -> (us_alphas, us_peaks); dither is fixed to 0.

    The SANM encoder is driven with ``mask=None`` (batch 1, no padding: the
    all-ones mask is an exact identity in every layer it feeds).
    """

    def __init__(self, auto_model, debug: bool = False):
        super().__init__()
        fe = auto_model.kwargs["frontend"]
        fe.dither = 0.0
        if fe.fs != SAMPLE_RATE or fe.lfr_m != 7 or fe.lfr_n != 6 or fe.n_mels != 80:
            raise ValueError(f"unexpected frontend config: {fe.fs} {fe.lfr_m} {fe.lfr_n} {fe.n_mels}")
        if fe.cmvn is None:
            raise ValueError("frontend has no CMVN")
        fbank = KaldiFbankTorch(
            sample_frequency=fe.fs,
            num_mel_bins=fe.n_mels,
            frame_length_ms=fe.frame_length,
            frame_shift_ms=fe.frame_shift,
            dither=0.0,
            window_type=fe.window,
            upsacle_samples=fe.upsacle_samples,
        )
        self.frontend = ZhFrontend(fbank, fe.cmvn, fe.lfr_m, fe.lfr_n)
        net = auto_model.model
        self.encoder = net.encoder
        self.predictor = net.predictor
        self.debug = debug
        self.scale = float(self.encoder.output_size()) ** 0.5

    def _encode(self, input_values: torch.Tensor):
        feats = self.frontend(input_values)  # [T_lfr, 560]
        xs = feats[None, :, :] * self.scale
        xs = self.encoder.embed(xs)
        xs = self.encoder.encoders0(xs, None)[0]
        xs = self.encoder.encoders(xs, None)[0]
        xs = self.encoder.after_norm(xs)
        return feats, xs

    def _predict(self, hidden: torch.Tensor, token_num: torch.Tensor):
        p = self.predictor
        context = hidden.transpose(1, 2)
        output2 = p.upsample_cnn(context)  # cnn_blstm upsampler, stride 3
        output2 = output2.transpose(1, 2)
        output2, _ = p.blstm(output2)
        alphas2 = torch.sigmoid(p.cif_output2(output2))
        alphas2 = torch.relu(alphas2 * p.smooth_factor2 - p.noise_threshold2)
        alphas2 = alphas2.squeeze(-1)
        token_num_f = token_num.to(alphas2.dtype)
        alphas2 = alphas2 * (token_num_f / alphas2.sum(-1)).repeat(1, alphas2.size(1))
        us_alphas = alphas2
        # closed-form cif_wo_hidden(us_alphas, threshold - 1e-4)
        thr = float(p.threshold) - 1e-4
        s = torch.cumsum(us_alphas, dim=1)
        k = torch.floor(s / thr)
        k_prev = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)
        us_peaks = s - k_prev * thr
        return us_alphas, us_peaks

    def forward(self, input_values: torch.Tensor, token_num: torch.Tensor):
        feats, enc = self._encode(input_values)
        us_alphas, us_peaks = self._predict(enc, token_num)
        if self.debug:
            return us_alphas, us_peaks, feats, enc
        return us_alphas, us_peaks


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #
def load_audio_16k(path: Path) -> np.ndarray:
    """Decode to mono float32 16 kHz (soundfile for wav, ffmpeg otherwise)."""
    if path.suffix.lower() in (".wav", ".flac"):
        import soundfile as sf

        x, sr = sf.read(str(path), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{path}: sample rate {sr} != {SAMPLE_RATE}")
        return np.ascontiguousarray(x, dtype=np.float32)
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").copy()


def load_normalize_cjk():
    """The engine's normaliser (`src/autosync/lyrics/normalize.py`), read-only."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_autosync_normalize", ROOT / "src" / "autosync" / "lyrics" / "normalize.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize_cjk


LRC_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")


def parse_lrc(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LRC_RE.match(line.strip())
        if not m:
            continue
        t = int(m.group(1)) * 60 + float(m.group(2))
        text = m.group(3).strip()
        if text:
            out.append((t, text))
    return out


def read_audio_text(name: str) -> str:
    lrc = AUDIO_DIR / f"{name}.lrc"
    lines = parse_lrc(lrc)
    if len(lines) != 1:
        raise ValueError(f"expected one line in {lrc}, got {len(lines)}")
    return lines[0][1]


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #
SONGS = [
    # alias, lrc file, how many pure-CJK lines to take
    ("songA", "不安灵魂收容所-ChiliChill乐团.lrc", 7),
    ("songB", "提瓦特民谣-宴宁,XY大甘蔗,柳知萧,闫夜桥,陶典,孙晔.lrc", 8),
    ("songC", "SAN値直葬 (feat. 初音ミク)-STEAKA,初音ミク.lrc", 6),
]


def build_audios(normalize_cjk) -> dict:
    """Return {name: audio record} for every evaluation clip."""
    audios: dict[str, dict] = {}
    songs: dict[str, dict] = {}

    def add(name, samples, kind, source, text=None, note=""):
        audios[name] = {
            "name": name,
            "samples": np.ascontiguousarray(samples, dtype=np.float32),
            "kind": kind,
            "source": source,
            "text_raw": text,
            "text_normalized": normalize_cjk(text) if text else None,
            "note": note,
        }

    # repository audio (local, gitignored)
    add(
        "fleurs1",
        load_audio_16k(AUDIO_DIR / "fleurs_zh_sample.wav"),
        "human speech",
        "experiments/audio/fleurs_zh_sample.wav (FLEURS zh, 7.44 s)",
        text=read_audio_text("fleurs_zh_sample"),
        note="real human speech",
    )
    add(
        "fleurs2",
        load_audio_16k(AUDIO_DIR / "fleurs_zh_sample2.wav"),
        "human speech",
        "experiments/audio/fleurs_zh_sample2.wav (FLEURS zh, 7.56 s)",
        note="no transcript available locally: frontend-equivalence only",
    )
    add(
        "synthetic",
        load_audio_16k(AUDIO_DIR / "synthetic_zh.wav"),
        "tts",
        "experiments/audio/synthetic_zh.wav (TTS cmn, 12.46 s)",
        text=read_audio_text("synthetic_zh"),
        note="ground truth char timings in synthetic_zh.ref.json",
    )

    # local songs: audio is used as *input* only; lyrics are never quoted into
    # any deliverable (see REPORT.md notes)
    test_dir = ROOT / "Test"
    mp3s = sorted(test_dir.glob("*.mp3"), key=lambda p: -len(p.stem)) if test_dir.exists() else []
    for alias, lrc_name, want in SONGS:
        lrc_path = test_dir / lrc_name
        mp3 = None
        for cand in mp3s:
            stem = lrc_path.stem
            if stem == cand.stem or stem.startswith(cand.stem):
                mp3 = cand
                break
        if not lrc_path.exists() or mp3 is None:
            print(f"  [songs] {alias}: lrc={lrc_path.exists()} mp3={mp3}")
            continue
        lines = parse_lrc(lrc_path)
        picked = 0
        for i, (t0, text) in enumerate(lines):
            if picked >= want:
                break
            norm = normalize_cjk(text)
            if norm != text or len(norm) < 3:
                continue
            t1 = lines[i + 1][0] if i + 1 < len(lines) else t0 + 4.0
            dur = t1 - t0
            if not (1.5 <= dur <= 10.0):
                continue
            if alias not in songs:
                songs[alias] = load_audio_16k(mp3)
            audio = songs[alias]
            a = int(max(0.0, t0 - 0.30) * SAMPLE_RATE)
            b = int(min(len(audio) / SAMPLE_RATE, t1 + 0.20) * SAMPLE_RATE)
            seg = audio[a:b]
            # trim trailing silence runs so the model sees a compact clip
            name = f"{alias}_l{i:03d}"
            add(
                name,
                seg,
                "singing",
                f"Test/ (local song, alias {alias}), LRC line #{i}, "
                f"t={t0:.2f}..{t1:.2f}s, window {a / SAMPLE_RATE:.2f}..{b / SAMPLE_RATE:.2f}s",
                text=text,
                note="third-party lyrics: evaluation input only, never quoted",
            )
            picked += 1
    return audios


# --------------------------------------------------------------------------- #
# reference computation (PyTorch / FunASR)
# --------------------------------------------------------------------------- #
def compute_pytorch_refs(auto, audios, cases, refs=None):
    """Fill refs[<audio>] and refs[<case>] from the PyTorch model."""
    from funasr.utils.load_utils import extract_fbank

    fix_dither(auto)
    net = auto.model
    frontend = auto.kwargs["frontend"]
    refs = {} if refs is None else refs

    for name, a in audios.items():
        audio = a["samples"]
        refs[f"audio.{name}"] = audio
        feats, feats_len = extract_fbank(
            [torch.from_numpy(audio)], data_type="sound", frontend=frontend
        )
        refs[f"feats.{name}"] = feats.numpy()
        refs[f"feats_len.{name}"] = np.array([int(feats_len.item())], dtype=np.int64)
        with torch.inference_mode():
            enc, enc_len = net.encode(feats, feats_len)
        refs[f"enc.{name}"] = enc.numpy()
        refs[f"enc_len.{name}"] = np.array([int(enc_len.item())], dtype=np.int64)

    for c in cases:
        name = c["name"]
        enc = torch.from_numpy(refs[f"enc.{c['audio']}"])
        enc_len = torch.from_numpy(refs[f"enc_len.{c['audio']}"])
        token_num = torch.tensor([int(c["token_num"])], dtype=torch.int64)
        with torch.inference_mode():
            _, _, us_a, us_p = net.calc_predictor_timestamp(enc, enc_len, token_num)
        t3 = int(enc_len[0].item()) * 3
        refs[f"case.{name}.us_alphas"] = us_a.numpy()
        refs[f"case.{name}.us_peaks"] = us_p.numpy()
        refs[f"case.{name}.t3_out"] = np.array([t3], dtype=np.int64)
        c["t3_out"] = t3
        c["encoder_shape"] = list(enc.shape)
        if c.get("tokens"):
            from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard

            a_flat = us_a[0, :t3].clone()
            p_flat = us_p[0, :t3].clone()
            _, ts = ts_prediction_lfr6_standard(a_flat, p_flat, list(c["tokens"]))
            refs[f"case.{name}.official_ts"] = np.array(ts, dtype=np.int64)
        if c.get("run_generate"):
            a = audios[c["audio"]]
            res = auto.generate(
                input=(a["samples"], c["text_norm"]),
                data_type=("sound", "text"),
                fs=SAMPLE_RATE,
                batch_size=1,
                disable_pbar=True,
            )
            item = res[0]
            gen_tokens = item["text"].split()
            gen_ts = np.array(item["timestamp"], dtype=np.int64)
            refs[f"case.{name}.generate_ts"] = gen_ts
            c["generate_tokens"] = gen_tokens
    return refs


def compute_dither_evidence(auto):
    """dither=1.0 vs dither=0 reproducibility: features and timestamps.

    Note: ``AutoModel.generate`` calls ``_reset_runtime_configs()`` which deep
    copies the kwargs snapshot taken at construction time, so a runtime
    ``frontend.dither = 0`` on ``auto.kwargs`` is *discarded* on every call.
    The baseline snapshot must be patched instead (see ``fix_dither``).
    """
    from funasr.utils.load_utils import extract_fbank

    frontend = auto.kwargs["frontend"]
    a = load_audio_16k(AUDIO_DIR / "fleurs_zh_sample.wav")
    text = read_audio_text("fleurs_zh_sample")
    ev = {}

    def run_gen(times, baseline_dither):
        patch_baseline_dither(auto, baseline_dither)
        out = []
        for _ in range(times):
            res = auto.generate(
                input=(a, text), data_type=("sound", "text"), fs=SAMPLE_RATE,
                batch_size=1, disable_pbar=True,
            )
            out.append(np.array(res[0]["timestamp"], dtype=np.int64))
        return out

    def run_fbank(times, dither):
        frontend.dither = dither
        feats = []
        for _ in range(times):
            f, _ = extract_fbank([torch.from_numpy(a)], frontend=frontend)
            feats.append(f.numpy())
        return feats

    f0 = run_fbank(3, 0.0)
    ev["fbank_dither0_max_abs_diff"] = float(
        max(np.abs(f0[i] - f0[0]).max() for i in range(len(f0)))
    )
    f1 = run_fbank(3, 1.0)
    ev["fbank_dither1_max_abs_diff"] = float(
        max(np.abs(f1[i] - f1[0]).max() for i in range(len(f1)))
    )
    ev["fbank_dither1_pairwise_max_abs_diff"] = [
        float(np.abs(f1[i] - f1[j]).max()) for i in range(len(f1)) for j in range(i + 1, len(f1))
    ]
    t0 = run_gen(5, 0.0)
    ev["generate_dither0_ts_max_abs_diff_ms"] = int(
        max(np.abs(t0[i] - t0[0]).max() for i in range(len(t0)))
    )
    t1 = run_gen(5, 1.0)
    ev["generate_dither1_ts_pairwise_max_abs_diff_ms"] = int(
        max(np.abs(t1[i] - t1[j]).max() for i in range(len(t1)) for j in range(i + 1, len(t1)))
    )
    ev["generate_dither1_vs_dither0_ts_max_abs_diff_ms"] = int(
        max(np.abs(t1[i] - t0[0]).max() for i in range(len(t1)))
    )
    ev["generate_dither1_ts_runs"] = [t.tolist() for t in t1]
    ev["generate_dither0_ts_runs"] = [t.tolist() for t in t0]
    patch_baseline_dither(auto, 0.0)
    frontend.dither = 0.0
    return ev


def patch_baseline_dither(auto, value: float):
    """FunASR resets ``auto.kwargs`` from a deep-copied baseline on every
    ``generate`` call; patch both the live kwargs and the baseline snapshot."""
    base = getattr(auto, "_base_kwargs_map", {}) or {}
    for holder in (auto.kwargs, base.get("kwargs", {})):
        fe = holder.get("frontend") if isinstance(holder, dict) else None
        if fe is not None:
            fe.dither = float(value)


def fix_dither(auto):
    """Make the *whole* inference path (including ``generate``) use dither=0."""
    patch_baseline_dither(auto, 0.0)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_fp32_full(wrapper: FaZhOnnx, path: Path):
    """Export with 4 outputs (delivered 2 + 2 debug) so that the delivered file
    is produced by graph surgery from the same traced graph."""
    dummy = (torch.zeros(1, 4000, dtype=torch.float32), torch.tensor([8], dtype=torch.int64))
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        input_names=["input_values", "token_num"],
        output_names=["us_alphas", "us_peaks", "lfr_feats", "encoder_out"],
        dynamic_axes={
            "input_values": {1: "n"},
            "us_alphas": {1: "t3"},
            "us_peaks": {1: "t3"},
            "lfr_feats": {0: "t_lfr"},
            "encoder_out": {1: "t_enc"},
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )


def extract_submodel(src: Path, dst: Path, outputs: list[str], inputs: list[str]):
    import onnx

    onnx.utils.extract_model(str(src), str(dst), inputs, outputs)


def enforce_io_shapes(path: Path):
    """Pin the declared I/O shapes to the contract.

    The TorchScript exporter names the *output* batch axis symbolically
    (e.g. ``Subus_peaks_dim_0``); pin it to the literal 1 that is actually
    produced, so the graph matches manifest.json (``[1, null]``).
    """
    import onnx

    m = onnx.load(str(path))

    def set_dims(value_info, dims):
        tt = value_info.type.tensor_type
        del tt.shape.dim[:]
        for d in dims:
            dim = tt.shape.dim.add()
            if isinstance(d, int):
                dim.dim_value = d
            else:
                dim.dim_param = d

    for v in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
        if v.name == "input_values":
            set_dims(v, [1, "n"])
        elif v.name == "token_num":
            set_dims(v, [1])
        elif v.name in ("us_alphas", "us_peaks"):
            set_dims(v, [1, "t3"])
    onnx.save(m, str(path))


def quantize_int8(src: Path, dst: Path, excluded: list[str]):
    """Dynamic INT8 quantisation, MatMul/Gemm only.

    The baked frontend multiplies by the DFT / mel matrices (MatMul with a
    constant weight).  Those must stay fp32; they are identified by their
    initializer names (and, as a fallback, by their unique shapes) and the
    consuming nodes are excluded.  The result is verified afterwards.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    m = onnx.load(str(src))
    front_init_names = {
        i.name
        for i in m.graph.initializer
        if i.name.endswith(("dft_re", "dft_im", "mel_banks"))
    }
    if len(front_init_names) != 3:  # fall back to the unique weight shapes
        front_init_names = {
            i.name for i in m.graph.initializer if list(i.dims) in ([400, 256], [256, 80])
        }
    if len(front_init_names) != 3:
        raise RuntimeError(
            f"could not identify the 3 frontend matmul weights "
            f"(names={sorted(front_init_names)}); refusing to quantise blindly"
        )
    dft_nodes = [
        n.name for n in m.graph.node if n.op_type == "MatMul" and any(i in front_init_names for i in n.input)
    ]
    if len(dft_nodes) != 3:
        raise RuntimeError(f"expected 3 frontend MatMul nodes, found {dft_nodes}")
    nodes_to_exclude = excluded + dft_nodes
    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_exclude=nodes_to_exclude,
        # Per-channel weight scales. Measured with the validation harness on the
        # same 34 cases / 33 timestamp lines: end-to-end timestamp MAE drops
        # 47.76 ms -> 19.96 ms (now inside the <= 20 ms gate) and us_alphas
        # cosine mean/min rise 0.9609/0.8375 -> 0.9888/0.9127, for +0.43 MB.
        # The CIF head's peaks stay the weak spot either way (cosine min 0.55);
        # see REPORT.md 4.3.
        per_channel=True,
        reduce_range=False,
    )
    # verify the frontend weights survived quantization untouched
    q = onnx.load(str(dst), load_external_data=False)
    q_init = {i.name: i for i in q.graph.initializer}
    for name in front_init_names:
        if name not in q_init:
            raise RuntimeError(f"frontend weight {name} missing after quantization")
        assert q_init[name].data_type == onnx.TensorProto.FLOAT, f"{name} was quantised"
    return {"excluded_nodes": dft_nodes, "frontend_weights": sorted(front_init_names)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def choose_vector_windows(audios, cases, refs):
    """Pick 2-5 s self-check windows with text from repository material only.

    vector1: first clause of the FLEURS line, window cut just after the clause
             (boundary taken from the PyTorch alignment, text is known a priori)
    vector2/vector3: windows of synthetic_zh.wav whose text comes from the
             known TTS ground-truth timings (`synthetic_zh.ref.json`).
    """
    specs = []

    # --- vector1: FLEURS first clause ------------------------------------- #
    fleurs = audios["fleurs1"]
    norm = fleurs["text_normalized"]
    case = next(c for c in cases if c["audio"] == "fleurs1")
    ts = refs[f"case.{case['name']}.official_ts"]
    k = 0
    while k < len(ts) - 1 and (ts[k][1] - ts[0][1]) < 2000 - 350:
        k += 1
    end_ms = int(ts[k][1])
    t_end = min(5.0, (end_ms + 350) / 1000.0)
    t_end = max(2.05, t_end)
    specs.append(
        {
            "name": "vec1",
            "audio": "fleurs1",
            "samples": fleurs["samples"][: int(t_end * SAMPLE_RATE)],
            "text": norm[: k + 1],
            "source": f"experiments/audio/fleurs_zh_sample.wav [0.000, {t_end:.3f}] s",
            "note": f"real human speech; first {k + 1} chars of the FLEURS line",
        }
    )

    # --- vector2/vector3: synthetic_zh TTS windows ------------------------ #
    syn = audios["synthetic"]
    syn_norm = syn["text_normalized"]
    for idx, (t0, t1, a, b) in enumerate([(0.2, 4.6, 0, 4), (8.3, 12.4, 8, 12)]):
        specs.append(
            {
                "name": f"vec{idx + 2}",
                "audio": "synthetic",
                "samples": syn["samples"][int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)],
                "text": syn_norm[a:b],
                "source": f"experiments/audio/synthetic_zh.wav [{t0:.2f}, {t1:.2f}] s",
                "note": "TTS cmn speech; window chosen from synthetic_zh.ref.json ground truth",
            }
        )
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-refs", action="store_true", help="recompute PyTorch references")
    ap.add_argument("--refs-only", action="store_true", help="stop after references")
    ap.add_argument("--no-validate", action="store_true", help="do not run validate_onnx.py")
    ap.add_argument("--skip-export", action="store_true", help="references only, keep ONNX as is")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    WORK.mkdir(parents=True, exist_ok=True)
    VECTORS.mkdir(parents=True, exist_ok=True)

    refs_path = WORK / "refs.npz"
    cases_path = WORK / "cases.json"

    if refs_path.exists() and cases_path.exists() and not args.force_refs:
        print(f"cached references present: {refs_path} (use --force-refs to recompute)")
        run_validation()
        return

    t0 = time.time()
    from funasr import AutoModel

    print(f"[{time.time() - t0:7.1f}s] funasr imported")
    auto = AutoModel(
        model=str(MODEL_DIR),
        device="cpu",
        ncpu=1,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
    )
    print(f"[{time.time() - t0:7.1f}s] AutoModel loaded")
    fix_dither(auto)
    print("frontend dither fixed to", auto.kwargs["frontend"].dither, "(live + baseline snapshot)")

    normalize_cjk = load_normalize_cjk()

    print("dither evidence ...")
    dither_evidence = compute_dither_evidence(auto)
    (WORK / "dither_evidence.json").write_text(
        json.dumps(dither_evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "  fbank dither1 max|d|={fbank_dither1_max_abs_diff:.4f} "
        "dither0 max|d|={fbank_dither0_max_abs_diff}".format(**dither_evidence)
    )

    audios = build_audios(normalize_cjk)

    # ---- robustness audios ------------------------------------------------ #
    fleurs = audios["fleurs1"]["samples"]
    audios["silence4k"] = {
        "name": "silence4k",
        "samples": np.zeros(4000, dtype=np.float32),
        "kind": "robustness",
        "source": "generated: 0.25 s of digital silence",
        "text_raw": None,
        "text_normalized": None,
        "note": "all-zero input",
    }
    audios["short400"] = {
        "name": "short400",
        "samples": fleurs[:400].copy(),
        "kind": "robustness",
        "source": "experiments/audio/fleurs_zh_sample.wav [0, 400 samples]",
        "text_raw": None,
        "text_normalized": None,
        "note": "minimum contract length",
    }
    long_samples = np.concatenate(
        [
            np.zeros(int(3.0 * SAMPLE_RATE), dtype=np.float32),
            audios["fleurs1"]["samples"],
            audios["fleurs2"]["samples"],
            audios["synthetic"]["samples"],
            audios["fleurs1"]["samples"],
        ]
    )
    audios["long30"] = {
        "name": "long30",
        "samples": long_samples,
        "kind": "robustness",
        "source": "3 s silence + fleurs1 + fleurs2 + synthetic + fleurs1",
        "text_raw": None,
        "text_normalized": None,
        "note": f"{len(long_samples) / SAMPLE_RATE:.2f} s long input",
    }

    # ---- alignment cases -------------------------------------------------- #
    tokenizer = auto.kwargs["tokenizer"]

    def tokens_of(text_norm: str) -> list[str]:
        ids = tokenizer.encode(text_norm)
        return tokenizer.ids2tokens(ids)

    cases: list[dict] = []
    for name, a in audios.items():
        if not a["text_normalized"]:
            continue
        toks = tokens_of(a["text_normalized"])
        cases.append(
            {
                "name": name,
                "audio": name,
                "tokens": toks,
                "token_num": len(toks) + 1,
                "text_norm": a["text_normalized"],
                "text_raw": a["text_raw"],
                "run_generate": True,
                "kind": a["kind"],
                "source": a["source"],
            }
        )

    # placeholder-text robustness cases
    cases.append(
        {
            "name": "silence4k_tn5",
            "audio": "silence4k",
            "tokens": ["测"] * 4,
            "token_num": 5,
            "text_norm": "测测测测",
            "text_raw": None,
            "run_generate": False,
            "kind": "robustness",
            "source": "placeholder text, digital silence",
        }
    )
    cases.append(
        {
            "name": "short400_tn2",
            "audio": "short400",
            "tokens": ["测"],
            "token_num": 2,
            "text_norm": "测",
            "text_raw": None,
            "run_generate": False,
            "kind": "robustness",
            "source": "placeholder text, 400 samples",
        }
    )
    cases.append(
        {
            "name": "long30_tn13",
            "audio": "long30",
            "tokens": ["测"] * 12,
            "token_num": 13,
            "text_norm": "测" * 12,
            "text_raw": None,
            "run_generate": False,
            "kind": "robustness",
            "source": "placeholder text, >=30 s input",
        }
    )
    for tn in (1, 2, 50, 200, 201):
        cases.append(
            {
                "name": f"synthetic_tn{tn}",
                "audio": "synthetic",
                "tokens": ["测"] * (tn - 1),
                "token_num": tn,
                "text_norm": "测" * (tn - 1),
                "text_raw": None,
                "run_generate": False,
                "kind": "robustness",
                "source": "placeholder text, token_num boundary",
            }
        )

    refs = compute_pytorch_refs(auto, audios, cases)
    print(f"[{time.time() - t0:7.1f}s] references computed: {len(cases)} cases")

    # ---- test vectors ----------------------------------------------------- #
    vec_specs = choose_vector_windows(audios, cases, refs)
    vec_audios = {}
    vec_cases = []
    for s in vec_specs:
        norm = normalize_cjk(s["text"])
        if norm != s["text"]:
            raise ValueError(f"vector text is not CJK-only: {s['text']!r}")
        toks = tokens_of(norm)
        vec_audios[s["name"]] = {
            "name": s["name"],
            "samples": np.ascontiguousarray(s["samples"], dtype=np.float32),
            "kind": "test vector",
            "source": s["source"],
            "text_raw": s["text"],
            "text_normalized": norm,
            "note": s["note"],
        }
        vec_cases.append(
            {
                "name": s["name"],
                "audio": s["name"],
                "tokens": toks,
                "token_num": len(toks) + 1,
                "text_norm": norm,
                "text_raw": s["text"],
                "run_generate": True,
                "kind": "test vector",
                "source": s["source"],
                "vector": True,
            }
        )
        d = len(s["samples"]) / SAMPLE_RATE
        assert 2.0 <= d <= 5.0, f"{s['name']}: {d:.2f} s outside 2-5 s"
    refs = compute_pytorch_refs(auto, vec_audios, vec_cases, refs=refs)
    audios.update(vec_audios)
    cases.extend(vec_cases)
    print(f"[{time.time() - t0:7.1f}s] test vectors: " +
          ", ".join(f"{s['name']}={len(s['samples'])/SAMPLE_RATE:.2f}s" for s in vec_specs))

    np.savez(refs_path, **refs)
    (WORK / "cases.json").write_text(
        json.dumps({"cases": cases, "audios": {k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                                               for k, v in audios.items()}},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[{time.time() - t0:7.1f}s] refs saved: {refs_path.stat().st_size / 1e6:.1f} MB")

    # post-processing cross-checks against the official implementation (funasr
    # is already imported here; validate_onnx.py reuses these caches)
    try:
        from postprocess_reference import check_official, check_peaks

        worst, n_cases, n_tokens = check_official(cases, refs)
        (WORK / "postprocess_check.json").write_text(
            json.dumps({"max_abs_ms": worst, "n_cases": n_cases, "n_tokens": n_tokens}, indent=2),
            encoding="utf-8",
        )
        pc_worst, pc_mismatch = check_peaks(cases, refs)
        (WORK / "peaks_closed_form.json").write_text(
            json.dumps({"max_abs": pc_worst, "n_mismatch": pc_mismatch}, indent=2),
            encoding="utf-8",
        )
        print(f"[{time.time() - t0:7.1f}s] postprocess check: max|Δ|={worst} ms over {n_tokens} tokens")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: postprocess cross-check failed: {exc!r}")

    if args.refs_only or args.skip_export:
        if not args.no_validate:
            del auto
            gc.collect()
            run_validation()
        return

    print(f"[{time.time() - t0:7.1f}s] exporting ONNX ...")
    wrapper = FaZhOnnx(auto, debug=True).eval()
    full = WORK / "fa-zh-fp32-full.onnx"
    export_fp32_full(wrapper, full)
    del wrapper
    gc.collect()
    del auto
    gc.collect()
    print(f"[{time.time() - t0:7.1f}s] PyTorch model released")

    fp32 = HERE / "fa-zh-fp32.onnx"
    extract_submodel(full, fp32, ["us_alphas", "us_peaks"], ["input_values", "token_num"])
    enforce_io_shapes(fp32)
    print(f"[{time.time() - t0:7.1f}s] fp32 exported: {fp32} {fp32.stat().st_size} bytes")

    fe_out = WORK / "frontend-only.onnx"
    enc_out = WORK / "encoder-only.onnx"
    extract_submodel(full, fe_out, ["lfr_feats"], ["input_values"])
    extract_submodel(full, enc_out, ["encoder_out"], ["input_values", "token_num"])
    print("debug sub-graphs written")

    int8 = HERE / "fa-zh-int8.onnx"
    excluded = quantize_int8(fp32, int8, excluded=[])
    print(f"[{time.time() - t0:7.1f}s] int8 exported: {int8} {int8.stat().st_size} bytes")
    (WORK / "quant_nodes_excluded.json").write_text(json.dumps(excluded, indent=1))

    if not args.no_validate:
        run_validation()


def run_validation():
    import validate_onnx

    validate_onnx.main()


# --------------------------------------------------------------------------- #
# performance mode (clean process, ONNX only)
# --------------------------------------------------------------------------- #
def perf_main(onnx_path: str, window_s: float = 12.0, runs: int = 10):
    import onnxruntime as ort
    import resource

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    audio = load_audio_16k(AUDIO_DIR / "fleurs_zh_sample.wav")
    x = np.zeros((1, int(window_s * SAMPLE_RATE)), dtype=np.float32)
    x[0, : len(audio)] = audio[: len(x[0])]
    tn = np.array([16], dtype=np.int64)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {"input_values": x, "token_num": tn})
        times.append((time.perf_counter() - t) * 1000.0)
    times_arr = np.array(times)
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {
        "onnx": onnx_path,
        "window_s": window_s,
        "runs": runs,
        "median_ms": float(np.median(times_arr)),
        "p90_ms": float(np.percentile(times_arr, 90)),
        "min_ms": float(times_arr.min()),
        "peak_rss_mb": rss_kb / 1024.0,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--perf":
        perf_main(sys.argv[2])
    else:
        main()
