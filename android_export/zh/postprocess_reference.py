#!/usr/bin/env python3
"""Independent reference implementation of the MeloX Android post-processing
for the Chinese (`funasr/fa-zh`) ONNX export.

This file is the **specification** the Android side must re-implement:

    us_alphas / us_peaks (ONNX, float32, [1, T3])
        -> per-token [start_ms, end_ms]

It is an independent implementation of FunASR's
``funasr.utils.timestamp_tools.ts_prediction_lfr6_standard`` (spec: job book
§2.4, constants below).  It is written in PyTorch so that the float semantics
are identical to the official code, but the control flow is written here from
the specification, not copied.

Constants (do not change):
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION  = 12        # upsampled frames
    upsample_rate       = 3         # T3 = 3 * T_enc
    TIME_RATE           = 10.0 * 6 / 1000 / 3 = 0.02 s per upsampled frame
    force_time_shift    = -1.5      # upsampled frames
    peak threshold      = 1.0 - 1e-4

Cross-check against the official implementation (requires ``funasr``):

    .venv/bin/python android_export/zh/postprocess_reference.py --check

which prints the per-token ``max|Δ|`` (target 0 ms) on every cached test case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

START_END_THRESHOLD = 5
MAX_TOKEN_DURATION = 12
UPSAMPLE_RATE = 3
TIME_RATE = 10.0 * 6 / 1000 / UPSAMPLE_RATE  # 0.02 s
FORCE_TIME_SHIFT = -1.5
PEAK_THRESHOLD = 1.0 - 1e-4

WORK = Path(__file__).resolve().parent / "_work"


# --------------------------------------------------------------------------- #
# cif_wo_hidden: the ONNX graph computes this with the closed form below.
# --------------------------------------------------------------------------- #
def cif_wo_hidden_loop(alphas: torch.Tensor, threshold: float = PEAK_THRESHOLD) -> torch.Tensor:
    """Official algorithm (sequential integrator), kept for cross-checking."""
    batch_size, len_time = alphas.size()
    integrate = torch.zeros([batch_size], dtype=alphas.dtype)
    list_fires = []
    for t in range(len_time):
        integrate = integrate + alphas[:, t]
        list_fires.append(integrate)
        fire_place = integrate >= threshold
        integrate = torch.where(fire_place, integrate - threshold, integrate)
    return torch.stack(list_fires, 1)


def cif_wo_hidden_closed_form(
    alphas: torch.Tensor, threshold: float = PEAK_THRESHOLD
) -> torch.Tensor:
    """Closed form of :func:`cif_wo_hidden_loop` (what the ONNX graph uses).

    With all alpha values < threshold (sigmoid * 0.25 - 0.01 <= 0.25), the
    integrator fires at most once per frame, so

        fires[t] = S[t] - threshold * floor(S[t-1] / threshold),  S = cumsum(alpha)

    i.e. the running sum minus one threshold per *previously* completed unit.
    """
    s = torch.cumsum(alphas, dim=1)
    k = torch.floor(s / threshold)  # completed units after frame t
    k_prev = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)
    return s - k_prev * threshold


# --------------------------------------------------------------------------- #
# ts_prediction_lfr6_standard
# --------------------------------------------------------------------------- #
def ts_prediction_lfr6_standard(
    us_alphas: torch.Tensor,
    us_peaks: torch.Tensor,
    char_list: list,
    vad_offset: float = 0.0,
    force_time_shift: float = FORCE_TIME_SHIFT,
    sil_in_str: bool = True,
    upsample_rate: int = UPSAMPLE_RATE,
):
    """Return (res_str, [[start_ms, end_ms], ...]) - one entry per char.

    ``us_alphas`` / ``us_peaks`` are the *truncated* model outputs
    (``[: T_enc * upsample_rate]``).  ``char_list`` are the text tokens.
    """
    if not len(char_list):
        return "", []
    time_rate = 10.0 * 6 / 1000 / upsample_rate
    if us_alphas.dim() == 2:
        alphas, peaks = us_alphas[0], us_peaks[0]
    else:
        alphas, peaks = us_alphas, us_peaks
    if char_list[-1] == "</s>":
        char_list = char_list[:-1]

    fire_place = torch.where(peaks >= 1.0 - 1e-4)[0].cpu().numpy() + force_time_shift

    if len(fire_place) != len(char_list) + 1:
        # re-normalise so that the alpha mass matches the token count
        alphas = alphas / (alphas.sum() / (len(char_list) + 1))
        peaks = cif_wo_hidden_loop(alphas.unsqueeze(0), threshold=1.0 - 1e-4)[0]
        fire_place = torch.where(peaks >= 1.0 - 1e-4)[0].cpu().numpy() + force_time_shift

    num_frames = peaks.shape[0]
    timestamp_list = []
    new_char_list = []

    if fire_place[0] > START_END_THRESHOLD:
        timestamp_list.append([0.0, fire_place[0] * time_rate])
        new_char_list.append("<sil>")

    for i in range(len(fire_place) - 1):
        new_char_list.append(char_list[i])
        if MAX_TOKEN_DURATION < 0 or fire_place[i + 1] - fire_place[i] <= MAX_TOKEN_DURATION:
            timestamp_list.append([fire_place[i] * time_rate, fire_place[i + 1] * time_rate])
        else:
            _split = fire_place[i] + MAX_TOKEN_DURATION
            timestamp_list.append([fire_place[i] * time_rate, _split * time_rate])
            timestamp_list.append([_split * time_rate, fire_place[i + 1] * time_rate])
            new_char_list.append("<sil>")

    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) * 0.5
        timestamp_list[-1][1] = _end * time_rate
        timestamp_list.append([_end * time_rate, num_frames * time_rate])
        new_char_list.append("<sil>")
    else:
        if len(timestamp_list) > 0:
            timestamp_list[-1][1] = num_frames * time_rate

    if vad_offset:
        for i in range(len(timestamp_list)):
            timestamp_list[i][0] += vad_offset / 1000.0
            timestamp_list[i][1] += vad_offset / 1000.0

    res_txt = ""
    for char, timestamp in zip(new_char_list, timestamp_list):
        if not sil_in_str and char == "<sil>":
            continue
        res_txt += "{} {} {};".format(
            char, str(timestamp[0] + 0.0005)[:5], str(timestamp[1] + 0.0005)[:5]
        )

    res = []
    for char, timestamp in zip(new_char_list, timestamp_list):
        if char != "<sil>":
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res_txt, res


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
def _load_cases():
    meta_path = WORK / "cases.json"
    ref_path = WORK / "refs.npz"
    if not meta_path.exists() or not ref_path.exists():
        raise SystemExit(
            f"no cached cases in {WORK}; run `python export_model.py` first "
            "(it writes _work/cases.json / _work/refs.npz)."
        )
    doc = json.loads(meta_path.read_text(encoding="utf-8"))
    cases = doc["cases"] if isinstance(doc, dict) else doc
    refs = np.load(ref_path)
    return cases, refs


def check_official(cases, refs, verbose=True):
    """Compare this implementation with the official FunASR implementation."""
    from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard as official

    worst = 0
    n_cases = 0
    n_tokens = 0
    for c in cases:
        key = c["name"]
        toks = list(c["tokens"])
        if not toks:
            continue
        us_a = torch.from_numpy(refs[f"case.{key}.us_alphas"])
        us_p = torch.from_numpy(refs[f"case.{key}.us_peaks"])
        t3 = int(np.asarray(refs[f"case.{key}.t3_out"]).reshape(-1)[0])
        a, p = us_a[0, :t3].clone(), us_p[0, :t3].clone()
        _, off = official(a.clone(), p.clone(), list(toks))
        _, mine = ts_prediction_lfr6_standard(a.clone(), p.clone(), list(toks))
        assert len(off) == len(mine) == len(toks), (len(off), len(mine), len(toks), key)
        for (s0, e0), (s1, e1) in zip(off, mine):
            worst = max(worst, abs(s0 - s1), abs(e0 - e1))
        n_cases += 1
        n_tokens += len(toks)
    if verbose:
        print(f"cases={n_cases} tokens={n_tokens} per-token max|Δ| vs official = {worst} ms")
    return worst, n_cases, n_tokens


def check_peaks(cases, refs, verbose=True):
    """Closed-form peaks vs the official sequential integrator on real alphas."""
    worst = 0.0
    n_mismatch = 0
    for c in cases:
        key = c["name"]
        if not c.get("tokens"):
            continue
        a = torch.from_numpy(refs[f"case.{key}.us_alphas"])[0]
        t3 = int(np.asarray(refs[f"case.{key}.t3_out"]).reshape(-1)[0])
        a = a[:t3]
        loop = cif_wo_hidden_loop(a.unsqueeze(0))[0]
        closed = cif_wo_hidden_closed_form(a.unsqueeze(0))[0]
        worst = max(worst, float((loop - closed).abs().max()))
        n_mismatch += int((loop >= 1.0 - 1e-4).sum().item() != (closed >= 1.0 - 1e-4).sum().item())
    if verbose:
        print(f"cif peaks: max|Δ| closed-form vs loop = {worst:.3e}, peak-count mismatches = {n_mismatch}")
    return worst, n_mismatch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="cross-check against official funasr")
    args = ap.parse_args()
    cases, refs = _load_cases()
    check_peaks(cases, refs)
    if args.check:
        check_official(cases, refs)


if __name__ == "__main__":
    main()
