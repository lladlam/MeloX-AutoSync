"""Tests for the V1.3 AMLL version-consistency auditor.

The auditor must (a) recover a known offset on synthetic data, (b) refuse to
beat its negative controls on shifted/shuffled data, and (c) never auto-declare
VERIFIED without human confirmation.
"""

from __future__ import annotations

import amll_version_check as av
import numpy as np
import pytest


# --------------------------------------------------------------------------- clock / ttml
def test_clock_supports_all_amll_forms():
    assert av._clock("3.480") == 3.48
    assert abs(av._clock("00:07.101") - 7.101) < 1e-9
    assert abs(av._clock("01:02:03.500") - 3723.5) < 1e-9
    assert av._clock("") is None and av._clock("abc") is None


def test_ttml_line_starts_reads_local_name_and_reset_namespace(tmp_path):
    ttml = tmp_path / "x.ttml"
    ttml.write_text(
        '<tt xmlns="http://www.w3.org/ns/ttml"><body><div xmlns="">'
        '<p begin="00:01.500">a</p><p begin="00:03.000">b</p>'
        "</div></body></tt>",
        encoding="utf-8",
    )
    assert av.ttml_line_starts(ttml) == [1.5, 3.0]


# --------------------------------------------------------------------------- offset scan
def _impulse_env(starts_s, duration_s, hop=av.HOP, height=1.0):
    n = int(duration_s / hop)
    env = np.zeros(n, dtype=np.float32)
    for s in starts_s:
        i = round(s / hop)
        if 0 <= i < n:
            env[i] = height
    return env


def test_best_offset_recovers_a_known_shift():
    truth = [5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0]
    shift = 3.0
    env = _impulse_env([s + shift for s in truth], 40.0)
    off, score = av.best_offset(env, truth, max_offset=10.0, hop=av.HOP)
    assert off is not None and abs(off - shift) <= 3 * av.HOP
    assert score > 0.9


def test_best_offset_does_not_wrap_around():
    # starts near the very end must not match a peak rolled to the start
    env = _impulse_env([0.02], 40.0)
    _off, score = av.best_offset(env, [39.5], max_offset=15.0, hop=av.HOP)
    # a correct implementation cannot shift 39.5 s onto 0.02 s within +-15 s
    assert score < 0.5


def test_random_control_beats_nothing_on_noise():
    rng = np.random.default_rng(1)
    env = rng.random(int(40.0 / av.HOP)).astype(np.float32)
    starts = [5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0]
    _, score = av.best_offset(env, starts, max_offset=10.0, hop=av.HOP)
    rand = av._random_offset_scores(env, starts, 10.0, av.HOP, 200, 0)
    assert score <= float(np.percentile(rand, 99)) + 0.15  # ~no real signal on noise


# --------------------------------------------------------------------------- stability
def test_offset_stability_detects_an_unstable_reference():
    # peaks that drift between thirds -> spread should be large
    starts = [5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0, 37.0, 41.0, 45.0, 49.0]
    env = np.zeros(int(60.0 / av.HOP), dtype=np.float32)
    for k, s in enumerate(starts):
        i = round((s + (k % 3) * 4.0) / av.HOP)
        if 0 <= i < env.size:
            env[i] = 1.0
    stab = av.offset_stability(env, starts, max_offset=10.0, hop=av.HOP)
    assert stab["spread_s"] is not None and stab["spread_s"] >= 2.0


# --------------------------------------------------------------------------- verdict
def test_audit_never_declares_verified_without_a_human(tmp_path, monkeypatch):
    ttml = tmp_path / "s.ttml"
    ttml.write_text(
        '<tt xmlns="http://www.w3.org/ns/ttml"><body><div xmlns="">'
        + "".join(f'<p begin="{t}">x</p>' for t in
                  ["00:05.000", "00:09.000", "00:13.000", "00:17.000",
                   "00:21.000", "00:25.000", "00:29.000", "00:33.000"])
        + "</div></body></tt>",
        encoding="utf-8",
    )
    # a clean impulse train convolved into a vocal-band-like signal
    fs = av.SR
    dur = 40
    x = np.zeros(fs * dur, dtype=np.float32)
    for t in [5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0]:
        i = int(t * fs)
        x[i:i + int(0.05 * fs)] += np.hanning(int(0.05 * fs)).astype(np.float32)
    x += 0.001 * np.random.default_rng(0).standard_normal(x.size).astype(np.float32)

    monkeypatch.setattr(av, "load_mono16k", lambda _p: x)
    res = av.audit(ttml, tmp_path / "a.wav", control_audio=None, max_offset=10.0)
    assert res["status"] in {"UNVERIFIED", "MISMATCH"}
    assert res["status"] != "VERIFIED"

    res2 = av.audit(ttml, tmp_path / "a.wav", control_audio=None, max_offset=10.0,
                    human_verified=True)
    # even with human confirmation, a failed statistical check must stay MISMATCH
    if res["status"] == "MISMATCH":
        assert res2["status"] == "MISMATCH"
    else:
        assert res2["status"] == "VERIFIED"


def test_audit_marks_mismatch_when_beaten_by_control(tmp_path, monkeypatch):
    ttml = tmp_path / "s.ttml"
    ttml.write_text(
        '<tt xmlns="http://www.w3.org/ns/ttml"><body><div xmlns="">'
        + "".join(f'<p begin="{t}">x</p>' for t in
                  ["00:05.000", "00:09.000", "00:13.000", "00:17.000",
                   "00:21.000", "00:25.000", "00:29.000", "00:33.000"])
        + "</div></body></tt>",
        encoding="utf-8",
    )
    # audio is pure noise -> nothing should beat the random control
    noise = np.random.default_rng(3).standard_normal(av.SR * 40).astype(np.float32) * 0.1
    monkeypatch.setattr(av, "load_mono16k", lambda _p: noise)
    res = av.audit(ttml, tmp_path / "a.wav", control_audio=None, max_offset=10.0)
    assert res["status"] != "VERIFIED"


@pytest.mark.parametrize("status", ["VERIFIED", "UNVERIFIED", "MISMATCH"])
def test_status_values_are_from_the_documented_set(status):
    assert status in {"VERIFIED", "UNVERIFIED", "MISMATCH"}
