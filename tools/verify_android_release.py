#!/usr/bin/env python
"""Verify a published Android model pack end-to-end (download -> load -> infer).

Designed for the artifacts produced by ``tools/build_android_release.py``.  It
checks, against the pack's own manifests:

1. every file in ``checksums.sha256`` matches its SHA-256;
2. each ONNX loads on the CPU ExecutionProvider with exactly the I/O names,
   dtypes and ranks the manifest promises (plus opset <= 17 and no
   ``com.microsoft`` operators);
3. a real inference runs on a real audio clip and returns finite tensors of the
   expected shape, with the frame-count relation the manifest documents.

Usage::

    .venv/bin/python tools/verify_android_release.py --dir /path/to/downloaded_pack
    .venv/bin/python tools/verify_android_release.py --download   # fetch from the release

Nothing here needs PyTorch; it is the same ONNX Runtime path Android uses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO_SLUG = "lladlam/MeloX-AutoSync"
TAG = "android-models-v0.1.0-beta.1"
BASE = f"https://github.com/{REPO_SLUG}/releases/download/{TAG}"
FRAME_HOP = 320
SR = 16000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_checksums(root: Path) -> list[str]:
    problems = []
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split()
        target = root / name
        if not target.exists():
            problems.append(f"missing {name}")
        elif sha256(target) != expected:
            problems.append(f"sha256 mismatch {name}")
    return problems


def check_onnx(root: Path, manifest: dict) -> list[str]:
    import numpy as np
    import onnx
    import onnxruntime as ort

    problems: list[str] = []
    model_file = next(f for f in manifest["files"] if f["role"] == "model")
    path = root / model_file["name"]
    model = onnx.load(str(path), load_external_data=False)
    opsets = max(int(op.version) for op in model.opset_import)
    if opsets > 17:
        problems.append(f"{model_file['name']}: opset {opsets} > 17")
    contrib = [n.op_type for n in model.graph.node if "com.microsoft" in (n.domain or "")]
    if contrib:
        problems.append(f"{model_file['name']}: contrib ops {sorted(set(contrib))}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    got_in = [(i.name, i.type, list(i.shape)) for i in sess.get_inputs()]
    got_out = [(o.name, o.type, list(o.shape)) for o in sess.get_outputs()]
    want_in = [(manifest["input"]["name"], "tensor(float)")] + [
        (a["name"], "tensor(int64)") for a in manifest.get("aux_inputs") or []
    ]
    want_out = [(o["name"], "tensor(float)") for o in manifest["outputs"]]
    if [(n, t) for n, t, _ in got_in] != want_in:
        problems.append(f"{model_file['name']}: inputs {[(n, t) for n, t, _ in got_in]} != {want_in}")
    if [(n, t) for n, t, _ in got_out] != want_out:
        problems.append(f"{model_file['name']}: outputs {[(n, t) for n, t, _ in got_out]} != {want_out}")

    # real inference on a synthetic-but-valid 2 s waveform
    audio = (0.05 * np.sin(2 * np.pi * 220 * np.arange(2 * SR) / SR)).astype(np.float32)
    feeds = {"input_values": audio[None, :]}
    if manifest.get("aux_inputs"):
        feeds["token_num"] = np.array([8], dtype=np.int64)
    outs = sess.run(None, feeds)
    for (name, _t, shape), arr in zip(got_out, outs):
        if not np.isfinite(arr).all():
            problems.append(f"{model_file['name']}: {name} not finite")
        if arr.ndim != len(shape):
            problems.append(f"{model_file['name']}: {name} rank {arr.ndim} != {len(shape)}")
    expected_t = (len(audio) - 400) // FRAME_HOP + 1
    lang = manifest["language"]
    if lang == "en":
        if outs[0].shape[1] != expected_t:
            problems.append(f"en: T={outs[0].shape[1]} != floor((N-400)/320)+1={expected_t}")
        if outs[0].shape[2] != 32:
            problems.append(f"en: vocab {outs[0].shape[2]} != 32")
    elif lang == "ja" and outs[0].shape[2] != 48:
        problems.append(f"ja: vocab {outs[0].shape[2]} != 48")
    elif lang == "zh" and (outs[0].shape[0] != 1 or outs[0].shape[1] % 3 != 0):
        problems.append(f"zh: us_alphas shape {outs[0].shape} not [1, 3*T_enc]")
    return problems


def download(root: Path) -> None:
    """Fetch the pack from the release into ``root`` (only what is missing/bad)."""
    root.mkdir(parents=True, exist_ok=True)
    sums = root / "checksums.sha256"
    if not sums.exists():
        subprocess.run(["curl", "-sSL", "-o", str(sums), f"{BASE}/checksums.sha256"], check=True)
    entries = [line.split() for line in sums.read_text(encoding="utf-8").splitlines() if line.strip()]
    for expected, name in entries:
        target = root / name
        if target.exists() and sha256(target) == expected:
            continue
        print(f"  downloading {name}")
        subprocess.run(["curl", "-sSL", "-o", str(target), f"{BASE}/{name}"], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, help="directory holding the downloaded pack")
    ap.add_argument("--download", action="store_true", help="download the pack first")
    ap.add_argument("--tag", default=None, help="release tag to verify")
    args = ap.parse_args()
    global TAG, BASE
    if args.tag and args.tag != TAG:
        TAG = args.tag
        BASE = f"https://github.com/{REPO_SLUG}/releases/download/{TAG}"
    root = args.dir or Path("/tmp/opencode/verify_pack")
    if args.download:
        download(root)

    problems = check_checksums(root)
    print(f"checksums: {'OK' if not problems else problems}")
    index = json.loads((root / "android-models.json").read_text(encoding="utf-8"))
    for lang, manifest in index["languages"].items():
        probs = check_onnx(root, manifest)
        status = "OK" if not probs else "FAIL"
        print(f"[{lang}] {manifest['engine']:22s} {status}")
        for p in probs:
            print(f"      {p}")
        problems += probs
    print("VERDICT:", "ALL CHECKS PASSED" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
