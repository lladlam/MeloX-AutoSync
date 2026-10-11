#!/usr/bin/env python
"""Build a *selective* dynamic-INT8 ONNX candidate by excluding module groups.

Dynamic quantisation of every MatMul/Gemm is what the beta.2 baselines use.  This
tool rebuilds a candidate while keeping chosen module groups in FP32, so the
effect of each group on real-music timing can be measured in isolation.

The workspace is language-agnostic: a group is just a substring that must appear
in the node name (e.g. ``lm_head``, ``/attention/``, ``/feed_forward/``).

Usage::

    .venv/bin/python tools/selective_quantize.py \
        --fp32 android_export/en/wav2vec2-base-960h-fp32.onnx \
        --out /tmp/cand_en_head.onnx --exclude lm_head
    # several groups
    ... --exclude lm_head --exclude /attention/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def build(fp32: Path, out: Path, excludes: list[str], exclude_regex: list[str],
          per_channel: bool, ops: list[str], weight_type: str = "QInt8",
          reduce_range: bool = False, weight_suffixes: list[str] | None = None) -> dict:
    import re

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = onnx.load(str(fp32), load_external_data=False)
    all_nodes = [n for n in model.graph.node if n.op_type in ops]
    compiled = [re.compile(r) for r in exclude_regex]
    excluded = {n.name for n in all_nodes
                if any(pat in n.name for pat in excludes)
                or any(c.search(n.name) for c in compiled)}
    # exclude nodes consuming specific weights (e.g. a baked DFT/Mel frontend)
    if weight_suffixes:
        init_names = {i.name for i in model.graph.initializer
                      if any(i.name.endswith(sfx) for sfx in weight_suffixes)}
        for n in all_nodes:
            if any(i in init_names for i in n.input):
                excluded.add(n.name)
    t0 = time.time()
    quantize_dynamic(
        str(fp32), str(out),
        weight_type=QuantType[weight_type],
        per_channel=per_channel,
        reduce_range=reduce_range,
        op_types_to_quantize=list(ops),
        nodes_to_exclude=sorted(excluded),
    )
    record = {
        "source": str(fp32), "out": str(out), "excludes": excludes,
        "exclude_regex": exclude_regex,
        "n_quantizable": len(all_nodes), "n_excluded": len(excluded),
        "bytes": out.stat().st_size,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "per_channel": per_channel, "op_types": ops,
        "weight_type": weight_type, "reduce_range": reduce_range,
        "weight_suffixes": weight_suffixes or [],
        "seconds": round(time.time() - t0, 1),
    }
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", action="append", default=[],
                    help="substring of node names to keep in FP32 (repeatable)")
    ap.add_argument("--exclude-regex", action="append", default=[],
                    help="regex of node names to keep in FP32 (repeatable)")
    ap.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    ap.add_argument("--ops", default="MatMul,Gemm")
    ap.add_argument("--weight-type", default="QInt8", choices=["QInt8", "QUInt8"])
    ap.add_argument("--reduce-range", action="store_true")
    ap.add_argument("--exclude-weight-suffix", action="append", default=[],
                    help="exclude nodes consuming weights whose name ends with this")
    ap.add_argument("--meta", type=Path, default=None, help="write a JSON record here")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rec = build(args.fp32, args.out, args.exclude, args.exclude_regex,
                args.per_channel, args.ops.split(","), args.weight_type, args.reduce_range,
                args.exclude_weight_suffix)
    if args.meta:
        args.meta.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
