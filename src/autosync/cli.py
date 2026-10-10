"""Command line interface for MeloX-AutoSync."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .alignment.heuristic import distribute
from .alignment.segment import SegmentOptions, align_lines
from .audio.io import duration_seconds, load_audio
from .config import CROP_PADDING_SECONDS, DEFAULT_SAMPLE_RATE, MODELS_DIR
from .export.json_export import write_json
from .lyrics.lrc import parse_lrc
from .lyrics.normalize import display_units
from .models.registry import (
    DEFAULT_BACKENDS,
    SUPPORTED_LANGUAGES,
    get_aligner,
    list_backends,
)
from .quality.gating import DEFAULT_POLICIES, gate_result, summarize
from .types import AlignedLine, AlignmentResult, ModelInfo


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _heuristic_lines(audio, sample_rate: int, lines, language: str):
    total = duration_seconds(audio, sample_rate)
    out: list[AlignedLine] = []
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        end_ref = nxt.start if nxt and nxt.start > line.start else min(total, line.start + 12.0)
        units = display_units(language, line.text)
        dist = distribute(line.start, end_ref, units) if units else []
        out.append(
            AlignedLine(
                index=line.index,
                text=line.text,
                start=dist[0].start if dist else line.start,
                end=dist[-1].end if dist else line.start,
                units=dist,
                source="heuristic" if dist else "failed",
                error=None if dist else "no display units",
                lrc_start=line.start,
                lrc_end=line.end,
            )
        )
    return out


def cmd_align(args: argparse.Namespace) -> int:
    lines, meta = parse_lrc(args.lyrics)
    if not lines:
        print("error: no timed lyric lines found", file=sys.stderr)
        return 2

    audio, sample_rate = load_audio(args.audio, target_sr=DEFAULT_SAMPLE_RATE)
    total = duration_seconds(audio, sample_rate)
    print(
        f"loaded audio: {total:.2f}s @ {sample_rate}Hz, {len(lines)} lyric lines",
        file=sys.stderr,
    )

    result: AlignmentResult
    if args.engine == "heuristic":
        aligned = _heuristic_lines(audio, sample_rate, lines, args.language)
        result = AlignmentResult(
            language=args.language,
            engine="heuristic",
            lines=aligned,
            model=None,
            warnings=["heuristic engine: timestamps are evenly distributed, NOT acoustic"],
            diagnostics={"source_metadata": meta},
        )
    else:
        aligner = get_aligner(args.language, backend=args.backend, cache_dir=Path(args.cache_dir))
        t0 = time.time()
        aligner.load()
        load_seconds = time.time() - t0
        print(
            f"backend: {aligner.meta.name}  status={aligner.meta.status}  "
            f"license={aligner.meta.license}",
            file=sys.stderr,
        )
        if args.device != "cpu":
            print(f"note: only CPU inference is validated; requested {args.device}", file=sys.stderr)
        options = SegmentOptions(
            padding=args.padding, fallback=args.fallback, refine=args.refine
        )
        aligned, warnings = align_lines(
            audio,
            sample_rate,
            lines,
            aligner,
            options,
            progress=not args.quiet,
            language=args.language,
        )
        engine = "model" if args.refine == "none" else f"model+{args.refine}"
        result = AlignmentResult(
            language=args.language,
            engine=engine,
            lines=aligned,
            model=ModelInfo(
                name=aligner.meta.name,
                version=aligner.meta.version or "unknown",
                language=aligner.meta.language,
                license=aligner.meta.license,
                device="cpu",
                load_seconds=round(load_seconds, 3),
                infer_seconds=round(getattr(aligner, "infer_seconds", 0.0) or 0.0, 3),
                frame_seconds=aligner.frame_seconds,
                status=aligner.meta.status,
            ),
            warnings=warnings,
            diagnostics={
                "source_metadata": meta,
                "padding_seconds": args.padding,
                "refine": args.refine,
                "backend": args.backend or DEFAULT_BACKENDS.get(args.language),
            },
        )

    # V0.8: resolve the quality gate (only model engines use a policy) BEFORE export.
    policy = DEFAULT_POLICIES.get(args.language) if args.engine == "model" else None
    gate_result(result, policy, mode=args.mode)

    out = write_json(result, args.output)
    n_model = sum(1 for ln in result.lines if ln.source == "model")
    n_failed = sum(1 for ln in result.lines if ln.source == "failed")
    qs = summarize(result.lines)
    print(
        f"wrote {out}\n  lines: {len(result.lines)}  model-aligned: {n_model}  failed: {n_failed}\n"
        f"  quality: reliable={qs['reliable']} uncertain={qs['uncertain']} "
        f"failed={qs['failed']} not_assessed={qs['not_assessed']}",
        file=sys.stderr,
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    print("MeloX-AutoSync backends (per language, loaded on demand):")
    current = None
    for spec in list_backends():
        if spec.language != current:
            current = spec.language
            print(f"\n[{current}]  default={DEFAULT_BACKENDS[current]}")
        default = " (default)" if spec.key == DEFAULT_BACKENDS[current] else ""
        print(
            f"  {spec.key}{default}\n"
            f"      repo={spec.repo}\n"
            f"      status={spec.status}  license={spec.license}  "
            f"distributable={'yes' if spec.distributable else 'no'}\n"
            f"      {spec.description}"
        )
    print(f"\nmodel cache: {MODELS_DIR}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autosync",
        description="Offline lyrics-to-audio forced alignment (zh/ja/en).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("align", help="align an LRC file against an audio file")
    a.add_argument("--audio", required=True, help="audio file (wav/mp3/flac/...)")
    a.add_argument("--lyrics", required=True, help="LRC lyrics file")
    a.add_argument("--language", required=True, choices=SUPPORTED_LANGUAGES)
    a.add_argument("--output", required=True, help="output JSON path")
    a.add_argument("--engine", choices=["model", "heuristic"], default="model")
    a.add_argument(
        "--backend",
        default=None,
        help="model backend key (default: language default). See `autosync info`.",
    )
    a.add_argument(
        "--mode",
        choices=["production", "experimental"],
        default="production",
        help="production falls back to line LRC unless RELIABLE; experimental also "
        "emits diagnostic word timings for UNCERTAIN (never FAILED)",
    )
    a.add_argument("--device", default="cpu", help="cpu (only cpu is validated)")
    a.add_argument(
        "--padding",
        type=float,
        default=CROP_PADDING_SECONDS,
        help="seconds of audio margin around each line",
    )
    a.add_argument(
        "--fallback",
        choices=["none", "heuristic"],
        default="none",
        help="what to do when a line fails to align (default: report failure)",
    )
    a.add_argument(
        "--refine",
        choices=["none", "energy"],
        default="none",
        help="V0.2 post-alignment boundary refinement (default: none = V0.1 behaviour)",
    )
    a.add_argument("--cache-dir", default=str(MODELS_DIR / "hf"), help="model cache directory")
    a.add_argument("-q", "--quiet", action="store_true", help="disable progress bar")
    a.set_defaults(func=cmd_align)

    i = sub.add_parser("info", help="list supported models")
    i.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
