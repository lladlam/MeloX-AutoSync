#!/usr/bin/env python
"""Golden G2P + character↔phoneme reference for the Android JA chain.

This is the single source of truth the Android port must match. It is produced
by calling the engine implementations (no fork of `src/autosync/`):

* G2P + per-character attribution: ``autosync.lyrics.ja_pronunciation``
  (``JapanesePronunciationMapper`` wired exactly as ``PhonemeCTCAligner``
  wires it: ``table={}``, ``g2p=pyopenjtalk_kana``,
  ``convert=pyopenjtalk_phonemes``).
* phoneme → vocab ids: ``autosync.models.phoneme_ctc_aligner``
  (``PhonemeCTCAligner._phoneme_ids``; phonemes outside the 48-token vocab are
  skipped, never mapped to UNK).
* phoneme spans → character units: ``map_spans_to_char_units`` with
  ``locate_inner_offset``, so the per-character phoneme sequence is located
  *inside* the ``pau``-wrapped full sequence (no leading-``pau`` leak).

Reference chain (task book §2.3)::

    text --NFKC/overrides--> run_frontend (surface, katakana pron)
         --katakana→hiragana--> kana
         --pyopenjtalk.g2p(kana)--> phonemes --wrap pau--> ["pau", *ph, "pau"]

Two engine paths exist and both are covered (verified field-by-field):
``token_based`` (MeCab surface/reading tokens) and ``fallback`` (whole-string
kana + DP char alignment). The fallback is taken e.g. when NFKC changes the
string length or the frontend normalizes an ASCII space to U+3000, so the
per-token surfaces no longer rebuild the input.

Usage::

    .venv/bin/python android_export/ja/g2p_reference.py map "こんにちは せかい"
    .venv/bin/python android_export/ja/g2p_reference.py selftest
    .venv/bin/python android_export/ja/g2p_reference.py corpus FILE1.lrc ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SNAPSHOT = (
    REPO_ROOT
    / "models"
    / "hf"
    / "models--prj-beatrice--japanese-hubert-base-phoneme-ctc-v4"
    / "snapshots"
    / "f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99"
)

from autosync.lyrics.ja_pronunciation import (  # noqa: E402
    JapanesePronunciationMapper,
    _is_kana,
    katakana_to_hiragana,
    locate_inner_offset,
)
from autosync.models.phoneme_ctc_aligner import (  # noqa: E402
    PhonemeCTCAligner,
    map_spans_to_char_units,
    pyopenjtalk_kana,
    pyopenjtalk_phonemes,
)

BLANK_ID = 0


# --------------------------------------------------------------------------- #
# wiring (identical to PhonemeCTCAligner.__init__)
# --------------------------------------------------------------------------- #
def build_mapper() -> JapanesePronunciationMapper:
    """The engine's G2P wiring; do not change (that would fork the spec)."""
    return JapanesePronunciationMapper(
        table={}, g2p=pyopenjtalk_kana, convert=pyopenjtalk_phonemes
    )


def load_vocab(path: Path | None = None) -> dict[str, int]:
    with open(path or (SNAPSHOT / "vocab.json"), encoding="utf-8") as f:
        return json.load(f)


class _VocabShim:
    """Lets us call the engine's ``_phoneme_ids`` / ``_blank_id`` verbatim."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = dict(vocab)

        class _Tok:
            pad_token_id = BLANK_ID

        class _Proc:
            tokenizer = _Tok()

        self._processor = _Proc()


def phoneme_ids(phonemes: list[str], vocab: dict[str, int]) -> tuple[list[int], list[int]]:
    """Engine logic: keep only phonemes present in the 48-token vocab."""
    return PhonemeCTCAligner._phoneme_ids(_VocabShim(vocab), phonemes)


def blank_id(vocab: dict[str, int]) -> int:
    return PhonemeCTCAligner._blank_id(_VocabShim(vocab))


# --------------------------------------------------------------------------- #
# independent re-derivation of the documented chain
# --------------------------------------------------------------------------- #
def spec_tokens(norm_text: str) -> list[tuple[str, str]]:
    """``(surface, hiragana reading)`` from pyopenjtalk's MeCab frontend."""
    import pyopenjtalk

    out: list[tuple[str, str]] = []
    for m in pyopenjtalk.run_frontend(norm_text):
        surface = m.get("string") or m.get("orig") or ""
        pron = m.get("pron") or ""
        if not surface:
            continue
        clean = "".join(c for c in pron if _is_kana(c) or c == "\u30fc")
        out.append((surface, katakana_to_hiragana(clean)))
    return out


def _prefix_counts(mapper: JapanesePronunciationMapper, kana: str, inner_len: int) -> list[int]:
    """Documented monotonic prefix-count repair (mirrors `_attribute_counts`)."""
    n = len(kana)
    counts = [len(mapper._strip_pau(mapper._convert(kana[:i], mapper._table))) for i in range(n + 1)]
    counts[n] = inner_len
    for i in range(n - 1, -1, -1):
        counts[i] = min(counts[i], counts[i + 1])
    for i in range(1, n + 1):
        counts[i] = max(counts[i], counts[i - 1])
    counts[n] = inner_len
    return counts


def derive_chain(text: str, mapper: JapanesePronunciationMapper | None = None) -> dict:
    """Re-derive ``text`` from pyopenjtalk following the documented chain."""
    mapper = mapper or build_mapper()
    norm = unicodedata.normalize("NFKC", mapper.apply_overrides(text))
    chars = list(text)

    engine_tokens = mapper._tokenize(norm)  # engine tokenizer (for comparison)
    tokens = spec_tokens(norm)              # independent frontend call
    token_path = (
        tokens is not None
        and "".join(s for s, _ in tokens) == norm
        and len(norm) == len(chars)
    )

    kana_all = "".join(k for _, k in tokens)
    phonemes = pyopenjtalk_phonemes(kana_all, {})
    inner = mapper._strip_pau(phonemes)

    char_spans: list[tuple[int, int] | None]
    if token_path:
        char_spans = [None] * len(chars)
        pos = kpos = 0
        for surface, kana in tokens:
            local = mapper.align_kana_to_chars(list(surface), kana)
            if local is None:
                token_path = False
                break
            for ci, span in enumerate(local):
                if span is not None:
                    char_spans[pos + ci] = (kpos + span[0], kpos + span[1])
            pos += len(surface)
            kpos += len(kana)
    if not token_path:
        char_spans = mapper.align_kana_to_chars(chars, kana_all)

    # Engine behaviour: when the DP cannot attribute the kana (e.g. mixed
    # latin/kanji/kana lines where the reading is longer than the capacity of
    # the characters), `map()` returns *no* per-character attribution at all
    # (kana="", phonemes=[], uncertain=True) while keeping the full phoneme
    # sequence. Mirror that exactly.
    no_attribution = char_spans is None
    if no_attribution:
        char_spans = [None] * len(chars)

    counts = _prefix_counts(mapper, kana_all, len(inner))
    counts_engine = mapper._attribute_counts(kana_all, len(inner))

    per_char = []
    for ch, span in zip(chars, char_spans):
        if span is None:
            per_char.append({"char": ch, "kana": "", "phonemes": [], "uncertain": True})
            continue
        a, b = span
        ph = inner[counts[a] : counts[b]]
        per_char.append(
            {
                "char": ch,
                "kana": kana_all[a:b],
                "phonemes": ph,
                "uncertain": not ph,
            }
        )
    return {
        "norm": norm,
        "tokens": tokens,
        "engine_tokens": engine_tokens,
        "token_path": token_path,
        "kana": kana_all,
        "phonemes": phonemes,
        "inner": inner,
        "per_char": per_char,
        "no_attribution": no_attribution,
        "prefix_counts": counts,
        "prefix_counts_match_engine": counts == counts_engine,
        "char_spans": char_spans,
    }


def reference_map(text: str, mapper: JapanesePronunciationMapper | None = None) -> dict:
    """Full golden record for ``text`` (engine output + span bookkeeping)."""
    mapper = mapper or build_mapper()
    vocab = load_vocab()

    readings, phonemes = mapper.map(text)  # engine authority
    derived = derive_chain(text, mapper)
    no_attribution = derived["no_attribution"]

    # Which engine path was taken? `_map_token_based` returning non-None means
    # the context-aware token path; None means the whole-string fallback.
    norm = unicodedata.normalize("NFKC", mapper.apply_overrides(text))
    try:
        engine_token_path = mapper._map_token_based(norm, list(text)) is not None
    except Exception:  # noqa: BLE001
        engine_token_path = None

    inner = [p for r in readings for p in r.phonemes]
    offset = locate_inner_offset(phonemes, inner) if inner else 0

    char_spans = []
    cursor = offset
    for r in readings:
        n = len(r.phonemes)
        char_spans.append(
            {
                "char": r.char,
                "kana": r.kana,
                "phoneme_index_start": cursor,
                "phoneme_index_end": cursor + n,
                "uncertain": bool(r.uncertain),
            }
        )
        cursor += n

    ids, kept = phoneme_ids(phonemes, vocab)
    dropped = [{"index": i, "phoneme": p} for i, p in enumerate(phonemes) if p not in vocab]

    engine_fields = {
        "kana": "".join(r.kana for r in readings),
        "phonemes": phonemes,
        "per_char": [
            {"char": r.char, "kana": r.kana, "phonemes": list(r.phonemes), "uncertain": bool(r.uncertain)}
            for r in readings
        ],
    }
    derived_fields = {
        # like-for-like: engine's `kana` is the concatenation of per-char kana
        # (empty when the DP could not attribute the reading at all)
        "kana": "".join(pc["kana"] for pc in derived["per_char"]),
        "phonemes": derived["phonemes"],
        "per_char": derived["per_char"],
    }
    field_ok = {k: engine_fields[k] == derived_fields[k] for k in engine_fields}
    field_ok["tokens"] = derived["engine_tokens"] == derived["tokens"]
    field_ok["prefix_counts"] = derived["prefix_counts_match_engine"]

    return {
        "text": text,
        "tokens": [{"surface": s, "kana": k} for s, k in (derived["tokens"] or [])],
        "kana": engine_fields["kana"],
        "kana_reading": derived["kana"],
        "phonemes": list(phonemes),
        "phoneme_ids": ids,
        "kept_phoneme_indices": kept,
        "dropped_phonemes": dropped,
        "char_phoneme_spans": char_spans,
        "blank_id": blank_id(vocab),
        "inner_offset": offset,
        "no_attribution": no_attribution,
        "engine_token_based_path": engine_token_path,
        "derived_token_based_path": derived["token_path"],
        "path_matches": engine_token_path == derived["token_path"],
        "field_ok": field_ok,
        "all_fields_ok": all(field_ok.values()),
        "reconstruct_ok": JapanesePronunciationMapper.reconstruct(readings) == text,
        "readings": readings,
        "engine_fields": engine_fields,
        "derived_fields": derived_fields,
        "vocab": vocab,
    }


# --------------------------------------------------------------------------- #
# corpus statistics (real lyric lines)
# --------------------------------------------------------------------------- #
_TS = re.compile(r"^\s*\[[^\]]*\]\s*(.*)$")
_CREDIT = re.compile(
    r"^\s*(作词|作詞|作曲|编曲|編曲|制作|製作|吉他|贝斯|ベース|ドラム|键盘|鍵盤|和声|混音|母带|录音|錄音|"
    r"封面|设计|插画|映像|监督|企划|发行|出品|统筹|协力|翻译|字幕|演唱|演奏|歌|词|曲)"
)


def _is_kanji(c: str) -> bool:
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF


def lrc_lines(path: Path) -> list[tuple[int, str]]:
    """``(line_no, text)`` for lyric-looking lines (credits dropped)."""
    out: list[tuple[int, str]] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        m = _TS.match(raw)
        text = (m.group(1) if m else raw).strip()
        if not text:
            continue
        if _CREDIT.match(text) or " : " in text or "\uff1a" in text:
            continue
        if any(_is_kanji(c) for c in text) or any(_is_kana(c) for c in text):
            out.append((i, text))
    return out


def corpus_stats(lines: list[str], mapper: JapanesePronunciationMapper | None = None) -> dict:
    """OOV rate + `characters with no phoneme` rate over real Japanese lines."""
    import pyopenjtalk

    mapper = mapper or build_mapper()
    vocab = load_vocab()
    n_phon = n_oov = n_chars = n_no_ph = n_kanji = n_uncertain = 0
    n_naive_ok = n_naive_ctx = 0  # option-B straw man: isolated-kana conversion
    no_ph_by_class: dict[str, int] = {}
    n_no_attr_lines = n_no_attr_chars = 0
    per_line = []
    field_fail = 0
    for text in lines:
        rec = reference_map(text, mapper)
        if not rec["all_fields_ok"]:
            field_fail += 1
        if rec["no_attribution"]:
            n_no_attr_lines += 1
            n_no_attr_chars += len(text)
        ph = rec["phonemes"]
        oov = sum(1 for p in ph if p not in vocab)
        no_ph = 0
        for r in rec["readings"]:
            if r.phonemes:
                continue
            no_ph += 1
            c = r.char
            if c.isspace():
                cls = "space"
            elif _is_kana(c) or c == "\u30fc":
                cls = "kana"
            elif _is_kanji(c):
                cls = "kanji"
            elif c.isascii() and c.isalnum():
                cls = "latin_digit"
            else:
                cls = "punct_symbol"
            no_ph_by_class[cls] = no_ph_by_class.get(cls, 0) + 1
        kanji = sum(1 for c in text if _is_kanji(c))
        n_phon += len(ph)
        n_oov += oov
        n_chars += len(text)
        n_no_ph += no_ph
        n_kanji += kanji
        n_uncertain += sum(1 for r in rec["readings"] if r.uncertain)
        for r in rec["readings"]:
            if _is_kana(r.char) and r.char not in "\u30fc":
                try:
                    ph_iso = [p for p in pyopenjtalk.g2p(r.char).split() if p != "pau"]
                except Exception:  # noqa: BLE001
                    ph_iso = []
                if ph_iso == r.phonemes:
                    n_naive_ok += 1
                else:
                    n_naive_ctx += 1
            elif r.phonemes:
                n_naive_ctx += 1  # kanji: needs the dictionary
        per_line.append(
            {
                "n_chars": len(text),
                "n_phonemes": len(ph),
                "n_oov": oov,
                "n_chars_no_phoneme": no_ph,
                "n_kanji_chars": kanji,
            }
        )
    denom = n_naive_ok + n_naive_ctx
    return {
        "n_lines": len(lines),
        "n_lines_field_mismatch": field_fail,
        "n_lines_no_attribution": n_no_attr_lines,
        "n_phonemes": n_phon,
        "n_oov_phonemes": n_oov,
        "oov_rate": (n_oov / n_phon) if n_phon else 0.0,
        "n_chars": n_chars,
        "n_chars_no_phoneme": n_no_ph,
        "chars_no_phoneme_rate": (n_no_ph / n_chars) if n_chars else 0.0,
        "n_chars_no_phoneme_excl_no_attribution": n_no_ph - n_no_attr_chars,
        "chars_no_phoneme_by_class": no_ph_by_class,
        "n_kanji_chars": n_kanji,
        "kanji_rate": (n_kanji / n_chars) if n_chars else 0.0,
        "n_uncertain_chars": n_uncertain,
        "option_b_isolated_kana_agreement": (n_naive_ok / denom) if denom else 0.0,
        "option_b_isolated_kana_ok": n_naive_ok,
        "option_b_isolated_kana_total": denom,
        "per_line": per_line,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _jsonable(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k not in ("readings", "engine_fields", "derived_fields", "vocab")}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_map = sub.add_parser("map", help="export the golden record for one text")
    p_map.add_argument("text")
    p_map.add_argument("--json", type=Path, default=None)

    p_st = sub.add_parser("selftest", help="independent chain vs engine, field by field")
    p_st.add_argument("texts", nargs="*", default=None)

    p_c = sub.add_parser("corpus", help="OOV statistics over real lyric lines")
    p_c.add_argument("lrcs", nargs="+", type=Path)
    p_c.add_argument("--json", type=Path, default=None)

    args = ap.parse_args(argv)
    if args.cmd == "map":
        rec = _jsonable(reference_map(args.text))
        text = json.dumps(rec, ensure_ascii=False, indent=2)
        if args.json:
            args.json.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    if args.cmd == "selftest":
        texts = args.texts or [
            "みなさん、こんにちは",
            "東京の街を歩いてる",
            "シュビデュビ　娑婆Lover",
            "きょうはいい天気ですね",
            "こんにちは せかい",
            "興味をそそられる村を30分ほど散策するのは価値があります。",
            "Hello 世界 123",
        ]
        bad = 0
        for t in texts:
            r = reference_map(t)
            ok = r["all_fields_ok"] and r["path_matches"] and r["reconstruct_ok"]
            bad += 0 if ok else 1
            print(
                f"[{'OK ' if ok else 'FAIL'}] {t!r}: chars={len(t)} "
                f"phonemes={len(r['phonemes'])} dropped={len(r['dropped_phonemes'])} "
                f"path={'token' if r['engine_token_based_path'] else 'fallback'} "
                f"fields={r['field_ok']} reconstruct={r['reconstruct_ok']}"
            )
        print(f"selftest: {len(texts) - bad}/{len(texts)} texts match the engine field-by-field")
        return 1 if bad else 0
    if args.cmd == "corpus":
        lines: list[tuple[str, int, str]] = []
        for p in args.lrcs:
            for ln, text in lrc_lines(p):
                lines.append((p.name, ln, text))
        stats = corpus_stats([t for _, _, t in lines])
        stats["sources"] = [
            {
                "file": str(p),
                "n_lines": sum(1 for name, _, _ in lines if name == p.name),
            }
            for p in args.lrcs
        ]
        text = json.dumps(stats, ensure_ascii=False, indent=2)
        if args.json:
            args.json.write_text(text + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in stats.items() if k != "per_line"}, ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
