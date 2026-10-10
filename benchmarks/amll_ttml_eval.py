"""Independent AMLL evaluation. No imports from the alignment implementation.

All detailed output belongs in ignored directories. Timing never enters matching.
AMLL clock values are media-absolute (including nested spans), per its specification.
Unsupported timing containers fail closed rather than inventing reference times.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

import numpy as np

TTM = "http://www.w3.org/ns/ttml#metadata"
TTP = "http://www.w3.org/ns/ttml#parameter"


def local(tag):
    return tag.rsplit("}", 1)[-1]


def norm(text):
    """NFKC/casefold, omit whitespace/punctuation; retain letters, marks, numbers."""
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if unicodedata.category(c)[0] in "LNM")


def timestamp(value, frame_rate=30.0, sub_frame_rate=1, tick_rate=1.0):
    if not value:
        raise ValueError("missing timestamp")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|h|m|s|f|t)?", value)
    if m:
        factors = {None: 1, "ms": .001, "h": 3600, "m": 60, "s": 1,
                   "f": 1 / frame_rate, "t": 1 / tick_rate}
        return float(m[1]) * factors[m[2]]
    m = re.fullmatch(r"(\d+):(\d{2}):(\d{2}):(\d+)(?:\.(\d+))?", value)
    if m:
        h, minute, sec, frame = map(int, m.groups()[:4])
        if minute >= 60 or sec >= 60 or frame >= frame_rate:
            raise ValueError(f"invalid frame clock: {value}")
        return h * 3600 + minute * 60 + sec + (frame + int(m[5] or 0) / sub_frame_rate) / frame_rate
    parts = value.split(":")
    if len(parts) not in (2, 3) or not all(re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
        raise ValueError(f"unsupported timestamp: {value}")
    if any(float(p) >= 60 for p in parts[1:]):
        raise ValueError(f"invalid clock: {value}")
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))


def parse_ttml(path):
    root = ET.parse(path).getroot()
    fps = float(root.get(f"{{{TTP}}}frameRate", "30"))
    mult = root.get(f"{{{TTP}}}frameRateMultiplier", "1 1").split()
    fps *= float(mult[0]) / float(mult[1])
    sub = int(root.get(f"{{{TTP}}}subFrameRate", "1"))
    tick = float(root.get(f"{{{TTP}}}tickRate", str(fps * sub if f"{{{TTP}}}frameRate" in root.attrib else 1)))

    def time(v):
        return timestamp(v, fps, sub, tick)

    metadata = defaultdict(list)
    counts = Counter()
    lines, warnings = [], []
    auxiliary_roles = {"x-translation", "x-roman", "x-romanization"}

    def singing_text(e):
        if e.get(f"{{{TTM}}}role") in auxiliary_roles or local(e.tag) == "empty-beat":
            return ""
        return (e.text or "") + "".join(singing_text(c) + (c.tail or "") for c in e)

    def timed_child(e):
        for c in e:
            if c.get(f"{{{TTM}}}role") in auxiliary_roles or local(c.tag) == "empty-beat":
                continue
            if "begin" in c.attrib or timed_child(c):
                return True
        return False
    for e in root.iter():
        if e.get("timeContainer") == "seq":
            raise ValueError("sequential timing containers are not supported")
        if local(e.tag) == "meta" and e.get("key"):
            metadata[e.get("key")].append(e.get("value"))
        if local(e.tag) == "agent":
            metadata["_agents"].append({"id": e.get("{http://www.w3.org/XML/1998/namespace}id"),
                                        "type": e.get("type")})
    body = next(e for e in root.iter() if local(e.tag) == "body")
    for pi, p in enumerate(e for e in body.iter() if local(e.tag) == "p"):
        line = {"index": pi, "start": time(p.get("begin")), "end": time(p.get("end")),
                "agent": p.get(f"{{{TTM}}}agent"), "lead": [], "background": []}
        if line["end"] < line["start"]:
            raise ValueError("negative line duration")

        def walk(e, role="lead", agent=None, line=line, pi=pi):
            erole = e.get(f"{{{TTM}}}role", "")
            if erole in auxiliary_roles:
                counts[erole] += 1
                return
            if erole == "x-bg":
                role = "background"
                counts["background_containers"] += 1
            agent = e.get(f"{{{TTM}}}agent", agent)
            if local(e.tag) == "empty-beat":
                counts["empty_beat_elements"] += 1
                return
            beat = next((v for k, v in e.attrib.items() if local(k) == "empty-beat"), None)
            if beat is not None:
                counts["empty_beat_annotations"] += 1
            # Containers are never duplicated as singing units. Plain formatting
            # children are permitted; nested timed/role children are walked.
            structured = timed_child(e)
            if local(e.tag) == "span" and "begin" in e.attrib and "end" in e.attrib and not structured:
                text = singing_text(e)
                if not norm(text):
                    counts["nonlexical_timed_spans"] += 1
                    return
                start, end = time(e.get("begin")), time(e.get("end"))
                if end < start:
                    raise ValueError("negative unit duration")
                unit = {"text": text, "start": start, "end": end, "agent": agent,
                        "empty_beat": beat, "line_index": pi, "unit_index": len(line[role])}
                line[role].append(unit)
                if start < line["start"] or end > line["end"]:
                    warnings.append({"line": pi, "role": role, "issue": "outside_p"})
                return
            for child in e:
                walk(child, role, agent)

        walk(p, agent=line["agent"])
        line["text"] = "".join(u["text"] for u in line["lead"])
        lines.append(line)
    # Distinct non-background singers may overlap. Preserve their identity and
    # report overlap instead of asserting one globally monotonic vocal stream.
    all_agents = defaultdict(list)
    for line in lines:
        for unit in line["lead"]:
            all_agents[unit["agent"] or "_unspecified"].append(unit)
    for agent, units in all_agents.items():
        units.sort(key=lambda u: (u["start"], u["end"]))
        active = []
        for unit in units:
            active = [u for u in active if u["end"] > unit["start"]]
            counts["same_agent_overlap_pairs"] += sum(u["end"] > unit["start"] for u in active)
            active.append(unit)
    agent_ids = list(all_agents)
    for i, agent in enumerate(agent_ids):
        for other in agent_ids[i + 1:]:
            aa = sorted(all_agents[agent], key=lambda u: u["start"])
            bb = sorted(all_agents[other], key=lambda u: u["start"])
            j = 0
            for a in aa:
                while j < len(bb) and bb[j]["end"] <= a["start"]:
                    j += 1
                for b in bb[j:]:
                    if b["start"] >= a["end"]:
                        break
                    if a["start"] < b["end"] and b["start"] < a["end"]:
                        counts["cross_agent_overlap_pairs"] += 1
    metadata["_agent_unit_counts"] = {a: len(u) for a, u in all_agents.items()}
    metadata["_background_agent_unit_counts"] = dict(Counter(
        u["agent"] or "_unspecified" for line in lines for u in line["background"]))
    return {"lines": lines, "metadata": dict(metadata), "counts": dict(counts), "warnings": warnings}


def unique_alignment(a, b, transitions):
    """Return only edges present unambiguously across ALL optimal monotonic paths.

    Each transition is (new_i,new_j,score,payload). Gap paths are also considered.
    No arbitrary traceback tie-break is allowed to assign repeated choruses.
    """
    n, m = len(a), len(b)
    forward = np.full((n + 1, m + 1), -np.inf)
    backward = forward.copy()
    forward[0, 0] = backward[n, m] = 0
    edges = {}
    for i in range(n + 1):
        for j in range(m + 1):
            es = transitions(i, j)
            edges[i, j] = es
            for ni, nj, score, _ in es:
                forward[ni, nj] = max(forward[ni, nj], forward[i, j] + score)
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            for ni, nj, score, _ in edges[i, j]:
                backward[i, j] = max(backward[i, j], score + backward[ni, nj])
    candidates, ar, br = [], defaultdict(set), defaultdict(set)
    best = forward[n, m]
    for (i, j), es in edges.items():
        for ni, nj, score, payload in es:
            if abs(forward[i, j] + score + backward[ni, nj] - best) > 1e-7:
                continue
            key = (i, j, ni, nj)
            for k in range(i, ni):
                ar[k].add(key)
            for k in range(j, nj):
                br[k].add(key)
            if payload:
                candidates.append((key, payload))
    return [(key, payload) for key, payload in candidates
            if all(len(ar[k]) == 1 for k in range(key[0], key[2]))
            and all(len(br[k]) == 1 for k in range(key[1], key[3]))]


def match_lines(refs, preds):
    a = [norm(r["text"]) for r in refs]
    b = [norm(p["text"]) for p in preds]

    def transitions(i, j):
        out = []
        if i < len(a):
            out.append((i + 1, j, -max(1, len(a[i])), None))
        if j < len(b):
            out.append((i, j + 1, -max(1, len(b[j])), None))
        for x, y in ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1)):
            if i + x > len(a) or j + y > len(b):
                continue
            aa, bb = "".join(a[i:i + x]), "".join(b[j:j + y])
            if not aa or not bb:
                continue
            equal = sum(k.size for k in SequenceMatcher(None, aa, bb, autojunk=False).get_matching_blocks())
            ratio = 2 * equal / (len(aa) + len(bb))
            if (x != 1 or y != 1) and aa != bb:
                continue
            if ratio >= .6:
                out.append((i + x, j + y, 4 * equal - len(aa) - len(bb), {"ratio": ratio}))
        return out

    return unique_alignment(a, b, transitions)


def match_units(refs, preds):
    """Common complete text boundaries; NEVER subdivide a human timestamp.

    Returns many-to-many common-boundary intervals. Strict one-reference-unit
    scores are reported separately from merged syllable/word intervals.
    """
    def flatten(units):
        texts = [norm(u["text"]) for u in units]
        bounds = [0]
        for t in texts:
            bounds.append(bounds[-1] + len(t))
        return "".join(texts), bounds

    a, ra = flatten(refs)
    b, pb = flatten(preds)
    if not a or not b:
        return []
    if a == b:
        mapping = dict(enumerate(range(len(a))))
    else:
        def transitions(i, j):
            out = []
            if i < len(a):
                out.append((i + 1, j, -1, None))
            if j < len(b):
                out.append((i, j + 1, -1, None))
            if i < len(a) and j < len(b):
                equal = a[i] == b[j]
                out.append((i + 1, j + 1, 2 if equal else -1, equal))
            return out
        mapping = {key[0]: key[1] for key, _ in unique_alignment(a, b, transitions)}
    pstarts = {v: i for i, v in enumerate(pb[:-1]) if pb[i + 1] > v}
    pends = {v: i + 1 for i, v in enumerate(pb[1:]) if v > pb[i]}
    groups, used = [], set()
    for i in range(len(refs)):
        if i in used or ra[i] == ra[i + 1]:
            continue
        start = mapping.get(ra[i])
        if start not in pstarts:
            continue
        for endi in range(i + 1, len(refs) + 1):
            end = start + ra[endi] - ra[i]
            if any(mapping.get(k) != start + k - ra[i] for k in range(ra[i], ra[endi])):
                break
            if end not in pends:
                continue
            j, endj = pstarts[start], pends[end]
            if a[ra[i]:ra[endi]] != b[start:end]:
                raise AssertionError("non-exact unit aggregation")
            groups.append((i, endi, j, endj))
            used.update(range(i, endi))
            break
    return groups


def metrics(rows):
    if not rows:
        return {"n": 0}
    result = {"n": len(rows)}
    for boundary in ("start", "end"):
        signed = np.array([r[f"{boundary}_error"] for r in rows])
        errors = np.abs(signed)
        result[boundary] = {"mae": float(errors.mean()), "median": float(np.median(errors)),
                            "p90": float(np.percentile(errors, 90)), "p95": float(np.percentile(errors, 95)),
                            "signed_bias": float(signed.mean()),
                            "exceed": {str(t): {"count": int((errors > t).sum()), "rate": float((errors > t).mean())}
                                       for t in (.1, .2, .3, .5, 1.)}}
    result["either_boundary_exceed"] = {
        str(t): {"count": sum(max(abs(r["start_error"]), abs(r["end_error"])) > t for r in rows),
                 "rate": sum(max(abs(r["start_error"]), abs(r["end_error"])) > t for r in rows) / len(rows)}
        for t in (.1, .2, .3, .5, 1.)}
    return result


def compare(reference, prediction, reserved=()):
    if "<redacted>" in json.dumps(prediction, ensure_ascii=False):
        raise ValueError("redacted predictions cannot be evaluated; rerun real CLI")
    refs = reference["lines"]
    preds = [p for p in prediction["lines"] if p.get("kind") == "singing"]
    modes = {p["index"]: p["mode"] for p in prediction["player_timeline"]}
    matches = match_lines(refs, preds)
    rows, mapped_refs, mapped_lines, fallback_refs = [], set(), set(), set()
    # Every reference remains in the denominator, including unmatched/fallback.
    for (i, j, ni, nj), info in matches:
        mapped_lines.update(range(i, ni))
        rr = [u for r in refs[i:ni] for u in r["lead"]]
        pp = []
        for p in preds[j:nj]:
            if p["source"] != "model":
                continue
            for u in p["units"]:
                if u.get("source") == "model":
                    pp.append({**u, "line": p})
        if all(modes.get(p["index"]) != "word" for p in preds[j:nj]):
            fallback_refs.update((u["line_index"], u["unit_index"]) for u in rr)
        for ri, rend, pi, pe in match_units(rr, pp):
            ru, pu = rr[ri:rend], pp[pi:pe]
            if len({u["agent"] for u in ru if u.get("agent")}) > 1:
                continue
            if any(a["start"] > b["start"] or a["end"] > b["end"]
                   for us in (ru, pu) for a, b in pairwise(us)):
                continue
            pl = {u["line"]["index"]: u["line"] for u in pu}
            accepted = all(modes.get(k) == "word" and not p.get("force_fallback") for k, p in pl.items())
            ids = [(u["line_index"], u["unit_index"]) for u in ru]
            mapped_refs.update(ids)
            if not accepted:
                fallback_refs.update(ids)
            row = {"reference_ids": ids, "prediction_lines": list(pl),
                   "text": "".join(u["text"] for u in ru), "reference_units": len(ru),
                   "prediction_units": len(pu), "reference_start": ru[0]["start"],
                   "reference_end": ru[-1]["end"], "prediction_start": pu[0]["start"],
                   "prediction_end": pu[-1]["end"], "accepted": accepted,
                   "language": "+".join(sorted({p.get("line_language") or prediction["language"] for p in pl.values()})),
                   "model": "+".join(sorted({p.get("route_model") or prediction["model"]["name"] for p in pl.values()})),
                   "quality": "+".join(sorted({p.get("quality") or "not_assessed" for p in pl.values()})),
                   "force_fallback": any(p.get("force_fallback") for p in pl.values()),
                   "role": "lead", "line_match_ratio": info["ratio"],
                   "reserved_anchor": any(u["line_index"] in reserved for u in ru)}
            row["span_duration_ratio"] = (
                (row["prediction_end"] - row["prediction_start"])
                / max(0.001, row["reference_end"] - row["reference_start"])
            )
            row["start_error"] = row["prediction_start"] - row["reference_start"]
            row["end_error"] = row["prediction_end"] - row["reference_end"]
            rows.append(row)
    all_refs = {(u["line_index"], u["unit_index"]) for r in refs for u in r["lead"]}
    eval_refs = {uid for uid in all_refs if uid[0] not in reserved}
    eval_matched = {uid for uid in mapped_refs if uid[0] not in reserved}
    eval_accepted = {uid for r in rows if r["accepted"] and not r["reserved_anchor"]
                     for uid in r["reference_ids"]}
    unmatched = [{"line": i, "unit": j, "reason": "LINE_UNMATCHED_OR_AMBIGUOUS" if i not in mapped_lines
                  else "PRODUCTION_FALLBACK_NO_COMPARABLE_MODEL_UNIT" if (i, j) in fallback_refs
                  else "TEXT_OR_GRANULARITY_UNMATCHED_OR_AMBIGUOUS"}
                 for i, j in sorted(all_refs - mapped_refs)]
    scored = [r for r in rows if not r["reserved_anchor"]]
    accepted = [r for r in scored if r["accepted"]]
    summary = {"reference_lines": len(refs), "reference_units": len(all_refs),
               "background_reference_units": sum(len(r["background"]) for r in refs),
               "background_metrics": {"n": 0, "reason": "no independently labelled background prediction track"},
               "agent_unit_counts": reference["metadata"].get("_agent_unit_counts", {}),
               "background_agent_unit_counts": reference["metadata"].get("_background_agent_unit_counts", {}),
               "cross_agent_overlap_pairs": reference["counts"].get("cross_agent_overlap_pairs", 0),
               "same_agent_overlap_pairs": reference["counts"].get("same_agent_overlap_pairs", 0),
               "singing_lines": len(preds), "model_lines": sum(p["source"] == "model" and bool(p["units"]) for p in preds),
               "production_word_lines": sum(modes.get(p["index"]) == "word" for p in preds),
               "production_fallback_lines": sum(modes.get(p["index"]) != "word" for p in preds),
               "matched_reference_units": len(mapped_refs), "text_match_rate": len(mapped_refs) / max(1, len(all_refs)),
               "score_eligible_reference_units": len(eval_refs),
               "score_matched_reference_units": len(eval_matched),
               "score_text_match_rate": len(eval_matched) / max(1, len(eval_refs)),
               "matched_accepted_reference_units": sum(r["reference_units"] for r in rows if r["accepted"]),
               "score_accepted_reference_units": len(eval_accepted),
               "score_accepted_reference_rate": len(eval_accepted) / max(1, len(eval_refs)),
               "model_generated_units": sum(len(p["units"]) for p in preds if p["source"] == "model"),
               "production_accepted_model_units": sum(len(p["units"]) for p in preds if p["source"] == "model" and modes.get(p["index"]) == "word"),
               "unmatched_reference_units": len(unmatched), "unmatched_reasons": dict(Counter(u["reason"] for u in unmatched)),
               "fallback_reference_units_lower_bound": len(fallback_refs),
               "matched_reference_lines": len(mapped_lines), "reserved_reference_lines": list(reserved),
               "excluded_anchor_intervals": len(rows) - len(scored),
               "all_model_common_boundary_intervals": metrics(scored),
               "accepted_common_boundary_intervals": metrics(accepted),
               "accepted_single_reference_unit_intervals": metrics([r for r in accepted if r["reference_units"] == 1]),
               "rejected_diagnostic_intervals": metrics([r for r in scored if not r["accepted"]]),
               "reliable_intervals": metrics([r for r in scored if r["quality"] == "reliable"]),
               "merged_reference_intervals": sum(r["reference_units"] > 1 for r in scored),
               "span_duration_ratio_outliers_lt_0_25_or_gt_4": sum(
                   r["span_duration_ratio"] < .25 or r["span_duration_ratio"] > 4 for r in scored)}
    for key in ("language", "quality", "model"):
        summary[f"by_{key}"] = {v: metrics([r for r in accepted if r[key] == v]) for v in sorted({r[key] for r in accepted})}
    # Error candidates, not diagnosed root causes. Exact details remain local.
    worst = {b: sorted(scored, key=lambda r: abs(r[f"{b}_error"]), reverse=True)[:10] for b in ("start", "end")}
    span_outliers = [r for r in scored if r["span_duration_ratio"] < .25 or r["span_duration_ratio"] > 4]
    line_errors = []
    for li in sorted({k for r in scored for k in r["prediction_lines"]}):
        rs = [r for r in scored if li in r["prediction_lines"]]
        biases = [r["start_error"] for r in rs]
        line_errors.append({"line": li, "n": len(rs), "start_bias": float(np.median(biases)),
                            "start_bias_mad": float(np.median(np.abs(np.array(biases) - np.median(biases)))),
                            "accepted_over_500ms": sum(r["accepted"] and max(abs(r["start_error"]), abs(r["end_error"])) > .5 for r in rs),
                            "duration_ratio_extremes": sum((r["prediction_end"] - r["prediction_start"]) / max(.001, r["reference_end"] - r["reference_start"]) > 3 for r in rs)})
    line_mapping = [{"reference_lines": list(range(i, ni)),
                     "prediction_lines": [p["index"] for p in preds[j:nj]], **info}
                    for (i, j, ni, nj), info in matches]
    return summary, {"rows": rows, "unmatched": unmatched, "worst": worst,
                     "line_errors": line_errors, "line_mapping": line_mapping,
                     "span_duration_outliers": span_outliers}


SONGS = {
    "DAMIDAMI": ("2755332551", "en", 50, 429),
    "预言 Prophecy": ("3402223603", "en", 58, 485),
    "不安灵魂收容所": ("2051582470", "zh", 54, 434),
    "提瓦特民谣": ("2750140001", "zh", 53, 469),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_suite(root):
    """Prepare reference-only anchor clips BEFORE loading prediction files.

    The first, middle and final reference lines are held out in every score.
    This is a listening template, NOT a claim that listening has taken place.
    """
    out = root / "experiments/amll_results"
    out.mkdir(parents=True, exist_ok=True)
    audits = {}
    page = ["<!doctype html><meta charset='utf-8'><title>AMLL 独立锚点试听</title>",
            ("<h1>待人工听音确认</h1><p>以下锚点仅由参考行顺序选择，未使用预测时间。"
             "请记录实际人声起音、版本是否一致、审核人及日期；这些整行不参与评分。</p>")]
    for song, (ident, _, expected_lines, expected_units) in SONGS.items():
        refpath = root / f"experiments/amll_reference/{ident}.ttml"
        ref = parse_ttml(refpath)
        if (len(ref["lines"]), sum(len(l["lead"]) for l in ref["lines"])) != (expected_lines, expected_units):
            raise ValueError(f"upstream counts changed for {song}; investigate before scoring")
        files = [p for p in (root / "Test").rglob("*")
                 if p.stem == song and p.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")]
        if len(files) != 1:
            raise ValueError(f"ambiguous exact audio match: {song}")
        audio = files[0]
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_format", "-of", "json", str(audio)]))
        duration = float(probe["format"]["duration"])
        reserve = sorted({0, len(ref["lines"]) // 2, len(ref["lines"]) - 1})
        anchors = []
        for li in reserve:
            line = ref["lines"][li]
            start = max(0, line["start"] - 2)
            clip = out / f"anchor_{ident}_{li}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-ss", str(start),
                            "-t", str(min(duration - start, line["end"] - start + 2)),
                            "-ac", "1", "-ar", "16000", str(clip)], check=True)
            anchors.append({"reference_line": li, "reference_start": line["lead"][0]["start"],
                            "clip_start": start, "clip": clip.name, "listened": False,
                            "human_onset_seconds": None, "reviewer": None, "notes": ""})
            page.append(f"<h2>{html.escape(song)} / reference line {li}</h2><p>"
                        f"clip starts at {start:.3f}s; reference onset {line['start']:.3f}s</p>"
                        f"<p>{html.escape(line['text'])}</p><audio controls src='{clip.name}'></audio>")
        audits[song] = {"status": "PENDING_HUMAN_AUDIO_VERIFICATION", "audio_sha256": digest(audio),
                        "audio": str(audio.relative_to(root)), "reference_sha256": digest(refpath),
                        "audio_duration": duration, "audio_tags": probe["format"].get("tags", {}),
                        "reference_metadata": ref["metadata"], "reference_first": ref["lines"][0]["start"],
                        "reference_last": max(l["end"] for l in ref["lines"]),
                        "reference_long_gaps": [{"after_line": i, "seconds": b["start"] - a["end"]}
                                                for i, (a, b) in enumerate(pairwise(ref["lines"]))
                                                if b["start"] - a["end"] > 5],
                        "parser_counts": ref["counts"], "reference_warnings": ref["warnings"],
                        "reserved_lines": reserve, "anchors": anchors,
                        "version_confirmed": False, "fixed_offset_seconds": None}
    # Preserve existing human entries. A changed input demands a new audit.
    dest = out / "version_audit.json"
    if dest.exists():
        old = json.loads(dest.read_text())
        for song, audit in audits.items():
            if old[song]["audio_sha256"] != audit["audio_sha256"] or old[song]["reference_sha256"] != audit["reference_sha256"]:
                raise ValueError("audit inputs changed; archive old audit before preparing")
    else:
        dest.write_text(json.dumps(audits, indent=2, ensure_ascii=False))
    (out / "listen.html").write_text("\n".join(page))


def evaluate_suite(root):
    out = root / "experiments/amll_results"
    audits = json.loads((out / "version_audit.json").read_text())
    manifest = json.loads((root / "experiments/amll_reference/manifest.json").read_text())
    summaries, details, pooled = {}, {}, []
    for song, (ident, _, _, _) in SONGS.items():
        refpath = root / f"experiments/amll_reference/{ident}.ttml"
        ref = parse_ttml(refpath)
        predpath = out / f"fullsong_amll_ad1d26c_{song.replace(' ', '_')}_production.json"
        pred = json.loads(predpath.read_text())
        audit = audits[song]
        if audit["status"] == "VERSION_MISMATCH":
            summaries[song] = {"status": "VERSION_MISMATCH", "metrics": None}
            continue
        if digest(refpath) != audit["reference_sha256"] or digest(root / audit["audio"]) != audit["audio_sha256"]:
            raise ValueError("audit inputs changed")
        if pred["diagnostics"]["quality_gate"]["mode"] != "production" or not pred["diagnostics"]["route_lines"]:
            raise ValueError("unexpected inference configuration")
        summary, detail = compare(ref, pred, audit["reserved_lines"])
        summary.update({"status": "PROVISIONAL_UNVERIFIED_AUDIO_VERSION", "offset_seconds": 0,
                        "corrected_metrics": None, "prediction_sha256": digest(predpath),
                        "audio_sha256": audit["audio_sha256"], "reference_sha256": digest(refpath),
                        "audio_duration": audit["audio_duration"], "reference_first": audit["reference_first"],
                        "reference_last": audit["reference_last"], "models": pred["diagnostics"]["routed_models"],
                        "quality_gate": pred["diagnostics"]["quality_gate"]})
        # Paired routing ablation at the SAME code/weights, never select a winner
        # for production. Newly covered groups are reported separately.
        offpath = out / f"fullsong_amll_routing_off_{song.replace(' ', '_')}_production.json"
        if offpath.exists():
            offsummary, offdetail = compare(ref, json.loads(offpath.read_text()), audit["reserved_lines"])
            eligible = lambda rows: {tuple(tuple(i) for i in r["reference_ids"]): r for r in rows
                                    if r["accepted"] and not r["reserved_anchor"]}
            on, off = eligible(detail["rows"]), eligible(offdetail["rows"])
            common = on.keys() & off.keys()
            summary["routing_ablation"] = {
                "off_prediction_sha256": digest(offpath), "off_summary": offsummary,
                "common_on": metrics([on[k] for k in sorted(common)]),
                "common_off": metrics([off[k] for k in sorted(common)]),
                "new_on": metrics([on[k] for k in sorted(on.keys() - off.keys())]),
                "lost_off": metrics([off[k] for k in sorted(off.keys() - on.keys())])}
            detail["routing_off"] = offdetail
        summaries[song], details[song] = summary, detail
        pooled.extend(r for r in detail["rows"] if r["accepted"] and not r["reserved_anchor"])
    result = {"baseline_commit": "ad1d26cf71883213e996058cf2f06e395b9ce930", "upstream": manifest,
              "status": "PROVISIONAL_UNVERIFIED_AUDIO_VERSION", "songs": summaries,
              "pooled_accepted": metrics(pooled),
              "pooled_by_language": {v: metrics([r for r in pooled if r["language"] == v]) for v in sorted({r["language"] for r in pooled})}}
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (out / "comparison_detailed.json").write_text(json.dumps(details, indent=2, ensure_ascii=False))
    for song, summary in summaries.items():
        print(song, json.dumps(summary.get("accepted_common_boundary_intervals"), ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ttml", type=Path)
    ap.add_argument("--prediction", type=Path)
    ap.add_argument("--out", type=Path, help="ignored LOCAL detailed JSON")
    ap.add_argument("--prepare-suite", type=Path, help="repository root; make independent listening anchors")
    ap.add_argument("--evaluate-suite", type=Path, help="repository root; evaluate the four fixed songs")
    ap.add_argument("--reserve-lines", default="", help="0-based independent listening anchor lines")
    args = ap.parse_args()
    if args.prepare_suite:
        prepare_suite(args.prepare_suite.resolve())
        return
    if args.evaluate_suite:
        evaluate_suite(args.evaluate_suite.resolve())
        return
    if not all((args.ttml, args.prediction, args.out)):
        ap.error("provide --ttml, --prediction and --out")
    reference = parse_ttml(args.ttml)
    prediction = json.loads(args.prediction.read_text())
    reserve = tuple(int(i) for i in args.reserve_lines.split(",") if i)
    summary, detail = compare(reference, prediction, reserve)
    report = {"status": "PROVISIONAL_UNVERIFIED_AUDIO_VERSION", "offset_seconds": 0,
              "offset_corrected_metrics": None, "reference_sha256": hashlib.sha256(args.ttml.read_bytes()).hexdigest(),
              "prediction_sha256": hashlib.sha256(args.prediction.read_bytes()).hexdigest(),
              "summary": summary, "detail": detail}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
