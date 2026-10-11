"""Tests for the V1.1 quantisation tooling (playlist dataset + selective quant)."""

from __future__ import annotations

import build_playlist_dataset as bp
import numpy as np
import selective_quantize as sq


# --------------------------------------------------------------------------- dataset
def test_is_credit_recognises_credits_not_lyrics():
    assert bp.is_credit("作词 : 雷声")
    assert bp.is_credit("Lyrics by Someone")
    assert bp.is_credit("ミックス : 誰か")
    assert not bp.is_credit("船 摇啊摇")
    assert not bp.is_credit("初音ミクを引退します")


def test_detect_language_by_script():
    assert bp.detect_language(["いつも一つだけ", "幸せなんて"])[0] == "ja"
    assert bp.detect_language(["红透晚烟青", "摇到陌生的远方"])[0] == "zh"
    assert bp.detect_language(["going along slushy country roads"])[0] == "en"
    assert bp.detect_language(["不安灵魂收容所在", "we brave"])[0] == "mixed"


def test_parse_lrc_handles_colon_centiseconds():
    entries, meta = bp.parse_lrc(_write(tmp_text="[00:05:32]みなさん\n[ti:Demo]\n"))
    assert len(entries) == 1
    assert abs(entries[0][0] - 5.32) < 1e-6
    assert entries[0][1] == "みなさん"
    assert meta["ti"] == "Demo"


def test_match_lyrics_prefix_and_near_prefix(tmp_path):
    audio = tmp_path / "DAMIDAMI.mp3"
    audio.write_bytes(b"x")
    lyr = tmp_path / "DAMIDAMI-Sihan,三Z-STUDIO.lrc"
    lyr.write_text("[00:01.00]hi\n", encoding="utf-8")
    assert bp.match_lyrics(audio, [lyr]) == lyr

    # CJK audio stem shorter than 3 chars must still match
    a2 = tmp_path / "绘本.mp3"
    a2.write_bytes(b"x")
    l2 = tmp_path / "绘本-三Z-STUDIO.lrc"
    l2.write_text("[00:01.00]x\n", encoding="utf-8")
    assert bp.match_lyrics(a2, [l2]) == l2

    # one-character difference (游记 vs 游计) still matches
    a3 = tmp_path / "乐园梦游记.mp3"
    a3.write_bytes(b"x")
    l3 = tmp_path / "乐园梦游计-三Z-STUDIO.lrc"
    l3.write_text("[00:01.00]x\n", encoding="utf-8")
    assert bp.match_lyrics(a3, [l3]) == l3


def test_assign_splits_disjoint_and_deterministic():
    recs = [{"song_id": f"s{i}", "language": "en"} for i in range(6)]
    s1 = bp.assign_splits(recs)
    s2 = bp.assign_splits(recs)
    assert s1 == s2
    assert set(s1.values()) <= {"tune", "val", "blind"}
    assert sum(v == "blind" for v in s1.values()) >= 1
    # every song gets exactly one split
    assert len(s1) == 6


# --------------------------------------------------------------------------- quantise
def _tiny_onnx(path, node_names):
    import onnx
    from onnx import TensorProto, helper

    w = np.random.RandomState(0).randn(8, 8).astype(np.float32)
    inits, nodes = [], []
    for i, name in enumerate(node_names):
        wn = f"w{i}"
        inits.append(helper.make_tensor(wn, TensorProto.FLOAT, [8, 8], w.flatten()))
        nodes.append(helper.make_node("MatMul", [f"in{i}", wn], [f"out{i}"], name=name))
    graph = helper.make_graph(
        nodes, "t", [helper.make_tensor_value_info(f"in{i}", TensorProto.FLOAT, [1, 8])
                     for i in range(len(node_names))],
        [helper.make_tensor_value_info(f"out{i}", TensorProto.FLOAT, [1, 8])
         for i in range(len(node_names))], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def test_selective_quantize_excludes_named_nodes(tmp_path):
    src = tmp_path / "m.onnx"
    _tiny_onnx(src, ["/model/lm_head/MatMul", "/model/keep/MatMul"])
    out = tmp_path / "m_q.onnx"
    rec = sq.build(src, out, ["lm_head"], [], True, ["MatMul", "Gemm"])
    assert rec["n_quantizable"] == 2
    assert rec["n_excluded"] == 1
    assert out.stat().st_size > 0


def test_selective_quantize_regex_and_weight_suffix(tmp_path):
    src = tmp_path / "m.onnx"
    _tiny_onnx(src, ["/e/layers.0/attention/MatMul", "/e/layers.1/attention/MatMul", "/e/ff/MatMul"])
    out = tmp_path / "m_q.onnx"
    rec = sq.build(src, out, [], [r"layers\.0/attention"], True, ["MatMul", "Gemm"])
    assert rec["n_excluded"] == 1
    # weight-suffix exclusion must find the consuming node by initializer name
    import onnx

    m = onnx.load(str(src), load_external_data=False)
    w0 = m.graph.initializer[0].name
    rec2 = sq.build(src, out, [], [], True, ["MatMul", "Gemm"], weight_suffixes=[w0])
    assert rec2["n_excluded"] == 1


# --------------------------------------------------------------------------- helper
def _write(tmp_text: str):
    import tempfile
    from pathlib import Path

    p = Path(tempfile.mkdtemp()) / "x.lrc"
    p.write_text(tmp_text, encoding="utf-8")
    return p
