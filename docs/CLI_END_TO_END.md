# CLI 端到端验证（CLI_END_TO_END） — V0.8

> 从公开 CLI 入口验证完整链路：LRC → 模型选择 → 音频推理 → 质量决策 → JSON 导出 → `player_timeline`。
> **测试替身（FakeAligner）与真实模型推理严格区分**。

## 1. 后端选择与元信息（`autosync info`，真实运行）

```
[ja]  default=hubert-phoneme-ctc-v4
  hubert-phoneme-ctc-v4 (default)
      repo=prj-beatrice/japanese-hubert-base-phoneme-ctc-v4
      status=experimental  license=apache-2.0  distributable=yes
  pyshiro-v2        status=research  license=GPLv3 code / weights: no explicit licence  distributable=no
  reazonspeech-wav2vec2-base  status=experimental  license=apache-2.0  distributable=yes
```

## 2. 真实 Jaeger/日语 CLI 运行（真实模型推理）

命令（默认后端 = HuBERT）：

```
autosync align --audio <gtsinger segment>.wav --lyrics <line>.lrc --language ja --output out.json -q
```

| 片段 | backend | status | quality | player_timeline mode |
|---|---|---|---|---|
| heartful song（女高） | prj-beatrice/japanese-hubert-base-phoneme-ctc-v4 | experimental | uncertain=1 | **line_lrc（回退）** |
| さんぽ（男高，已知严重） | 同上 | experimental | **reliable=1** | **word** ⚠️ |

**⚠️ 诚实记录**：`さんぽ` 被质量门控**误判为 reliable**（false accept）——这正是 V0.7/V0.8 反复
证实的「门控不可靠」问题。默认保守阈值（`max_intra_char_span_s=1.5`）未能覆盖该行。
我们不隐藏该失败样本；生产模式仍会在多数不可靠行回退，但不能保证全部。

显式选择旧基线：

```
autosync align --language ja --backend reazonspeech-wav2vec2-base ...
# backend: reazon-research/japanese-wav2vec2-base-rs35kh  status=experimental
```

## 3. 真实英文 CLI 运行（stable）

```
autosync align --audio synthetic_en.wav --lyrics synthetic_en.lrc --language en --output out.json -q
# backend: facebook/wav2vec2-base-960h  status=stable  license=apache-2.0
# quality: not_assessed=1   (en 不启用门控)
```

- `player_timeline[0].mode == "word"`，`lrc_start == 0.0`，`lrc_end == null`。
- 精度回归：`start_mae = 0.0672`、`end_mae = 0.3936`（与 V0.1 起一致，**未退化**）。

## 4. 测试替身验证（控制流，非真实推理）

`tests/test_cli_integration.py` 用 monkeypatch 注入 `FakeAligner`，验证：

- CLI 参数解析 → 模型构造/`load()` → `align_lines` → 质量门控 → 导出；
- 导出含 `model.name/status`、`schema_version=0.2`、`quality_summary`、`player_timeline`；
- `reliable` → `word`；`uncertain` → `line_lrc`；启发式（en，不门控）→ `word`；
- 原始歌词文字与 `lrc_start`/`lrc_end` 端到端保留。

> 这些测试**不**声称真实推理成功；真实推理证据见 §2/§3。

## 5. 复现

```bash
uv run autosync info
uv run autosync align --audio SONG.wav --lyrics SONG.lrc --language ja --output out.json
uv run autosync align --audio SONG.wav --lyrics SONG.lrc --language ja \
  --backend reazonspeech-wav2vec2-base --output out_base.json
uv run autosync align --audio SONG.wav --lyrics SONG.lrc --language ja --mode experimental --output out_exp.json
uv run --extra dev pytest -q tests/test_cli_integration.py tests/test_fallback_timestamps.py
```

## 6. 仍未完成

- **真实整曲**（多行 + 伴奏 + 长时轴）CLI 验证：缺授权素材，未完成；已有严格合成多行测试
  （`tests/test_song_chain.py`）。
- ja HuBERT 为 experimental：生产模式应视为**实验候选**，逐字时间需显式实验模式或更高置信。
