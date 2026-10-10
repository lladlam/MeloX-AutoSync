# 引擎整合审计（ENGINE_INTEGRATION_AUDIT） — V0.8

> 本轮为**正确性/工程**修复，不追求实验分数。全部改动均有测试与真实 CLI 运行证据。

## P0.1 模型注册表与 CLI 不一致 — 已修复

**问题**：`get_aligner("ja")` 仍返回 ReazonSpeech；研究最佳日语模型是 HuBERT Phoneme CTC。

**修复**：`models/registry.py` 改为**显式后端注册表**：

| 语言 | 后端 key | repo | 状态 | 许可 | 可分发 |
|---|---|---|---|---|---|
| en | `wav2vec2-base-960h`（默认） | facebook/wav2vec2-base-960h | stable | Apache-2.0 | 是 |
| **ja** | **`hubert-phoneme-ctc-v4`（默认）** | prj-beatrice/japanese-hubert-base-phoneme-ctc-v4 | experimental | Apache-2.0 | 是 |
| ja | `reazonspeech-wav2vec2-base` | reazon-research/… | experimental | Apache-2.0 | 是 |
| ja | `pyshiro-v2` | pyshiro-jp-v2 | research | GPLv3 / 权重许可不明 | **否** |
| zh | `funasr-fa-zh`（默认） | funasr/fa-zh | stable | FunASR Model License | 是 |

- CLI 新增 `--backend <key>` 与 `--mode production|experimental`；`autosync info` 打印
  真实 repo / 状态 / 许可 / 是否可分发。
- 运行日志打印 `backend: <repo> status=<..> license=<..>`；JSON 的 `model.status` 记录状态。
- **生产兼容模式**默认**安全回退**（非 RELIABLE → 整行 LRC）；实验模式可显式输出诊断逐字时间。
- 按需加载、语言包独立不变（只构造被请求的后端）。

## P0.2 原始 LRC 时间未保留 — 已修复

**问题**：`player_timeline` 把模型预测时间当作原始 LRC 行时间；回退时用的是模型时间。

**修复**：`AlignedLine` 新增 `lrc_start` / `lrc_end`（来自原始 LRC，`segment.py` 与启发式分支均设置）。
`player_timeline` 现在独立输出：`lrc_start`、`lrc_end`（无显式结束则为 **null**，绝不伪造）、
`model_start`、`model_end`、`display_end`（+`display_end_source`，由下一行原始 LRC 开始时间派生，明确标记）。
回退（UNCERTAIN/FAILED）使用**原始 LRC 时间与文字**。见 `docs/LRC_FALLBACK_CONTRACT.md`。

## P0.3 质量评估失败时错误放行 — 已修复

**问题**：质量分析异常 → `last_quality=None` → 被当成允许逐字显示。

**修复**：
- `NOT_ASSESSED` 与 `RELIABLE` 明确区分；`gate_result(policy)` 在**启用门控**时，把**未评估的模型行**
  标为 `UNCERTAIN`（安全回退）。
- `player_timeline` 规则：显式 `uncertain` 仅在**实验模式**允许诊断输出；`failed` 永不；未评估仅在**未启用门控**时按字输出。
- `quality_summary` 在**门控决策之后**统计；结构性失败精确计入 `failed`。
- 中英文门控由独立策略控制（`DEFAULT_POLICIES` 中 `en/zh=None`）。
- 测试：`tests/test_fallback_timestamps.py`（含“质量返回 None 且有门控 → 回退”）。

## P0.4 交叉验证特征选择泄漏 — 已修复

**问题**：`confidence_eval.py` 在全量数据上选 primary feature，再让各折使用。

**修复**：**每个折只在训练子集上同时选择特征与阈值**；测试子集只用于评分。
风险—覆盖率使用**预先指定**的特征 `max_intra_char_span_s`（不据评估数据挑选）。

**更正后的结果（与 V0.7 泄漏版对比）**：

| 验证 | V0.7（泄漏，feature 全局选） | **V0.8（折内选择）** |
|---|---|---|
| leave-one-singer-out F1 | 0.65 | **0.58** |
| leave-one-song-out F1 | 0.67 | **0.59** |

→ **结论未改变且更强**：质量门控判别力不足。V0.7 的数字偏乐观，已作废为对比项
（`experiments/reports/confidence_eval_v07_leaky.json`）。

## P1 其他时间轴风险 — 审计与测试

| 风险 | 状态 | 测试 |
|---|---|---|
| 启发式时间转换到绝对时间 | ✅ | `test_cli_heuristic_engine_marks_not_assessed`、`test_heuristic.py` |
| 单行失败不影响其他行 | ✅ | `test_one_failing_line_does_not_affect_others` |
| 文本重建失败仍进逐字输出 | ✅（reconstruction 校验 + 失败行回退） | `test_mapping.py`、`test_song_chain.py` |
| 单元时间越界/相邻行交叉 | ✅ | `test_lines_are_monotonic_and_do_not_cross` |
| 逐字开始/结束单调 | ✅ | `test_units_are_monotonic_within_each_line` |
| 空歌词行安全 | ✅ | `test_empty_lyric_line_is_safe` |
| LRC offset / 多时间戳 / 增强 LRC | ✅ | `test_lrc.py`、`test_lrc_timestamp_offset_is_honoured` |

## 遗留问题

- 质量门控**已确认不可靠**（`docs/SELECTIVE_ALIGNMENT_REPORT.md`）；本轮仅用于保守回退。
- 真实整曲（多行 + 伴奏 + 长时轴）测试仍未完成（缺授权整曲素材）；已有严格合成多行测试。
- ja HuBERT 仍是 experimental：生产模式默认安全回退，逐字时间需显式实验模式或更高置信。
