# 对齐质量与置信度（ALIGNMENT_CONFIDENCE） — V0.7

> 目标：判断一次强制对齐是否**可靠**，在缺乏声学证据时**拒绝**输出貌似准确的逐字时间轴。
> 代码：`src/autosync/quality/`；评测：`benchmarks/confidence_eval.py`；
> 结果：`experiments/reports/confidence_eval.json`、`docs/SELECTIVE_ALIGNMENT_REPORT.md`。
> **不修改 V0.5 的 CTC 默认路径**——质量评估只读地分析其结果。

## 1. 质量特征（P0 §1）

`QualityFeatures`（`quality/features.py`）从**模型自身输出**计算，不使用任何参考标注：

| 组别 | 特征 | 说明 |
|---|---|---|
| 路径得分 | `score_margin_total/per_token/per_frame` | 强制路径相对「全 blank 路径」的对数概率优势 |
| | `mean_token_logp` | token 帧上目标音素的平均 log 概率 |
| | `mean_peak_advantage` | token 峰值帧上「目标音素 − 最强竞争类」的差距 |
| 路径构成 | `blank_frame_ratio` | 路径分配给 blank 的帧比例 |
| | `blank_dominant_ratio` | blank 概率 >0.5 的帧比例（**不单独作为质量分**） |
| | `longest_blank_run_s` | 最长 blank 主导连续段（秒） |
| 目标完整性 | `missing_ratio`、`sparse_ratio` | 无 span / ≤1 帧的 token 比例 |
| | `free_decode_agreement` | 自由解码（collapse）与目标音素序列的 1−归一化编辑距离 |
| 时间合理性 | `max_inter_token_gap_s` | 相邻 token 最大间隔 |
| | `mean/max_intra_char_span_s` | 同一字符内音素跨度 |
| | `char_duration_max/median/ratio` | 字符时长异常度 |
| | `token_duration_cv` | token 时长离散度 |

**注意**：CTC 平均 blank 概率不单独作为失败判定；对齐得分也不等价于校准后的正确率。

## 2. 三状态决策（P1 §3）

`QualityState` = `RELIABLE` / `UNCERTAIN` / `FAILED`；`QualityPolicy` 提供阈值；
`decide()` 返回状态、原因列表与特征。

- **FAILED（结构性失败）**：`n_tokens==0` 或 `missing_ratio ≥ fail_missing_ratio`（默认 0.5）。
- **UNCERTAIN（声学依据不足）**：任一阈值被违反（`free_decode_agreement`、
  `score_margin_per_token`、`mean_peak_advantage`、`blank_dominant_ratio`、
  `max_intra_char_span_s`、`char_duration_ratio`、`max_inter_token_gap_s`）。
- **RELIABLE**：全部满足。

默认策略保守（`ja-hubert-conservative-v07`：`max_intra_char_span_s=1.5s`、`char_duration_ratio=10`）。
**未采用**在整套 30 段上挑出的「最优」阈值（避免过拟合与虚假的独立测试结论）。

## 3. 保守回退（P1 §4）

`quality/gating.py`：

- `player_timeline(result)`：`RELIABLE`（或未门控语言）→ `mode="word"`；
  `UNCERTAIN` → `mode="line_lrc"`（保留诊断时间轴，播放器用**原始逐行 LRC**）；
  `FAILED` → `mode="line_lrc"` + 结构化失败原因。
- **逐行回退**（同一行不做逐字/逐行混排，避免视觉闪烁）。
- **原始歌词永不被覆盖/删除**：`player_timeline` 始终包含 `text`、`line_start`、`line_end`。

英文与中文**不启用**日语门控（`DEFAULT_POLICIES` 中 `en/zh = None`），其 `mode="word"` 保持不变。

## 4. JSON / CLI 集成

导出新增（schema `0.2`）：

```
quality_summary : {reliable, uncertain, failed, not_assessed}
player_timeline : [{index, text, line_start, line_end, mode, quality, reasons, units?}]
lines[i].quality / lines[i].quality_reasons
```

CLI 结束时打印质量计数。英文/中文结果不受影响（回归见 §6）。

## 5. 指标有效性（P0 §2，实测）

`benchmarks/confidence_eval.py` 在 30 段日语上（**探索性**，2 位歌手）：

| 特征 | 检测「行内极值误差>1.0s」的 AUC |
|---|---|
| `max_intra_char_span_s` | **0.665** |
| `char_duration_ratio` | **0.665** |
| `mean_token_logp` | 0.625 |
| `free_decode_agreement` | 0.605 |
| `blank_dominant_ratio` | 0.590 |
| `score_margin_per_token` | 0.515 |

**时间合理性类特征**（字符内跨度过大）最有效；原始后验类特征判别力有限。
详细检测/选择性指标与风险—覆盖率见 `docs/SELECTIVE_ALIGNMENT_REPORT.md`。

## 6. 回归

英文 `facebook/wav2vec2-base-960h` `lrc_only` sMAE **0.138**（未变）；
中文 `funasr/fa-zh` sMAE **0.141–0.142**（未变）。质量接口按语言独立，三语言包仍按需加载。
