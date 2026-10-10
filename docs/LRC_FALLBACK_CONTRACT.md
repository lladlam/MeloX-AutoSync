# LRC 回退契约（LRC_FALLBACK_CONTRACT） — V0.8

> 定义 AutoSync 输出中**时间字段的语义**、**质量状态到播放器模式的映射**，以及
> **回退保证**。任何消费方（含未来 Android Library）都应遵循本契约。
> JSON schema 版本：`0.2`。

## 1. 每行的独立时间字段

`player_timeline[i]`：

| 字段 | 含义 | 何时为 null |
|---|---|---|
| `text` | **原始歌词文字**（永不丢失/改写） | 从不 |
| `lrc_start` | **原始 LRC 行开始时间** | 从不（缺省时回退到该行解析时间） |
| `lrc_end` | **原始 LRC 行结束时间** | LRC 未给出显式结束时为 `null`（**不伪造**） |
| `model_start` | 模型预测行开始 | 未对齐（failed/heuristic 无单元）时为 `null` |
| `model_end` | 模型预测行结束 | 同上 |
| `display_end` | **派生**的显示截止提示（下一行的 `lrc_start`） | 末行或下一行无原始时间为 `null` |
| `display_end_source` | 派生来源标记（`derived_from_next_lrc_start`） | 与 `display_end` 同为 null |
| `mode` | `word`（逐字）或 `line_lrc`（整行） | 从不 |
| `source` | `model` / `heuristic` / `failed` | 从不 |
| `quality` | `reliable` / `uncertain` / `failed` / `not_assessed` | 从不 |
| `reasons` | 质量决策原因 | 空列表 |
| `units` | 逐字单元（仅 `mode=word`） | `mode=line_lrc` 时不出现 |

**关键**：`lrc_start`/`lrc_end` 与 `model_start`/`model_end` **互相独立**，绝不混用。
`display_end` 是派生值，**不是**原始 LRC 的结束时间。

## 2. 质量状态 → 播放器模式

| 质量状态 | `production` 模式（默认） | `experimental` 模式 |
|---|---|---|
| `reliable` | `word` | `word` |
| `uncertain` | **`line_lrc`**（回退） | `word`（诊断） |
| `failed` | **`line_lrc`**（回退） | **`line_lrc`**（永不输出逐字） |
| `not_assessed` + 启用门控 | **`line_lrc`**（安全回退） | 同左 |
| `not_assessed` + 未启用门控（en/zh） | `word`（保持） | `word` |

- 日语**启用门控**；英文/中文**不启用**（各自策略独立）。
- 当门控启用且模型行**未评估**（质量分析缺失/异常）→ 视为 `uncertain`，安全回退。

## 3. 回退保证

1. `line_lrc` 使用 **`lrc_start`（及 `lrc_end`，若存在）与原始 `text`**，即**原始 LRC 的时间与文字**。
2. 原始歌词与原始时间**永不被覆盖、删除或改写**。
3. `FAILED` 行附带结构化 `reasons`（如 `missing_ratio=...`），并计入 `quality_summary.failed`。
4. 逐行回退（同一行不混排逐字/逐行），避免视觉闪烁。

## 4. 顶层字段

```
schema_version : "0.2"
generator      : "MeloX-AutoSync"
language, engine
model          : {name, version, language, license, status, device, load_seconds, ...}
quality_summary: {reliable, uncertain, failed, not_assessed}
diagnostics    : {quality_gate: {enabled, policy, mode}, backend, padding_seconds, ...}
lines[]        : 诊断用完整行记录（含 lrc_start/lrc_end/units/quality/...）
player_timeline[] : 播放器时间轴（见 §1）
```

## 5. 示例

```jsonc
{
  "index": 1,
  "text": "second line",
  "lrc_start": 6.0,
  "lrc_end": null,                       // 普通 LRC 无显式结束
  "model_start": 5.72,                   // 模型预测（与 LRC 独立）
  "model_end": 6.31,
  "display_end": 11.0,                   // 由下一行 lrc_start 派生
  "display_end_source": "derived_from_next_lrc_start",
  "mode": "line_lrc",                    // uncertain -> 回退
  "source": "model",
  "quality": "uncertain",
  "reasons": ["max_intra_char_span_s=2.10 > 1.5"]
}
```

## 6. 兼容与迁移

- `schema_version` 变更时需提供迁移说明；消费方应忽略未知字段。
- 早期（`0.1`）结果缺少 `lrc_start`/`player_timeline`/`quality_summary`；`0.2` 起稳定提供。
- 决策逻辑与阈值属于**算法层**，不在本契约内固定；消费方只依赖字段语义与模式映射。
