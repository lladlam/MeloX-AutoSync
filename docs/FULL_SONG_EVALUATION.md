# 六首真实歌曲完整评测（FULL_SONG_EVALUATION） — V1.0 P0/P3

> 本地素材：用户自备的 6 首商业歌曲（`Test/`，**已 gitignore，未进入公开仓库**）。
> 全部通过**真实 CLI**（`autosync align`）推理；评测脚本：`benchmarks/full_song_eval.py`；
> 原始输出：`experiments/reports/full_song_eval_{baseline,v1b,v1b_ja}.json`。
> **不提交任何第三方音频或完整歌词。**

## 1. 数据清单（P0）

| 歌曲 | 语言（声明） | 时长 | LRC 行数 | LRC 范围 | 编码 |
|---|---|---|---|---|---|
| DAMIDAMI | en | 191.0s | 62 | 0–171.3s | UTF-8 |
| 预言 Prophecy | en | 254.4s | 86 | 0–244.6s | UTF-8 |
| SAN値直葬 | ja | 145.7s | 73 | 0–129.9s | UTF-8 |
| 歌姫失格 | ja | 200.4s | 73 | 0–195.3s | UTF-8 |
| 不安灵魂收容所 | zh（含英文段） | 170.9s | 58 | 0–164.0s | UTF-8 |
| 提瓦特民谣 | zh（含英文段） | 217.5s | 73 | 0–216.6s | UTF-8 |

音频↔LRC 配对：**精确 stem 相等优先，其次前缀（`stem-`/`stem (`）**；6 首均唯一匹配，无歧义
（歧义会直接报错，不做模糊匹配）。语言为**显式声明**（中文歌含英文段，脚本计数会误判）。

## 2. 时间轴合法性（两轮全部通过）

| 指标 | 基线(V0.9) | V1b | 说明 |
|---|---|---|---|
| 跨行交叉 | **0** | **0** | 上一演唱行 units.end ≤ 下一演唱行 `lrc_start`+padding |
| 越界单元 | **0** | **0** | 所有单元在音频时长内 |
| 非单调/重叠 | **0** | **0** | 逐行 units 严格递增 |
| 原文与 `lrc_start` 保留 | 6/6 | 6/6 | 逐行比对，完全一致 |
| 非演唱行产生逐字 | 0 | 0 | metadata/instrumental 一律无 units |

## 3. 基线 vs 修复后（production，真实 CLI）

| 歌曲 | 语言 | 基线 model/fail/word | **V1b model/fail/word** | 变化 |
|---|---|---|---|---|
| DAMIDAMI | en | 51 / 0 / 51 | 51 / 0 / 51 | — |
| 预言 Prophecy | en | 62 / 0 / 62 | 62 / 0 / 62 | — |
| SAN値直葬 | ja | 56 / 0 / 34 | 56 / 0 / 34 | — |
| 歌姫失格 | ja | 71 / 0 / 55 | 71 / 0 / 55 | — |
| **不安灵魂收容所** | zh | 39 / **15** / 39 | **49 / 5 / 49** | 英文段落改由英文模型对齐 |
| **提瓦特民谣** | zh | 45 / **9** / 45 | **50 / 4 / 50** | 英文行路由；4 行拉丁含糊保守回退 |

- 6 首均 **rc=0**，无崩溃、无跨行、无越界。
- 单曲耗时 18–30s，峰值 RSS 1035–1429 MB（CPU 单进程）。
- 非演唱行分类（V0.9）：173 行中 40 metadata + 37 instrumental，**0 unknown**（详见 §6）。

## 4. 窗口策略（P1 修复）

- 新增**保守窗口上限**：`max_line_seconds=12s`；若截断点仍在**有声**（长音），
  向前延伸到连续有声段结束，**硬上限 30s**。
- 实测 `window_seconds_max`：**12.3s**（长间奏被截断），长音场景不超过 30.4s（单元测试覆盖）。
- 每行的实际输入音频时长记录在结果的 `window_seconds` 字段（可审计）。
- 长尾奏（末行之后）同样被截断：`预言` 末行窗口 ≤11.15s（音频 254s）。

## 5. 情况专项

| 情况 | 结果 |
|---|---|
| 拉丁字母行（zh 歌） | 15/9 行失败 → **5/4**；其余为**显式保守回退**（无伪造时间） |
| 重复歌词 | `DAMIDAMI`/`预言` 副歌重复行逐行独立对齐，文本保留 |
| 间奏/前奏/尾奏 | instrumental 37 行跳过；间奏处窗口被 12s 上限截断 |
| 失败行 | 保留原文与 `lrc_start`，`mode=line_lrc`，`error` 说明原因 |
| 日文 production/experimental | `SAN値直葬` word 34→**56**、`歌姫失格` word 55→**71**（诊断时间轴可见） |

## 6. 分类质量（P5 审查）

6 首共 **348 singing / 40 metadata / 37 instrumental / 0 unknown**（V0.9 数字复核一致）。

人工审查（脚本 + 人工抽查）：

- **被误跳过的演唱行：0**（无 singing 行含署名关键词或冒号格式）；
- **被判 singing 的署名行：0**（40 条 metadata 全部为制作者/演奏/录音署名）；
- 唯一“长 metadata”命中是 `预言` 的 `和声 : …（7 位歌手）`——**本就是署名**，非误判；
- `UNKNOWN` 本轮 0 条；其**安全回退**由单测覆盖（`tests/test_line_classify.py`）；
- 署名行**不截断**相邻歌词窗口（单测 `test_skipped_metadata_does_not_truncate_window`，实测窗口延伸到下一演唱行）。

## 7. 准确率：**尚未人工评测**（P4）

**没有人工确认的参考边界，就没有可报告的真实精度。** 本轮完成的是：

- 生成标注模板：`tools/make_annotation_template.py template`（每首最多 30 个代表单元，
  含首尾行、最长/最短单元、重复歌词与随机样本）→ `experiments/annotation/annotation_template.csv`（300 行，**已 gitignore**）；
- 评分器：`tools/make_annotation_template.py score --csv <filled.csv>`，输出 start/end MAE、P90、
  >300/>500/>1000ms 比例（按语言/歌曲分组）；**未填写的行计为 unannotated 并排除**，绝不当成正确。

> 合成歌声/语音的既有精度（英文合成 start MAE 67ms、中文歌声 start MAE 141ms 等）见
> `docs/EXPERIMENT_RESULTS.md`；**不得**将其当作真实歌曲精度。

## 8. 复现

```bash
# 基线（V0.9 行为）：production，无逐行路由
.venv/bin/python benchmarks/full_song_eval.py --tag baseline --modes production
# 修复后：逐行语言路由
.venv/bin/python benchmarks/full_song_eval.py --tag v1b --modes production --routing
# 日文 production / experimental 对照
.venv/bin/python benchmarks/full_song_eval.py --tag v1b_ja --modes production,experimental \
    --routing --only SAN値直葬,歌姫失格
# 人工标注模板 + 评分
.venv/bin/python tools/make_annotation_template.py template \
    --results experiments/reports/fullsong_v1b_*.json --out experiments/annotation/annotation_template.csv
.venv/bin/python tools/make_annotation_template.py score --csv experiments/annotation/annotation_template.csv
```
