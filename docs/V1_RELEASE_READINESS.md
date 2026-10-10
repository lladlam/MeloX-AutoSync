# V1.0 发布就绪评估（V1_RELEASE_READINESS）

> 依据：六首真实歌曲端到端运行（`docs/FULL_SONG_EVALUATION.md`）、混合语言路由
> （`docs/MIXED_LANGUAGE_ROUTING.md`）、质量门控结论（`docs/SELECTIVE_ALIGNMENT_REPORT.md`）。
> 基准提交：`bb8f1dd` → 本轮提交见文末。

## 0. 证据概览

| 项目 | 结果 |
|---|---|
| 真实整曲（真实 CLI） | **6/6 完成**，rc=0，无崩溃 |
| 时间轴合法性 | 跨行交叉 **0**、越界 **0**、非单调 **0**（6 首 × 两轮） |
| 原文/原始 LRC 时间保留 | **6/6 逐行一致** |
| 非演唱行 | 40 metadata + 37 instrumental 全部跳过，0 误跳（人工审查） |
| 混合语言路由 | zh 歌失败行 15→**5**、9→**4**；ja/en 歌 0 失败 |
| 逐字覆盖（production） | **301/348 演唱行（86.5%）** 输出 word；47 行按质量/路由安全回退 |
| 逐字覆盖（ja experimental） | 301 + 38 = **339/348（97.4%）**（诊断时间轴） |
| 单曲耗时 / 峰值 RSS | 18–30 s / 1035–1429 MB（CPU） |
| 自动化测试 | **178 项**，ruff 全绿 |

## 1. 六首歌曲是否全部完成真实推理？

**是。** 6 首（DAMIDAMI、预言 Prophecy、SAN値直葬、歌姫失格、不安灵魂收容所、提瓦特民谣）
均通过公开 CLI `autosync align` 真实推理，两轮（V0.9 基线 + V1 修复后）以及日文
production/experimental 对照，全部 rc=0。素材为本地授权文件，**未提交任何音频或完整歌词**。

## 2. V0.9 的两个整曲测试结论是否依然成立？

**成立且被加强。**
- `歌姫失格`：73 行、0 失败、原文与 `lrc_start` 完全保留、0 跨行 —— 与 V0.9 一致。
- `提瓦特民谣`：V0.9 的 9 行拉丁失败 → 现为 **4 行显式回退 + 5 行由英文模型成功对齐**；
  时间轴仍 0 跨行/越界。V0.9 的“非演唱行分类”与“非法边界为 0”结论在本轮 6 首上继续成立。

## 3. 修复后有多少歌词行真正获得逐字时间？

- production：**301 / 348 演唱行 = 86.5%** 输出 `mode=word`。
- 其余 47 行为安全回退（`uncertain` 质量门控、语言路由含糊、或模型失败），保留原始 LRC 时间与文字。
- ja 歌曲在 `--mode experimental` 下可额外输出 38 行诊断时间轴（97.4%），仍**不覆盖 FAILED**。

## 4. 混合语言路由是否有效？

**有效且可审计。** 中文歌的英文段落路由到英文模型（zh 歌失败 15→5、9→4）；
拉丁字母**不无条件**判为英语（`(Bora sana, …)` → 显式回退，不伪造时间）；
模型**惰性加载、整曲复用**；每行在 JSON 记录 `line_language`/`route_model`/`route_reason`。

## 5. 还存在哪些严重错位或误放行？

- **质量门控仍不可靠**（V0.7/V0.8 实测：折内选择 F1 0.58/0.59；`さんぽ` 被误判 reliable）。
  本轮只把它用作**保守回退**，不作为精度保证。
- **日语 production 回退较多**：`SAN値直葬` 39/73、`歌姫失格` 18/73 行为 `line_lrc`。
- **中文歌的拉丁含糊行无法对齐**（4+5 行），只能整行回退（不伪造时间）。
- **真实歌曲精度尚未人工评测**：已生成标注模板与评分器，但**没有人工参考边界就没有精度数字**。
- 未做基于音频的“这行没唱”判定；无 cue 文字的器乐行仍会进入对齐。

## 6. 是否具备进入 Android ONNX 原型开发的条件？

**有条件具备——建议以 zh + en 先行。**

可以稳定依赖的部分：

- **数据协议稳定**（JSON schema `0.3`，`docs/LRC_FALLBACK_CONTRACT.md`）：原始 LRC 时间、
  模型时间、派生显示时间、行类型、质量三状态、回退原因、逐字单元 —— 均有契约与测试。
- **zh（fa-zh）/ en（wav2vec2-base-960h）为 stable 后端**，Apache-2.0/自定义许可（中文需商用复核）；
  6 首歌 0 失败（zh 的失败均来自无法建模的拉丁行，已显式回退）。
- 时间轴不变量（单调、不跨行、不越界、原文可还原）在真实整曲上验证。

进入原型前必须完成的工作：

1. **ONNX 导出 + int8 量化**，并用**同一批真实歌曲与人工标注**复测精度（量化可能回退）。
2. **ja 仍为 experimental**：HuBERT 候选可移植（Apache-2.0），但需先解决回退率与置信度问题。
3. 需要一个**平台无关的 Kotlin/Java 侧 Viterbi 与 LRC 回退实现**（协议本已平台无关）。

## 7. 哪些语言仍只能作为实验功能？

| 语言 | 状态 | 说明 |
|---|---|---|
| **zh** | **stable** | fa-zh；可用；拉丁行需回退（已显式处理） |
| **en** | **stable** | wav2vec2-base-960h；6 首歌 0 失败 |
| **ja** | **experimental** | HuBERT 音素 CTC（Apache-2.0）；production 回退 39/18 行；质量门控不可靠 |
| ja（pyshiro） | **research only** | 权重无明确许可，**不得分发** |

## 8. 本轮交付

- 代码：`benchmarks/full_song_eval.py`、`tools/make_annotation_template.py`、
  `lyrics/language.py`、`models/router.py`、窗口上限/启发式偏移/重建强制回退修复。
- 测试：`tests/test_window_and_fallback.py`（6）、`tests/test_language_routing.py`（13）、
  `tests/test_full_song_eval.py`（6），总计 **178 项**通过。
- 文档：`docs/FULL_SONG_EVALUATION.md`、`docs/MIXED_LANGUAGE_ROUTING.md`、本文件；
  更新 `LRC_FALLBACK_CONTRACT.md`（schema 0.3）、`CLI_END_TO_END.md`、`KNOWN_LIMITATIONS.md`。

## 9. 未完成 / 下一步

1. **人工标注**：填写 `experiments/annotation/annotation_template.csv`（每首 30 单元）→
   `tools/make_annotation_template.py score` 得到六首真实 MAE/P90/错误率与语言路由错误率。
2. **Android ONNX 原型**：zh/en 导出与量化，量化后重跑本六首评测。
3. **暂停感知/置信度**：改善 ja 回退率（需歌声域模型或域适配，见 `NEXT_STEPS.md`）。
