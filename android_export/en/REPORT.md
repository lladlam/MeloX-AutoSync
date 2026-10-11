# AutoSync Android ONNX 导出报告

> 语言：英文（`facebook/wav2vec2-base-960h`）
> 生成方式：`/home/lladlam/项目/MeloX-AutoSync/.venv/bin/python android_export/en/validate_onnx.py`（本文件由脚本自动生成，所有数字来自本机实测；生成时间 2026-10-11 13:00:18）

## 1. 结论（是否可交付 / 未达标项）

**存在未达标项：FP32 max|Δlogits|、FP32 mean|Δlogits|、INT8 cosine min、Alignment start MAE (INT8 vs FP32)（下表 FAIL 行，已如实报告，未粉饰）。**

| 验收项 | 结果 | 实测 | 阈值 |
|---|---|---|---|
| FP32 max|Δlogits| | **FAIL** | 5.018e-03 | <= 5e-04 |
| FP32 mean|Δlogits| | **FAIL** | 1.677e-04 | <= 5e-06 |
| FP32 argmax agreement (confident frames) | PASS | 100.0000% (2573 frames) | == 100% |
| Baked normalization vs AutoProcessor | PASS | graph max|Δ|=9.537e-07 (formula 9.537e-07) | <= 1e-05 |
| Tokenizer encoder == HF tokenizer | PASS | 6/6 probe strings | all equal |
| INT8 cosine mean | PASS | 0.999108 | >= 0.995 |
| INT8 cosine min | **FAIL** | 0.963666 | >= 0.97 |
| INT8 determinism 3x bit-equal | PASS | True | True |
| Alignment start MAE (INT8 vs FP32) | **FAIL** | 121.19 ms over 168 words | <= 30.0 ms |
| Alignment coverage change | PASS | 100.00% vs 100.00% (Δ 0.00 pp) | <= 2.0 pp |
| Alignment feasibility flips | PASS | 0 of 29 lines | <= 1 |
| Alignment sample size | PASS | 29 lines / 4 songs | >= 20 lines, >= 3 songs |
| fp32 loadable (checker/CPU EP/names/shapes) | PASS | see section 3 | required |
| fp32 standard ops / opset<=17 / single file | PASS | see section 3 | required |
| int8 loadable (checker/CPU EP/names/shapes) | PASS | see section 3 | required |
| int8 standard ops / opset<=17 / single file | PASS | see section 3 | required |
| Test vectors regenerated | PASS | 3 expected.json files | required |

## 2. 交付文件（路径 / 字节 / sha256）

| 文件 | 字节 | sha256 |
|---|---|---|
| `/home/lladlam/项目/MeloX-AutoSync/android_export/en/wav2vec2-base-960h-fp32.onnx` | 377815805 | `d4ac91ec0093a3f2f63fa27dd39ea5db2847634b1c3d893e0b34930d41d4d34d` |
| `/home/lladlam/项目/MeloX-AutoSync/android_export/en/wav2vec2-base-960h-int8.onnx` | 122373796 | `1c8213118da7f555e4b68c0e08e85a4c471ea2a3e0fdf5620d27412800babd3f` |
| `/home/lladlam/项目/MeloX-AutoSync/android_export/en/manifest.json` | 1438 | `053a1a83d38d5d089a88fc43092b8e9683b8f9f4bb638113584f8a278a2ab6a7` |
| `/home/lladlam/项目/MeloX-AutoSync/android_export/en/validate_onnx.py` | 60485 | `64938278ed98704a69325cc04a1e2b18dbb11b5db587cfb1ae67b25bc06f4d8c` |
| `/home/lladlam/项目/MeloX-AutoSync/android_export/en/export_model.py` | 15262 | `66ad30fe35cb9a21e98a48fb726e349f7a6e81765c6925e328d4c413521d23d3` |
| `/home/lladlam/项目/MeloX-AutoSync/android_export/en/test_vectors/vector1..3.wav + vectorN.expected.json` | (小文件，见目录) | — |

## 3. 契约核对（I/O 名、形状、opset、算子、external data、归一化比对）

### fp32

- `onnx.checker.check_model`: 通过
- ORT `CPUExecutionProvider` 加载: 通过
- 输入: input_values shape=[1, None] type=tensor(float)
- 输出: logits shape=[1, None, 32] type=tensor(float); log_probs shape=[1, None, 32] type=tensor(float)
- 与 `manifest.json` 名称/形状一致: 是；sha256/字节一致: 是
- opset_import: [['', 17]]（≤17: 是）；算子 domain: 全部为标准 ONNX domain；含 `com.microsoft` 算子: 否
- external data: 无（单文件自包含）；IR version 8
- 图中算子: Add, Cast, Concat, Constant, Conv, Div, Erf, Gather, InstanceNormalization, LayerNormalization, LogSoftmax, MatMul, Mul, ReduceMean, Reshape, Shape, Slice, Softmax, Sqrt, Sub, Transpose, Unsqueeze

### int8

- `onnx.checker.check_model`: 通过
- ORT `CPUExecutionProvider` 加载: 通过
- 输入: input_values shape=[1, None] type=tensor(float)
- 输出: logits shape=[1, None, 32] type=tensor(float); log_probs shape=[1, None, 32] type=tensor(float)
- 与 `manifest.json` 名称/形状一致: 是；sha256/字节一致: 是
- opset_import: [['', 17]]（≤17: 是）；算子 domain: 全部为标准 ONNX domain；含 `com.microsoft` 算子: 否
- external data: 无（单文件自包含）；IR version 8
- 图中算子: Add, Cast, Concat, Constant, Conv, Div, DynamicQuantizeLinear, Erf, Gather, InstanceNormalization, LayerNormalization, LogSoftmax, MatMul, MatMulInteger, Mul, ReduceMean, Reshape, Shape, Slice, Softmax, Sqrt, Sub, Transpose, Unsqueeze

### 归一化（烘焙 vs AutoProcessor）

- 同一批 5 段音频逐段比较归一化张量：
  - 烘焙公式（torch，导出源）vs `AutoProcessor` 输出: max|Δ| = 9.537e-07
  - 单独导出的归一化子图（ONNX + ORT）vs `AutoProcessor` 输出: max|Δ| = 9.537e-07（阈值 1e-5）
- 公式: `(x - mean(x)) / sqrt(var(x) + 1e-7)`，population variance，整段求均值/方差，与 `Wav2Vec2FeatureExtractor(do_normalize=True)` 一致。
- 词表编码器与 HF tokenizer 一致性: 6/6 探针串相等；注意：词表中没有数字，`0-9` 与未知字符均映射为 `<unk>`(3)（与 HF tokenizer 行为一致，Android 端需同样处理）。

## 4. 验证数字（按第 5 节 1–5 逐项列表，含素材来源与样本数）

### 5.1 导出正确性（FP32 ONNX vs PyTorch）

素材：5 段、合计 51.78 s；逐段独立前向。

| 片段 | 来源 | 类型 | 秒 | T | max&#124;Δlogits&#124; | mean&#124;Δlogits&#124; | 置信帧 | argmax 一致率 |
|---|---|---|---|---|---|---|---|---|
| S1-librispeech-sample1 | `experiments/audio/librispeech_sample1.flac` | read speech (LibriSpeech dev-clean, public domain) | 13.69 | 684 | 2.748e-03 | 1.875e-04 | 684 | 100.0000% |
| S2-librispeech-sample2 | `experiments/audio/librispeech_sample2.flac` | read speech (LibriSpeech dev-clean, public domain) | 14.215 | 710 | 5.018e-03 | 1.833e-04 | 709 | 100.0000% |
| S3-synthetic-en | `experiments/audio/synthetic_en.wav` | synthetic speech (espeak-ng, repo test material) | 6.575 | 328 | 2.418e-03 | 2.514e-04 | 323 | 100.0000% |
| S4-gtsinger-edelweiss | `experiments/singing/gtsinger/English/EN-Alto-1/Vibrato/edelweiss/Vibrato_Group/0000.wav` | real singing (GTSinger EN-Alto-1, CC BY-NC-SA, local only) | 8.758 | 437 | 1.077e-03 | 8.592e-05 | 437 | 100.0000% |
| S5-gtsinger-mylove | `experiments/singing/gtsinger/English/EN-Alto-1/Glissando/my love/Glissando_Group/0000.wav` | real singing (GTSinger EN-Alto-1, CC BY-NC-SA, local only) | 8.542 | 426 | 1.144e-03 | 1.291e-04 | 420 | 100.0000% |
| **合计** | 5 段 | | 51.78 | 2585 | **5.018e-03** | **1.677e-04** | 2573 | **100.0000%** |

- 阈值：max ≤ 5e-04，mean ≤ 5e-06，置信帧 argmax 一致率 = 100% —— **未通过**。
- 烘焙归一化 vs `AutoProcessor` max|Δ| = 9.537e-07（阈值 1e-5）—— 通过。

### 5.2 INT8 精度（INT8 vs FP32）

**逐帧 `log_probs` 余弦相似度**

| 片段 | 帧数 | cosine mean | cosine min |
|---|---|---|---|
| S1-librispeech-sample1 | 684 | 0.999164 | 0.989034 |
| S2-librispeech-sample2 | 710 | 0.998911 | 0.982021 |
| S3-synthetic-en | 328 | 0.999411 | 0.973179 |
| S4-gtsinger-edelweiss | 437 | 0.999259 | 0.984563 |
| S5-gtsinger-mylove | 426 | 0.998957 | 0.963666 |
| **合计** | 2585 | **0.999108** | **0.963666** |

- 阈值 mean ≥ 0.995 / min ≥ 0.97 —— **未通过**；同批输入 int8-vs-fp32 argmax 一致率（fp32 置信帧）= 95.919%。

**端到端强制对齐（`src/autosync/alignment/ctc.py::forced_align`, blank=0）**

素材：本地 `Test/` 第三方歌曲 4 首（仅作评测输入，避免第三方歌词文本，以别名 S1..S4 + 行号引用），共 29 行、168 个词单元；INT8/FP32 各自对同一裁剪窗推理并强制对齐，逐词比较。

| 歌曲别名 | 行数 | 词数 | FP32 feasible 行 | INT8 feasible 行 | 翻转 | 词起始 MAE (ms) |
|---|---|---|---|---|---|---|
| S1 | 8 | 54 | 8 | 8 | 0 | 35.19 |
| S3 | 8 | 50 | 8 | 8 | 0 | 207.20 |
| S4 | 5 | 15 | 5 | 5 | 0 | 265.33 |
| S6 | 8 | 49 | 8 | 8 | 0 | 84.08 |
| **合计** | **29** | **168** | 29 | 29 | **0** | **121.19** |

- 起始时间 MAE（INT8 vs FP32，逐词）= **121.19 ms**（阈值 ≤ 30 ms）—— **未通过**；结束时间 MAE（附加）= 129.88 ms。
- 覆盖率（有起止时间的词单元占比）：FP32 100.00% vs INT8 100.00%，变化 **0.00 pp**（阈值 ≤ 2 pp）—— 通过。
- feasible 判定翻转 **0** 行（阈值 ≤ 1）—— 通过。

> 说明：该项测量的是 INT8 与 FP32 **输出的一致性**（不是绝对准确率）；对齐输入为真实歌曲音频 + LRC 行起点与文本。歌词原文不写入交付物。

### 5.3 健壮性

- 静音（全 0，2 s）输入：FP32 无 NaN/Inf；INT8 无 NaN/Inf。
- 同一输入连跑 3 次：INT8 输出逐位一致 = 是（logits 与 log_probs 的原始字节完全相等）。
- 极短输入 400 采样：FP32/INT8 均正常返回（T=1）；>=30 s 长输入：FP32 用 51.8 s 连续音频，INT8 用 51.8 s，均无报错、输出有限。

### 5.4 可加载性

见第 3 节；`onnx.checker`、CPUExecutionProvider、I/O 名称/形状、单文件、opset ≤ 17、无 `com.microsoft` 算子均逐项检查。

### 5.5 性能与内存（桌面参考值，单线程 CPU）

| 模型 | 12 s 窗口中位数 (ms) | P90 (ms) | 加载 (s) | 10 窗推理峰值 RSS (MiB) |
|---|---|---|---|---|
| int8 (`wav2vec2-base-960h-int8.onnx`) | 1050.4 | 1062.0 | 0.16 | 551.5 |
| fp32 (`wav2vec2-base-960h-fp32.onnx`) | 1597.8 | 1603.7 | 0.35 | 929.7 |

- 计时：`intra_op_num_threads=1`、10 次推理的中位数/P90（2 次预热后）；窗口为 12 s LibriSpeech 片段。

| 文件 | 字节数 | 说明 |
|---|---|---|
| `wav2vec2-base-960h-fp32.onnx` | 377815805 | 对比用 |
| `wav2vec2-base-960h-int8.onnx` | 122373796 | Android 端实际使用（目标 ≤ 120 MB） |

### 设备端自检向量

| 文件 | 时长 | 源 | INT8 T | blank_ratio | INT8 feasible | FP32 feasible |
|---|---|---|---|---|---|---|
| `test_vectors/vector1.wav` | 4.38s | `experiments/audio/librispeech_sample1.flac` | 218 | 0.5367 | True | True |
| `test_vectors/vector2.wav` | 3.5s | `experiments/audio/librispeech_sample1.flac` | 174 | 0.3908 | True | True |
| `test_vectors/vector3.wav` | 4.26s | `experiments/audio/librispeech_sample2.flac` | 212 | 0.5802 | True | True |

## 5. 一键复跑命令

```bash
cd /home/lladlam/项目/MeloX-AutoSync
.venv/bin/python android_export/en/validate_onnx.py        # 重新跑全部检查并重写 REPORT.md
# 如需重新导出模型（约需数分钟、多进程分批执行）：
.venv/bin/python android_export/en/export_model.py --force
```

## 6. 已知限制与风险

- **量化在歌声上的表现**：INT8 余弦/对齐一致性在真实歌曲（含伴奏、嘶吼、混响）上成立，但它衡量的是与 FP32 的**一致性**；两者共同的绝对误差在歌声/强伴奏上仍可能较大（引擎侧已有质量分级兜底）。本次评测素材中 2 段 GTSinger 为干声演唱，其余段落为朗读/合成语音与商业歌曲混音。
- **动态量化的确定性**：ORT 动态量化推理逐位可复现（已 3× 验证）；但不同 ORT 版本/不同设备的MatMulInteger 内核可能带来极小数值差异（设备端以 test_vectors 对拍为准）。
- **内存**：本机可用内存仅约 1.5 GB（另有并发任务），校验脚本按阶段拆分子进程；桌面 fp32 会话本身约需数百 MB，Android 端 INT8 会话显著更小（见 5.5）。
- **桌面 ≠ 设备**：5.5 是 x86 单线程参考值；Android ARM 上动态量化 MatMulInteger 通常更快，但内存带宽与调度不同，应以真机 benchmark 为准。
- **归一化是整段统计**：与 HF `Wav2Vec2FeatureExtractor` 完全一致；这意味着**裁剪窗口会改变归一化**，设备端对同一段音频须整段喂入（引擎端也是逐行裁剪后整体归一化）。

## 7. 未做/做不了的部分

- 未在 Android 真机/ARM 上运行（本机为 x86 CPU，单机验证）。
- 未量化 Conv 前端（按任务书要求保持 fp32）；未尝试 QDQ/静态量化、未做 QAT——任务书未要求。
- 未做绝对精度（人工标注）评测：INT8-vs-FP32 一致性已按要求测评；真实歌曲的人工词级标注本地不可得（GTSinger 有标注，但为干声且非商业混音）。
- 中文/日文链路不在本轮范围。

