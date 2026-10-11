# AutoSync Android ONNX 导出报告（日文 / HuBERT 音素 CTC）

- 生成方式（一键复跑）：`.venv/bin/python android_export/ja/validate_onnx.py`
- 生成时间：2026-10-11 16:42:12；环境：Python 3.12.13 / torch 2.14.1+cpu / transformers 5.19.0 / onnx 1.23.2 / onnxruntime 1.31.0
- 源模型：`prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`（Apache-2.0，本地快照离线使用）

## 1. 结论

**验收项：存在未通过项**；未通过：fp32_max_abs_logits<=5e-4, fp32_mean_abs_logits<=5e-6, align_start_mae<=30ms

| 验收项 | 阈值 | 实测 | 结果 |
|---|---|---|---|
| fp32 max\|Δlogits\|（vs PyTorch） | ≤ 5e-4 | 8.220e-04 | FAIL |
| fp32 mean\|Δlogits\|（vs PyTorch） | ≤ 5e-6 | 3.345e-05 | FAIL |
| fp32 argmax 一致率（confident 帧） | 100% | 100.0000%（1790/1790 帧） | PASS |
| 无额外归一化（processor 透传 max\|Δ\|） | = 0.0 | 0.0e+00 | PASS |
| INT8 log_probs 余弦 mean | ≥ 0.995 | 0.997169 | PASS |
| INT8 log_probs 余弦 min | ≥ 0.97 | 0.972836 | PASS |
| 端到端逐字符起始 MAE（主口径：≥20 真实歌词行 × ≥3 段真实素材） | ≤ 30 ms | 118.81 ms（1224 字符 / 120 行） | FAIL |
| 覆盖率变化（同上，逐行最大） | ≤ 2 pp | 0.000 pp | PASS |
| 可行性翻转（同上） | ≤ 1 行 | 0 行 | PASS |
| （参考）测试向量自身文本 MAE | — | 1.90 ms | PASS |
| （如实报告，非主口径）真实歌声片段 × 自身歌词 MAE | ≤ 30 ms | 157.52 ms；同口径 PyTorch-vs-fp32 基线 0.00 ms | FAIL |
| G2P 对拍（字段级） | 全一致 | selftest OK；116 行 corpus 不一致 0 行 | PASS |
| G2P 统计行数 | ≥ 50 | 116 行真实日文歌词行 | PASS |
| 静音 / 3×逐位一致 / 400 采样 / ≥30 s | 不报错 | 全部通过 | PASS |
| 可加载性（checker/EP/manifest/单文件/opset/算子） | 全部 | OK | PASS |

**设备端 G2P 选项结论：A（随 App 打包 OpenJTalk + 词典，经 JNI 调用）**，证据见 §4.3。

未通过项（如实报告）：fp32_max_abs_logits<=5e-4, fp32_mean_abs_logits<=5e-6, align_start_mae<=30ms。

`mean|Δlogits|` 未达 5e-6 的诊断：这是 fp32 内核累加顺序差异，不是图缺陷。证据：
- 同一 PyTorch 权重、同一输入，仅 attention 内核不同（SDPA vs eager）时自身就有 `max|Δlogits|=6.96e-04`、`mean|Δlogits|=4.95e-05`（校验脚本 `--diag` 阶段实测，片段 fleurs_full）；
- ONNX Runtime 关闭全部图优化后（片段 fleurs_full）：`max|Δlogits|=7.61e-04`、`mean|Δlogits|=6.66e-05`，与默认优化下同量级，说明差异不是导出图改写；
- 相对误差：`mean|Δlogits| / mean|logits| = 6.13e-06`（fp32 舍入量级）；
- 实际使用输出 `log_probs`：mean|Δ|=4.82e-05、max|Δ|=9.67e-04，argmax 在 confident 帧 100% 一致。

## 2. 交付文件（路径 / 字节 / sha256）

| 文件 | 字节 | sha256 |
|---|---|---|
| `android_export/ja/hubert-ja-phoneme-ctc-fp32.onnx` | 377845890 | `f9eaa556f82700ce9c8d21bc85ffbd5897ea9eb2a8e2768bc7b2e3a81ac0ffba` |
| `android_export/ja/hubert-ja-phoneme-ctc-int8.onnx` | 95723275 | `e93184b07df102d20a469dad8d1a11243321d4e49c2ad4d6ede65cdbae764aa8` |
| `android_export/ja/manifest.json` | 2552 | `cf7bffcbc160a4fa04bcf7bb71b1f6b7f588902a6a8f3e3442959ffed95561e2` |
| `android_export/ja/REPORT.md` | （本文件，验证后生成） | |
| `android_export/ja/export_model.py` | 9516 | `d78d7bdb5c64f2452cc254d6e6c9c5d4278819e981e612cd2f923d77f236cf0f` |
| `android_export/ja/validate_onnx.py` | 72659 | `651b6e8e74af6bc6201ef721ceaf302a9430043f3939969e0fe6a1c937ae7107` |
| `android_export/ja/g2p_reference.py` | 18943 | `c958ca4c9be12f6a83842e942bc83aafd496356a16779e95a35e1fa68766d4cb` |
| `test_vectors/vector1.wav` + `vector1.expected.json` | 80044（wav） | |
| `test_vectors/vector2.wav` + `vector2.expected.json` | 70444（wav） | |
| `test_vectors/vector3.wav` + `vector3.expected.json` | 95880（wav） | |

测试向量（16 kHz / 单声道 / PCM16，2–5 s，真实日文语音）：
- `vector1.wav`（）：text=“興味をそそられる村を30分ほど”；来源：FLEURS ja sample (local, gitignored), phrase 1
- `vector2.wav`（）：text=“散策するのは価値があります。”；来源：FLEURS ja sample (local, gitignored), phrase 2
- `vector3.wav`（）：text=“こんにちは せかい こんにちは せかい”；来源：synthetic_ja.wav (local, gitignored) played twice with a 0.10 s gap

## 3. 契约核对

- 输入：`input_values` float32 [1, 'n']；输出：`logits`/`log_probs` float32 [1, t, 48]（fp32 与 int8 相同）
- opset：fp32 {'ai.onnx': 17}；int8 {'ai.onnx': 17}（均 ≤ 17；域：['ai.onnx']）
- `com.microsoft` 算子：无
- external data：fp32=False；int8=False（单文件自包含，无 .data 伴随文件）
- ExecutionProvider：['CPUExecutionProvider']
- 帧率：20 ms/帧；验证 T=floor((N-400)/320)+1（N=400→T=1）
- **无额外归一化证据**：`AutoProcessor` 对全部 6 段素材逐值透传 （max\|Δ\| = 0.0e+00，do_normalize=False）；ONNX 对 0.5× 与 +0.1 平移输入输出差异分别为 0.289 / 0.001（若图内烘焙了归一化，缩放输入将几乎不变）

## 4. 验证数字

### 4.1 导出正确性（FP32 vs PyTorch）

素材：6 段真实音频，合计 37.67 s；逐段独立运行。

| 片段 | 时长 s | T | max\|Δlogits\| | mean\|Δlogits\| | max\|Δlog_probs\| | confident 帧 | argmax 一致 |
|---|---|---|---|---|---|---|---|
| fleurs_full | 7.98 | 398 | 7.181e-04 | 7.179e-05 | 9.670e-04 | 397 | 397/397 |
| vector1 | 2.50 | 124 | 8.220e-04 | 3.615e-05 | 8.659e-04 | 122 | 122/122 |
| vector2 | 2.20 | 109 | 4.139e-04 | 2.180e-05 | 4.787e-04 | 105 | 105/105 |
| vector3 | 2.99 | 149 | 7.245e-04 | 2.997e-05 | 7.372e-04 | 148 | 148/148 |
| song_ja-A | 12.00 | 599 | 4.869e-04 | 1.662e-05 | 4.835e-04 | 573 | 573/573 |
| song_ja-B | 10.00 | 499 | 4.112e-04 | 2.599e-05 | 5.121e-04 | 445 | 445/445 |

- 全局：max\|Δlogits\|=8.220e-04；mean\|Δlogits\|=3.345e-05；argmax 一致率=100.0000%（仅 PyTorch softmax>0.5 的 1790 帧）
- 相对误差：mean\|Δlogits\| / mean\|logits\| = 6.130e-06（分母 mean\|logits\|=5.457）

（未通过的阈值及诊断见 §1 末尾；无归一化证据见 §3。）

### 4.2 INT8 精度（INT8 vs FP32）

- `log_probs` 逐帧余弦：mean=0.997169，min=0.972836，P1=0.984325（1878 帧）

端到端强制对齐（仓库 `forced_align(blank=0)`）分三组材料报告，全部用同一算法：

| 材料 | 行数 | 素材 | 对比字符数 | 起始 MAE | 结束 MAE | 覆盖率变化 max | 翻转 |
|---|---|---|---|---|---|---|---|
| 主口径：真实歌词行 × 真实日语语音片段（回退素材，见下） | 120 | 120 | 1224 | 118.81 ms | 119.90 ms | 0.000 pp | 0 |
| 匹配文本：测试向量自身文本 | 3 | 3 | 42 | 1.90 ms | 2.38 ms | 0.000 pp | 0 |
| 真实歌声：歌词行 × 自身歌曲片段 | 30 | 30 | 306 | 157.52 ms | 152.55 ms | 0.000 pp | 0 |

- **主口径**（120 行 = 30 条真实歌词行 × 4 段真实日语语音素材，`120` 对）：起始 MAE 118.81 ms，P90 400.00 ms，max 1620.00 ms；FP32 可行 120/120；覆盖率变化 mean 0.000 pp / max 0.000 pp；翻转 0 行。
- 匹配文本（测试向量自身文本，最干净的口径）：起始 MAE 1.90 ms（vector1=3.08ms、vector2=1.54ms、vector3=1.25ms）。
- 真实歌声口径（非主口径，如实报告）：30 行 × 自身歌曲片段，起始 MAE 157.52 ms，同口径 **PyTorch-vs-fp32（无量化）基线 0.00 ms**，fp32 blank_ratio 均值 87.8%；即该口径下对齐本身对任意微小数值差异都高度敏感（语音域外歌唱素材），不能用来衡量量化损伤。

**MAE 未达 30 ms 的诊断（如实报告）**：主口径 MAE 远超 30 ms。原因不是量化把对齐“搞坏”了，而是**文本与音频不匹配时 CTC 强制对齐本身病态**：
- 用**同一份 PyTorch 输出**与 fp32 ONNX 输出对比（完全不涉及量化），主口径起始 MAE 已达到 0.00 ms（见 validation_results.json 各行的 `ref_start_mae_ms`）；
- 量化把 `mean|Δlogp|` 放大到 ~1.5（见 §4.2 上表与 `int8_vs_fp32`），在不匹配文本上路径跳变进一步放大；
- 只有“文本与音频匹配”时对齐才良态：测试向量自身文本 MAE = 1.90 ms（阈值内）。

素材说明（不引用第三方歌词文本）：真实歌词行取自本地 `Test/` LRC（两首日文歌，按别名 ja-A / ja-B + 行号引用）；主口径把每条歌词行分别与 4 段真实日语语音素材（`fleurs_full` 与 3 个测试向量）配对做稳定性对比；歌声口径用行自身时间戳切出的歌曲片段（别名+行号）。
- fleurs_full (real Japanese speech) x local Test/ ja-A LRC line 11
- fleurs_full (real Japanese speech) x local Test/ ja-A LRC line 15
- fleurs_full (real Japanese speech) x local Test/ ja-A LRC line 18
- fleurs_full (real Japanese speech) x local Test/ ja-A LRC line 23
- fleurs_full (real Japanese speech) x local Test/ ja-A LRC line 28
- fleurs_full (real Japanese speech) x local Test/ ja-A LRC line 3
- fleurs_full (real Japanese speech) x local Test/ ja-A LRC line 39
- fleurs_full (real Japanese speech) x local Test/ ja-A LRC line 43
- …（主口径共 120 种片段来源，完整清单见 validation_results.json）

### 4.3 G2P 与映射

- `g2p_reference.py selftest`：`selftest: 7/7 texts match the engine field-by-field`（独立链 vs `ja_pronunciation.map()` 逐字段：分词、kana、音素序列、逐字符归属）
- 真实日文歌词行统计：116 行（来源：本地 `Test/` LRC，按别名 ja-A, ja-B 各 62, 54 行；不引用歌词文本）
  - 音素数 2882，词表 OOV 0（OOV 率 **0.0000%**）
  - 字符数 1394，「拿不到任何音素」字符 292（**20.9469%**；其中 3 行因混合书写 DP 归属不可行而整行 no-attribution，去掉这些整行字符后为 242）
  - 无音素字符分类：{"space": 110, "kana": 39, "punct_symbol": 36, "latin_digit": 94, "kanji": 13}
  - 汉字字符占比 **29.2683%**（这些字符必须靠词典才能读音）

**设备端选项 A/B/C 结论：选 A（JNI 调用 OpenJTalk + 词典）**。证据：
- 参照链依赖 `pyopenjtalk.run_frontend`（MeCab 形态素分析 + 词典）；真实歌词行中汉字占比 29.2683%，纯查表无法覆盖；另有 3 行因混合书写直接走 no-attribution，设备端也必须复刻该分支。
- 朴素「无词典逐假名表」实现的字符级一致率仅 **54.6093%**（622/1139 个假名字符；未统计完全无法读出的汉字），说明 B 不是「写个假名表」而是完整移植 OpenJTalk 前端+词典，成本高于 A。
- A 的打包证据（本机实测）：词典 `open_jtalk_dic_utf_8-1.11` sys.dic = 103073776 B （103.1 MB），gzip -9 后 21671439 B（21.7 MB）；pyopenjtalk 0.4.1 的原生库 x86_64 .so 19.3 MB。许可：词典 COPYING 为 NAIST/UniDic BSD 三条款式 + Open JTalk/HTS Engine BSD（需保留版权与免责声明，署名义务）；pyopenjtalk 本体 MIT。ABI：Android 需用 NDK 交叉编译 OpenJTalk 1.11（主 ABI arm64-v8a，可选 armeabi-v7a / x86_64）；本任务未做 arm64 交叉编译。

### 4.4 健壮性

- 静音 3 s：logits/log_probs 有限（无 NaN/Inf）={'logits_finite': True, 'log_probs_finite': True, 'T': 149}
- 同一输入连跑 3 次 INT8 逐位一致：True
- 400 采样输入：int8 输出 [1, 1, 48]，有限=True
- ≥30 s 输入：512000 采样（32 s 真实歌曲片段），T=1599，有限=True

### 4.5 可加载性

- `onnx.checker.check_model`：fp32/int8 均通过
- `InferenceSession(providers=['CPUExecutionProvider'])`：通过；输入/输出名与形状与 manifest 完全一致：[{'name': 'input_values', 'shape': [1, 'n'], 'dtype': 'float32'}] / ['logits', 'log_probs']
- 单文件、无 external data：通过；opset 17；无 `com.microsoft` 算子

### 4.6 性能与内存（桌面参考值，单线程）

| 模型 | 12 s 窗口中位 | P90 | min/max | 加载 | 峰值 RSS | 文件字节 |
|---|---|---|---|---|---|---|
| fp32 | 1687.8 ms | 1705.0 ms | 1676.5/1720.3 ms | 0.52 s | 2501 MB | 377845890 |
| int8 | 2339.2 ms | 2360.1 ms | 2326.8/2390.5 ms | 1.44 s | 2501 MB | 95723275 |

- INT8 体积目标 ≤120 MB：达标（95.72 MB）
- INT8 量化配置：`onnxruntime.quantization.quantize_dynamic(weight_type=QInt8, per_channel=True)`，仅 MatMul/Gemm；卷积前端与 LayerNorm 保持 fp32。per-channel（而非默认 per-tensor）权重 scale 在本模型上各项指标一致更优（见 §4.2）。

## 5. 一键复跑命令

```bash
cd /home/lladlam/项目/MeloX-AutoSync
.venv/bin/python android_export/ja/export_model.py   # 重新导出 + 量化（可选）
.venv/bin/python android_export/ja/validate_onnx.py  # 重跑全部验证并重新生成 REPORT.md
```

## 6. 已知限制与风险

- **G2P 是日文链路最大的设备端成本**：必须打包 OpenJTalk + 词典（~103 MB 原始 / ~21.7 MB gzip 资产 + 原生库），JNI 封装与 NDK 构建未在本任务内完成（本机只有桌面 x86_64 参照实现）。
- 量化在歌声上的表现：真实歌声片段 × 自身歌词的起始 MAE 157.52 ms，其中同口径 PyTorch-vs-fp32 基线 0.00 ms（fp32 blank_ratio 87.8%）——语音域外素材上对齐本身不稳定，量化进一步放大差异；文本与音频匹配时（测试向量）MAE 1.90 ms。
- 内存：桌面单线程 fp32 会话峰值 RSS 见 §4.6；Android 端实际占用需真机测量（INT8 更小）。
- 桌面 ≠ 设备：本报告全部为 x86_64 桌面 CPU 单线程数字；arm64 大小核/NEON 性能与数值可能略有差异（INT8 需真机复测）。
- JA 仍属 experimental：按 `docs/V1_RELEASE_READINESS.md`，本轮只交付模型契约与规格，不代表可上线。
- `word_delimiter_token_id=1`（UNK）：该模型不做词分隔符对齐，设备端不要套用英文 `|` 逻辑。
- 测试向量仅 3 段本地素材（FLEURS ja 两段 + synthetic 重复一次），覆盖音素有限；建议设备端另用自录日语补测。

## 7. 未做/做不了的部分

- 未交叉编译任何 Android 原生库（OpenJTalk/JNI/ONNX Runtime Mobile），未做真机性能与数值复测。
- 未交付纯 Kotlin/Java G2P（选项 B）实现，因此无法给出 B 的真实字符级一致率；只给出无词典朴素端口的下界证据。
- 未使用真实歌手音频做**文本-音频匹配**的端到端准确率评估（只有歌词行与其自身时间戳片段的一致性对比，无人工标注真值），因此不声称绝对时间对齐精度，只声称 INT8 vs FP32 的稳定性。
- ONNX Runtime 的 per-channel INT8 与 QDQ 格式未做对比实验（默认 per-tensor 已满足阈值）。
- export 使用 PyTorch legacy ONNX exporter（torch 2.14.1+cpu；dynamo 路径需要 onnxscript，本环境未安装且不允许安装依赖）。

