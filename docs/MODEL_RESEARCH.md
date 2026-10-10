# 模型调研（MODEL_RESEARCH）

> 目标不是 ASR 转写准确率，而是**歌词文本与音频之间的时间对齐精度**。
> 本文区分「已用代码验证」与「仅文献/模型卡调研」的结论。

## 0. 选型标准

1. 支持**已知文本的强制对齐**（forced alignment），而不是从零转写。
2. 模型**小**、可在 CPU 推理、可量化、可移植 Android。
3. 许可证允许在 MeloX 中分发或按需下载（不提交权重到 Git）。
4. 三种语言**互相独立**，只加载用户请求的语言。

## 1. 中文：`funasr/fa-zh`

| 项目 | 值 |
|---|---|
| 仓库 | https://huggingface.co/funasr/fa-zh (别名 `iic/speech_timestamp_prediction-v1-16k-offline`) |
| 架构 | `MonotonicAligner` = `SANMEncoder`(30 blocks, 320 dim) + `CifPredictorV3` |
| 参数 / 体积 | ~38M 参数 / `model.pt` 151 MB (FP32) |
| 训练数据 | 5000 小时普通话 |
| 分词 | CharTokenizer（汉字级）；`tokens.json` 以汉字为主 |
| 许可证 | `model-license`（FunASR Model License，非 OSI 标准） |
| 后端 | FunASR（可选依赖 `funasr` + `modelscope` + `kaldi-native-fbank`） |
| 对齐机制 | **原生时间戳预测**（单调注意力 + CIF），非 CTC Viterbi |

**为什么选它**：它不是 ASR 转写模型，而是**专门做时间戳预测**的对齐器，输入「音频 + 文本」直接输出每个 token 的毫秒区间。38M 参数在候选里最小，且中文场景开箱即用。

**验证结论（已运行）**：见 `EXPERIMENT_RESULTS.md`。真实中文语音（FLEURS）上 35 个汉字全部对齐、时间单调；合成字边界评估中位误差约 0.18s。

**风险**：
- 许可证是自定义 FunASR Model License，**商业分发前需复核条款**（见 `ANDROID_FEASIBILITY.md`）。
- FunASR 依赖较重（引入 scikit-learn、sentencepiece、oss2、librosa 等），与我们的 transformers 链路并行存在一定依赖膨胀风险。
- 模型是 **FP32 PyTorch**，导出 ONNX/TFLite 需要额外工作（见 Android 可行性）。

## 2. 日文：`reazon-research/japanese-wav2vec2-base-rs35kh`

| 项目 | 值 |
|---|---|
| 仓库 | https://huggingface.co/reazon-research/japanese-wav2vec2-base-rs35kh |
| 架构 | `Wav2Vec2ForCTC` (base, 12 层, 768 dim) |
| 参数 / 体积 | 96.7M 参数 / 387 MB `model.safetensors` |
| 训练数据 | ReazonSpeech v2.0 (35k+ 小时) |
| 分词 | `Wav2Vec2CTCTokenizer`，词表 3003，**字符/子词级**（假名、汉字、`▁` 前缀子词） |
| 许可证 | **Apache-2.0** |
| 后端 | transformers（与英文同一套代码） |
| 对齐机制 | 字符/子词 CTC → 我们的 Viterbi 强制对齐 |

**关键调研发现**：词表**不是**需要外部 G2P 的音素表，而是字符/子词表。实测：

```
'こんにちは世界' -> ['こ','ん','に','ち','は','世界']     # 世界 是一个 token
'ありがとう'     -> ['あり','が','と','う']               # あり 是一个 token
'君の名前は'     -> ['君','の','名','前','は']
```

因此：(a) 可以直接对**原文**字符对齐，无需 pyopenjtalk 等 G2P 前端；(b) **绝不能假设一个汉字=一个音节**——一个 token 可能是多字词，一个字也可能跨多个 token。我们通过 `AlignUnit.raw` 保留「发音单元 ↔ 原文字符」的映射。

**验证结论（已运行）**：真实日语语音（FLEURS）上模型 CTC 转写与参考仅差几字（`そそられる`→`そられる` 等），19 个单元覆盖 0.0–7.96s、单调。**合成（espeak）日语几乎不可用**（见局限）。

**风险**：长音频 CER 高（模型卡：JSUT-BOOK 82.8%），但强制对齐只依赖局部声学后验，不受长距离解码影响；仍需真实歌声验证。

## 3. 英文：`facebook/wav2vec2-base-960h`

| 项目 | 值 |
|---|---|
| 仓库 | https://huggingface.co/facebook/wav2vec2-base-960h |
| 架构 | `Wav2Vec2ForCTC` (base) |
| 参数 / 体积 | 94.4M 参数 / 378 MB |
| 训练数据 | LibriSpeech 960h |
| 分词 | 字符级，词表 32（A-Z、空格=`|`、`'`） |
| 许可证 | **Apache-2.0** |
| 后端 | transformers |
| 对齐机制 | 字符 CTC → 我们的 Viterbi 强制对齐 |

**验证结论（已运行）**：合成词边界样本上**起始时间 MAE = 67 ms**（p90 = 98 ms）；真实语音（LibriSpeech 13.7s，42 词）对齐单调、覆盖合理。

**风险**：单词结尾容易偏早（合成样本 end MAE ≈ 0.39s，真实语音上明显改善）——属于 CTC 尾音/域问题，见 `KNOWN_LIMITATIONS.md`。

## 4. 歌声域：V0.2 实测结论（新增）

上述三个模型都是**语音**训练的，直接用于**歌声**会出现域不匹配。V0.2 用真实 GTSinger
歌声（见 `docs/SINGING_EVAL.md`）实测：

- 中文 `fa-zh`：歌声上依然可靠（汉字级 start MAE ≈ 0.15 s），是三者中最稳的。
- 英文 `wav2vec2-base-960h`：结束边界优于启发式，但**起音**定位不稳（P90 ≈ 1 s）。
- 日文 `japanese-wav2vec2-base-rs35kh`：约**一半**段落的 CTC 对齐在歌声上塌缩
  （被 V0.2 检测并判为失败），有效段落精度尚可但覆盖率低。

结论：语音模型的 ASR 表现不能代表歌声对齐表现。后续应评估**歌声域模型**或轻量域适配
（见 `docs/NEXT_STEPS.md`）。数据集授权与标注质量调研见 `docs/SINGING_EVAL.md`
（GTSinger CC BY-NC-SA 4.0；OpenCpop CC BY-NC-ND；PJS CC BY-SA 4.0；MIR-1K 链接失效）。

## 5. 允许的替代模型（调研，未运行）

| 语言 | 候选 | 说明 |
|---|---|---|
| zh | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | 字符级中文 CTC，可与英文共用代码；但是 **large（~1.2 GB / 315M 参数）**，本机内存（~1.1 GB 可用）难以承载 |
| ja | `reazon-research/japanese-wav2vec2-large-rs35kh` | 更高精度版本，~1.2 GB，内存受限 |
| en | `facebook/wav2vec2-large-960h` | 更准但更大（~1.2 GB），本阶段不采用 |
| 通用 | Whisper / Wav2Vec2-CTC + 更好的 forced aligner（如 MMS aligner） | 精度可能更高，但模型大、许可证与体积不利于 Android |

**结论**：V0.1 的三模型在「体积/内存/许可证/可运行性」上综合最优。中文的 fa-zh 是唯一「原生对齐器」；英/日共用同一套自研 CTC Viterbi，代码复用度高。

## 6. 许可证与再分发

| 模型 | 许可证 | 可否提交权重到 Git | 备注 |
|---|---|---|---|
| facebook/wav2vec2-base-960h | Apache-2.0 | **否**（体积大） | 运行时下载，保留 LICENSE |
| reazon-research/...rs35kh | Apache-2.0 | **否** | 运行时下载 |
| funasr/fa-zh | FunASR Model License (other) | **否** | 商业使用前需复核；本仓库不分发权重 |

所有权重均存放于 `models/`（已 `.gitignore`），由用户按需下载。
