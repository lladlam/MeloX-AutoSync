# Android 移植可行性（ANDROID_FEASIBILITY）

> 本阶段不写 Android 代码。本文评估「把 V0.1 的模型搬到 Android 是否可行、代价多大」。

## 1. 现状与目标

- 桌面验证（Linux + Python）已完成三语言推理（见 `EXPERIMENT_RESULTS.md`）。
- Android 目标：**离线、按需下载单语言模型、可量化、内存可控**。
- 关键约束：权重不能进 APK（体积 + 许可证），应从服务端按语言下载到应用私有目录。

## 2. 模型体积 / 内存（实测）

| 语言 | 格式 | 磁盘 | 桌面峰值 RSS | 参数 |
|---|---|---|---|---|
| en | safetensors FP32 | 378 MB | 862 MB | 94.4M |
| ja | safetensors FP32 | 387 MB | 865 MB | 96.7M |
| zh | PyTorch FP32 (`model.pt`) | 151 MB | 1080 MB | 38M |

**FP32 直搬不可接受**：桌面峰值已接近 1 GB，APK 下载体积也过大。必须量化。

## 3. 英文 / 日文（Wav2Vec2 CTC）——可行性较高

两者都是标准 `Wav2Vec2ForCTC`，导出路径成熟：

1. **导出 ONNX**：`torch.onnx.export`（动态轴处理时间维）。
2. **动态量化 int8**：ONNX Runtime 动态量化，体积约 378 MB → **~100 MB**，内存相应下降。
3. **推理**：Android 用 **ONNX Runtime Mobile** 或 **sherpa-onnx**（已内置针对 wav2vec2 的 CTC/forced-alignment 支持，含 Android 示例与量化工具）。
4. **对齐算法**：我们的 CTC Viterbi（纯 numpy，~150 行）可移植为 Kotlin/C++，与 ONNX 输出的 log-probs 对接；算法与语言无关。

**结论**：en/ja 移植**可行性高**，是首选落地方向。预计单模型量化后 ~100–130 MB，运行内存 < 500 MB。

## 4. 中文（fa-zh）——可行性较低

- `fa-zh` 是 FunASR 自定义 `MonotonicAligner`（SANMEncoder + CifPredictorV3），**不是标准架构**，ONNX 导出需要 FunASR 侧支持，工作量大。
- 需要连着 WavFrontend（kaldi fbank，LFR 处理）一起移植。
- 备选方案：
  - **A. 换用字符级中文 Wav2Vec2 CTC**（如 XLSR-53 中文，large）。可与 en/ja 共用 ONNX + Viterbi 管线，但模型大（~1.2 GB FP32 → 量化 ~300 MB）。
  - **B. 自实现 MonotonicAligner 推理**（C++/Kotlin）——工作量大，但体积最小。
  - **C. 云端/本地二选一**：中文暂缓，或仅在桌面端提供。

**结论**：中文移植**需要额外工程投入**，优先级低于 en/ja。许可证（FunASR Model License）也需商用复核。

## 5. 落地建议（下一阶段）

1. 先做 **en/ja 的 ONNX 导出 + int8 量化**，并在桌面验证量化后对齐精度损失。
2. 用 **sherpa-onnx** 评估端到端（含 Android demo），确认 log-probs 接口与我们的 Viterbi 对齐。
3. CTC Viterbi 用 Kotlin 实现并用**同一组测试向量**与 Python 版对齐（保证数值一致）。
4. 中文单独评估「换 CTC 模型」与「移植 fa-zh」两条路。
5. 权重按语言懒下载到应用私有目录，带许可证与校验和。

## 6. 风险清单

| 风险 | 说明 |
|---|---|
| 量化精度损失 | int8 可能损伤对角线概率，需量化后重测对齐误差 |
| 内存 | 低端设备 500 MB 以上预算偏紧，需进一步压缩（int8 + 分块） |
| 中文架构 | 非标准，导出成本高 |
| 许可证 | fa-zh 的 FunASR Model License 需法务确认 |
| 前端一致性 | Android 侧 fbank/重采样需与 Python 端严格一致，否则对齐漂移 |
