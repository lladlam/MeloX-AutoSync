# 日语歌声模型的 Android 可行性（JAPANESE_ANDROID_FEASIBILITY）

> 结论先行：**许可最干净、最适合 Android 本地部署的是 HuBERT 音素 CTC（Apache-2.0）**；
> **精度/资源最优的是 pyshiro，但 GPLv3 代码 + 权重许可不明，不适合直接打包进闭源 App**。
> 本轮不做 Android 集成；本文只做可行性评估。

## 1. 候选对比

| 维度 | pyshiro (HSMM) | HuBERT 音素 CTC | TIFA |
|---|---|---|---|
| 算法 | 两遍 HMM→HSMM 强制对齐（已知音素序列） | 音素 CTC + Viterbi（已知音素序列） | 音素对齐 + 发音打分 |
| 权重体积 | **v2 0.49 MB / v1 0.13 MB** | ~361 MB (FP32) | ~151 MB (zip) |
| 运行时 | 纯 **numpy + numba**（可选 Python→C 移植） | ONNX Runtime / PyTorch Mobile | 有官方 **ONNX 管线** |
| 峰值内存 | 低（模型 <1MB，特征 16k/5ms） | 中（~360MB 权重） | 中 |
| 预测 30 段耗时 | ~15s（含全部）| ~20s | 未实测 |
| 语言包独立性 | 日语专用 | 日语专用 | 多语言（可只装日语资源） |
| 许可 | **GPLv3（代码）/ 权重许可不明** | **Apache-2.0** | **MIT（代码）** |
| 再分发权限 | 代码需 GPL；权重**不可**打包 | 可（Apache-2.0） | 可（MIT）；需核实 g2pflow 及其词典许可 |

## 2. 关键结论

### 2.1 pyshiro — 精度/体积最佳，许可需以权重为准

- **优点**：模型 0.49 MB、纯 Python/numpy/numba、零失败、中位误差 ~0.1 s。理论上是手机端最轻的方案。
- **更正（V0.5 许可审计）**：**MeloX-Android 是 GPL-3.0**，因此 pyshiro 的 **GPLv3 代码与其兼容**，
  「GPL 不能进闭源 App」的说法**不成立**；集成时需遵守 GPLv3（提供源码、保留声明）。
- **真正的阻塞是权重**：pyshiro 的预训练 `.hsmm`（仓库与子模块）**没有明确许可**，
  **不得**放进可公开下载的语言包。可继续本地研究/评测，或取得作者授权、或用许可清晰的数据重训。
- G2P 需要 **pyopenjtalk**（代码 MIT）+ Open JTalk 词典（modified BSD，约 22 MB，可分发）。
- 结论：**代码可用，权重不可再分发**。见 `docs/LICENSE_AUDIT.md`。

### 2.2 HuBERT 音素 CTC — 许可与部署最优

- **Apache-2.0**，可再分发；标准 HuBERT transformer，可导出 ONNX，量化后约 **90 MB**。
- 复用我们既有的 **CTC Viterbi**（与英文完全同一套代码），工程成本最低。
- V0.5 修正索引偏移后，**成为起始指标最优**：sMAE **0.365**、中位 0.063、P90 1.121（优于 pyshiro）。
- 仍需 G2P（音素序列）→ 可复用 pyopenjtalk（MIT/BSD，可分发）。

**V0.5 结论：默认应选 HuBERT 音素 CTC**（代码与权重均可再分发，且精度最好）；
pyshiro 作为桌面/服务端的高精度路径，权重未授权前不进入公开语言包。

### 2.3 TIFA — MIT + ONNX 管线，工程较重

- MIT 许可 + 官方 ONNX 部署（`spectrogram/prepare/model/score/select.onnx`），对 Android 友好。
- 但依赖链重（g2pflow / lightning / dask / praat-parselmouth 等），且需核实 g2pflow 词典许可。
- 本轮未完成端到端运行（见 `JAPANESE_MODEL_COMPARISON.md` §4/§5）。

## 3. 单一语言包设计

三个候选都是**日语专用**（TIFA 为多语言，但可按配置只加载日语资源），满足
「日语模型可独立安装，不强迫下载中/英模型」的要求：

- pyshiro：一个 `.hsmm`(0.5MB) + phonemap + G2P 词典(22MB)。
- HuBERT CTC：一个 ~361MB 权重（量化 ~90MB）+ tokenizer + G2P 词典。
- TIFA：模型 zip(151MB) + 日语 G2P 词典。

## 4. 建议路线（下一阶段）

1. **首选 HuBERT 音素 CTC** 做 Android 移植验证：ONNX 导出 + int8，量化后**重跑本盲评**确认精度损失。
2. 把 **pyshiro 作为服务端/桌面高精度路径**（许可允许的场景），或用许可清晰的数据重训后再考虑手机端。
3. G2P（pyopenjtalk，BSD）与词典随语言包分发，注意许可与体积。
4. **不要**把许可不明的 pyshiro 权重打进公开产物。

## 5. 明确未做

- 未做 ONNX 导出、量化、Android 集成或 GUI（本轮范围外）。
- 未对 TIFA 做量化/部署验证。
