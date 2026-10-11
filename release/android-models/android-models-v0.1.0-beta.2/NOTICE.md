# NOTICE

MeloX-AutoSync Android 模型包 `0.1.0-beta.1`
（发布 Tag：`android-models-v0.1.0-beta.1`，仓库 <https://github.com/lladlam/MeloX-AutoSync>）

本包含有以下上游模型的**修改版权重**。修改方式与上游版权归属如下。
每个上游组件均以其自身许可发布；本包不改变其版权，也不暗示上游作者为
MeloX-AutoSync 背书。

## 1. 英文 · `facebook/wav2vec2-base-960h`

- 版权归 Facebook AI / Meta 及其贡献者所有，**Apache License 2.0**。
- 修改：动态 INT8 量化（QInt8 权重、按通道缩放、仅 MatMul/Gemm）；
  zero-mean unit-variance 输入归一化（`eps=1e-7`）烘焙进计算图；
  导出为单文件 ONNX（opset 17），并额外输出 `log_probs`。

## 2. 中文 · ModelScope `iic/speech_timestamp_prediction-v1-16k-offline`

- 上游（阿里达摩院 / FunASR）发布，**Apache License 2.0**
  （模型卡 README front matter 声明；核实记录见仓库 `docs/LICENSE_AUDIT.md` §5）。
- 修改：动态 INT8 量化（同上，且前端 DFT/Mel 权重显式排除在量化之外）；
  Kaldi 风格 fbank + LFR(7,6) + CMVN 前端**烘焙进图**（导出时 `dither=0` 以保证确定性）；
  `us_peaks` 以闭式 `cumsum/floor` 等价式替代 `cif_wo_hidden` 迭代（已知偏差，
  见 `android_export/zh/REPORT.md` §7）。
- FunASR 工具链本身为 **MIT**，未包含在本包中。

## 3. 日文 · `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`

- 版权归其作者所有，**Apache License 2.0**。
- 基础模型 `rinna/japanese-hubert-base` 亦为 Apache-2.0。
- 修改：动态 INT8 量化（QInt8 权重、按通道缩放、仅 MatMul/Gemm）；导出为
  单文件 ONNX（opset 17），并额外输出 `log_probs`。**未添加任何输入归一化**
  （上游 `preprocessor_config.json` 为 `do_normalize=false`）。

## 4. 未包含在本包中的依赖（仅说明）

| 依赖 | 许可 | 说明 |
|---|---|---|
| pyopenjtalk | MIT | 日文 G2P 的 Python 参考实现 |
| Open JTalk 词典 `open_jtalk_dic_utf_8-1.11` | **modified BSD（NAIST）** | 日文 G2P 必需；若随 App 打包，须复现 NAIST 版权声明与免责声明。约 107 MB |
| HTS voice `mei_normal.htsvoice` | 独立许可 | 仅用于语音合成，**不要**为 G2P 打包 |
| GTSinger 等评测数据 | CC BY-NC-SA 4.0 | **不随包分发**，仅本地评测 |

## 5. 许可全文

本包随附 `LICENSE-APACHE-2.0.txt`（Apache License 2.0 全文）。
按 Apache-2.0 第 4(b) 条，上述修改均已在本文件中声明。
