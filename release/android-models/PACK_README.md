# Android 模型包 · v0.1.0-beta.2（测试版）

> ⚠️ **这是测试版（prerelease），尚未通过逐字精度验收。**
> 三个模型的权重都能在 Android 上用 ONNX Runtime 直接加载、推理是确定性的，
> 但**逐字/逐词时间轴尚未与人工参考打轴比对验收**。在 GTSinger 清唱留出集上，
> 相对**人工边界**的绝对起始 MAE 约为 en **100 ms**、ja **119 ms**、zh **181 ms**；
> INT8 与 FP32 的差在清唱上 en **2.8 ms**、zh **0.8 ms**、ja **79 ms**。
> **请勿据此宣称逐字高亮精度。** 详见 `docs/QUANTIZATION_CALIBRATION.md`。

本包由 `tools/build_android_release.py` 从 `android_export/` 的已验证产物生成；
机器可读清单见 **`android-models.json`**，校验和见 **`checksums.sha256`**。
设备端**不需要 Python**。

> **beta.2 变更**：日文 INT8 改为**选择性量化**（CTC 头 `lm_head` 不量化）。
> 日文端到端逐字符起始 MAE 由 147.06 ms 降到 **118.81 ms**，GTSinger 留出集上
> INT8-vs-FP32 由 105.9 ms 降到 **79.05 ms**，文件仅增大 0.5 MB（95.19 → 95.72 MB）。
> 英文、中文产物未变。

## 1. 资产清单

| 文件 | 语言 | 角色 | 字节 | SHA-256 |
|---|---|---:|---:|---|
{{ASSET_TABLE}}

## 2. 每个语言包需要什么

| 语言 | ONNX 权重 | 词表/分词 | 前后处理 |
|---|---|---|---|
| **en** | `en-wav2vec2-base-960h-int8.onnx` | `en-vocab.json`（32 token） | **归一化已烘焙进图**，直接喂原始 PCM |
| **zh** | `zh-fa-zh-int8.onnx` | `zh-tokens.json`（8404）、`zh-seg_dict` | **前端已烘焙进图**（fbank+LFR+CMVN，dither=0）；仍需 `ts_prediction_lfr6_standard` 后处理 |
| **ja** | `ja-hubert-phoneme-ctc-int8.onnx` | `ja-vocab.json`（48 音素） | **不做任何归一化**（原样透传）；需要设备端 G2P |

`zh-am.mvn` 是中文前端的 CMVN 参数，**已经烘焙进 ONNX**，此处仅作审计留档。

## 3. 校验下载

```bash
sha256sum -c checksums.sha256
```

## 4. 接入协议（Android / Kotlin + ONNX Runtime）

统一的张量契约（详见各语言 `manifest.json` 的 `input` / `outputs`）：

| 语言 | 输入 | 输出 | 帧率 |
|---|---|---|---|
| en | `input_values` float32 `[1,N]` 原始 16k 单声道 PCM，[-1,1] | `logits` / `log_probs` `[1,T,32]` | 20 ms（320 采样） |
| zh | `input_values` float32 `[1,N]`；`token_num` int64 `[1]` | `us_alphas` / `us_peaks` `[1,T3]` | T3 = 3 × 编码器帧，时间步 20 ms |
| ja | `input_values` float32 `[1,N]` 原始 PCM，**不归一化** | `logits` / `log_probs` `[1,T,48]` | 20 ms（320 采样） |

### 4.1 英文（CTC）

1. 文本规范化：NFKC → 弯引号转 `'` → 删除 `[^A-Za-z0-9' ]` → 折叠空白 → **转大写**
   （等价 `src/autosync/lyrics/normalize.py::normalize_en`）。
2. 按 `|`（id=4）切词、按空格切 token；`blank_id = 0`。
3. 对每个歌词行窗口跑模型 → `log_probs` → **自己实现 CTC Viterbi**（blank=0，
   与 `src/autosync/alignment/ctc.py::forced_align` 同算法）。
4. 不可行时**必须显式回退原始 LRC**，不得平均分配伪造时间。

### 4.2 中文（MonotonicAligner / CIF，非 CTC）

1. 文本规范化：NFKC → **只保留 CJK/假名/数字（拉丁字母被有意丢弃）**。
2. 用 `zh-tokens.json`（+`zh-seg_dict` 处理英文子词 `@@`）分词，得到字符 token 数。
3. 输入 `token_num = token 数 + 1`。
4. 取 `us_alphas`/`us_peaks` 后用 `ts_prediction_lfr6_standard` 还原逐 token 时间：
   `TIME_RATE = 0.02 s`、`force_time_shift = -1.5`、`START_END_THRESHOLD = 5`、
   `MAX_TOKEN_DURATION = 12`；参考实现见仓库 `android_export/zh/postprocess_reference.py`
   （已与 FunASR 官方实现逐 token 比对，**max|Δ| = 0 ms**）。
5. **注意**：本号模型对 `token_num` 敏感。真实文本与音频不匹配时，官方实现会按
   `alphas / (alphas.sum()/(n+1))` 重新归一化，Android 端必须照此实现。

### 4.3 日文（音素 CTC，实验性）

1. **G2P 是必需依赖，且不在本包内**：建议随 App 打包 OpenJTalk（经 JNI）+
   NAIST 词典 `open_jtalk_dic_utf_8-1.11`。
   - 该词典为 **modified BSD（NAIST 三条款）**：分发时必须复现其版权声明与免责声明。
   - `htsvoice`（合成用语音库）**不要**一起打包，它与 G2P 无关。
   - 真值对拍用仓库 `android_export/ja/g2p_reference.py`（116 行真实歌词字段级 0 不一致）。
2. G2P 得到音素序列（首尾各补一个 `pau` = id 46），过滤词表外音素。
3. 跑模型 → `log_probs` `[1,T,48]` → CTC Viterbi（`blank_id = 0`）。
4. 音素区间 → 字符区间：按 `map_spans_to_char_units` 的规则（避免首字符吞掉前导 `pau`；
   无音素的字符给零宽区间并标 `uncertain`），**必须能无损重建原文**。
5. 帧 → 时间：`start = start_frame × 0.02`，`end = (end_frame + 1) × 0.02`。

## 5. 已知限制（务必如实对待）

- **量化漂移**：INT8 与 FP32 的起始时间差，分两个域报（`docs/QUANTIZATION_CALIBRATION.md`）：
  - **GTSinger 清唱**（n=12/语言）：en **2.78 ms**、ja **79.05 ms**、zh **0.82 ms**；
  - **`Test/` 整混音**（en，45 行/230 词）：en **121.30 ms**。
  说明英文的“121 ms”几乎全部来自**整混音域**，不是量化的普遍代价；**日文是唯一在清唱上
  也明显受量化影响的链路**（已用“排除 CTC 头”改进，见上）。
- **绝对精度**（相对 GTSinger 人工边界，清唱留出集）：en FP32 **100.1 ms** / INT8 **99.4 ms**；
  ja FP32 **118.7 ms** / INT8 **154.2 ms**；zh FP32 **181.4 ms** / INT8 **181.0 ms**。
  这是相对人工打轴的绝对数字，与上面的“INT8-vs-FP32 一致性”不同，勿混用。
- **未达标的验收项**：en 17 项中 4 项、zh 20 项中 6 项、ja 15 行中 3 项未通过（详见各
  `REPORT.md`）。其中 fp32 的 logits 阈值（`5e-4`/`5e-6`）比 PyTorch 自身内核差异还严。
- **zh 的 `us_peaks` 契约偏差**：用闭式 `cumsum/floor` 替代了 `cif_wo_hidden` 的迭代，
  `max|Δ| = 12`，2 个边界样本峰值数不一致；fp32 端到端时间戳仍与引擎 693/693 完全一致。
- **ja 仍是实验功能**：声学模型训练于朗读语音（ReazonSpeech），歌声域不匹配。
- **许可**：三个权重均为 **Apache-2.0**，但本包是**修改版**（INT8 量化/前端烘焙），
  署名与修改说明见 `NOTICE.md`。
- 上述均为**桌面 x86 CPU 单线程**参考值；ARM 需在真机复测。

## 6. 版本与协议

- 包版本：`0.1.0-beta.2`，发布 Tag：`android-models-v0.1.0-beta.2`（prerelease）。
- 最低后端协议：`min_backend_protocol = 1`（schema `melox.autosync.onnx.1`）。
- 固定上游版本：en/ja 使用 HF 快照 revision（见 manifest `source_revision`）；
  zh 为 ModelScope `iic/speech_timestamp_prediction-v1-16k-offline`。
