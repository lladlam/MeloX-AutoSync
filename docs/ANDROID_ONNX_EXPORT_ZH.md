# 任务：把中文 AutoSync 模型（funasr/fa-zh）导出为 Android 可用的 ONNX / INT8

> 这份文档是一份**可直接交给 AI 执行的任务书**，与 `ANDROID_ONNX_EXPORT_PROMPT.md`（英文链路）
> 并列。三语言是三份独立任务书，可并行执行。
>
> 执行者需要**真实运行**所有命令、**真实验证**所有数字，并**如实报告**失败项。
> 禁止编造验证结果，禁止用启发式（平均分配）冒充模型输出。

---

## 0. 背景与范围

- 仓库：`/home/lladlam/项目/MeloX-AutoSync`（Python 端引擎，V1.0 已完成）。
- 消费方：MeloX-Android。
- **中文链路与英文完全不同**：`funasr/fa-zh` 是 **MonotonicAligner**（SANM 编码器 + CIF/CIF2
  单调对齐预测器），它**直接吃「音频 + 已知文本」并输出每个字符 token 的 `(start_ms, end_ms)`**。
  - **没有 CTC**：不要套用 `src/autosync/alignment/ctc.py::forced_align`，也不要导出 CTC 头。
  - 因此 Android 端不需要 CTC Viterbi，需要的是：**前端（fbank+LFR+CMVN）+ 编码器 + CIF 上采样头**
    的 ONNX 图，以及一个**确定性的时间戳后处理**（见 §2.4）。
- 本任务书只覆盖 `funasr/fa-zh`（= `iic/speech_timestamp_prediction-v1-16k-offline`）。
- **许可提醒（与英文不同）**：英文 `wav2vec2-base-960h` 是 Apache-2.0；`fa-zh` 是
  **FunASR Model License**（见 https://github.com/modelscope/FunASR ）。导出物是否可随 App 分发
  必须先复核，并在 `manifest.json` 与 `REPORT.md` 中如实标注。**权重不得提交进 Git。**
- 不要修改 `src/autosync/` 下现有引擎的默认行为；导出与验证脚本请建在交付目录内。

## 1. 环境

```bash
cd /home/lladlam/项目/MeloX-AutoSync
.venv/bin/python -V                       # 已验证：Python 3.12 / torch 2.14.1+cpu / numpy 2.5.3
.venv/bin/python -c "import onnx, onnxruntime; print(onnx.__version__, onnxruntime.__version__)"
# 已验证：onnx 1.23.2 / onnxruntime 1.31.0（本轮已装；缺失时重装即可）
# uv pip install --python .venv/bin/python "onnx>=1.17" "onnxruntime>=1.20"
```

中文模型还依赖 FunASR 可选包。**注意** `uv sync`/`uv run` 换 extra 会裁剪已装依赖，
需要时把它们放在一起请求：

```bash
uv sync --extra dev --extra funasr --extra onnx
uv run --extra dev --extra funasr --extra onnx python -c "import funasr; print(funasr.__version__)"
# 已验证：funasr 1.4.16 / modelscope 1.40.2 / kaldi-native-fbank（无 torchaudio）
```

模型权重**已经在本地**，不要重复下载，离线使用：

```
models/funasr/fa-zh/
├── model.pt              158,469,618 bytes    # 已核实
├── config.yaml                                # MonotonicAligner + SANMEncoder + CifPredictorV3
├── tokens.json                                 8404 tokens（<blank> = id 0）
├── seg_dict              8,287,834 bytes      # 文本分词词典
└── am.mvn                                      # CMVN：<AddShift> 均值 + <Rescale> 方差
```

## 2. 硬性契约

> 本节是**本任务书定义**的契约；Android 端中文链路尚未实现，导出验证通过后以此为准定稿。
> 任何与本节不符的实现都必须在 `REPORT.md` 中作为偏差报告，不得默默改契约。

### 2.1 张量接口

| 项 | 值 |
|---|---|
| 输入 1 名 | `input_values` |
| 输入 1 dtype / shape | `float32` / `[1, N]`，**N 动态**（任意长度，≥ 400 采样） |
| 输入 1 内容 | **原始 16kHz 单声道 PCM 浮点，范围 [-1, 1]，不做任何外部归一化**（前端烘焙进图，见 2.2） |
| 输入 2 名 | `token_num` |
| 输入 2 dtype / shape | `int64` / `[1]`，值 = **文本 token 个数 + 1**（预测器 `predictor_bias=1`；已核实推断路径即 `len(ids)+1`） |
| 输出 1 | `us_alphas`，`float32`，`[1, T3]` |
| 输出 2 | `us_peaks`，`float32`，`[1, T3]` |
| 帧率关系 | `T3 = 3 × T_enc`；`T_enc = ceil(N_fbank / 6)`；`N_fbank` 由 25ms/10ms 前端决定，**以实测为准**（见下） |

已核实的真实形状（用 `experiments/audio/fleurs_zh_sample.wav` 7.44s = 119,040 采样实测）：

```
fbank+LFR 特征   (1, 124, 560)      # 560 = 80 mel × 7 (lfr_m)
编码器输出       (1, 124, 320)      # d_model = 320
us_alphas/us_peaks (1, 372)         # 372 = 124 × 3
```

`manifest.json` 必须写明：前端实现方式、帧数公式、`T3` 与 `N` 的关系（用至少 3 种长度实测核对）。

模型必须以 **eval 模式、float32、无 attention mask** 导出；图内不允许出现 `com.microsoft`
贡献算子，只允许标准 ONNX 算子；**opset = 17**；单文件、无 external data。

### 2.2 前端（必须烘焙进图，且必须实测比对）

推理时的前端**逐值等价**目标是 FunASR `WavFrontend` 的这一串（全部已核实）：

| 步骤 | 参数（来自 `config.yaml` / 实测） |
|---|---|
| 采样放大 | `waveform * 32768`（`upsacle_samples=True`，Kaldi 约定） |
| fbank | Kaldi 风格：`window=hamming`、`n_mels=80`、`frame_length=25ms`、`frame_shift=10ms`、`energy_floor=0.0`、`snip_edges=True`、`use_log_fbank=True`、`use_power=True`、`preemphasis_coefficient=0.97`、`remove_dc_offset=True` |
| LFR | `lfr_m=7`, `lfr_n=6`（左补 `(m-1)//2` 帧，末帧重复补齐） |
| CMVN | `am.mvn` 的 `<AddShift>` 均值 + `<Rescale>` 方差：`x = (x + mean) * var` |

**fbank 后端注意**：本机**未安装 torchaudio**，FunASR 走 `kaldi-native-fbank` 实现
（`funasr/utils/fbank.py::_fbank_knf`，已核实 `_HAS_TORCHAUDIO=False`）。
导出图必须与该后端对齐；若后续环境里出现 torchaudio，需重新核对（两者在边界帧上可能不同）。
其余默认参数：`preemphasis_coefficient=0.97`、`remove_dc_offset=True`、`use_power=True`、
`use_log_fbank=True`、`raw_energy=True`、`round_to_power_of_two=True`。

**`dither` 必须固定为 0**。实测证据（务必写进报告）：

- 默认 `dither=1.0` 时，`extract_fbank` 对**同一段音频两次调用**结果不同：
  实测 `max|Δ| = 0.3216`（特征本身带随机噪声）。
- 同一段音频 + 同一文本跑 5 次 `AutoModel.generate`，**时间戳抖动为 0.0 ms**（该样本上稳定），
  但这**不能**用来证明前端可复现。
- `dither=0` 时前端完全可复现：两次调用 `max|Δ| = 0.0`，3 次生成时间戳 `max|Δ| = 0.0 ms`。

因此：导出图的 dither 恒为 0（确定性的唯一办法），并必须报告
「Python dither=0 vs Python dither=1.0（多次）」以及「ONNX vs Python dither=0」的时间戳差异。

> 若前端烘焙无法在下列容差内复现，允许退回**方案 B**并如实报告：
> 输入改为 `speech` `float32 [1, T, 560]`（已 LFR+CMVN 的特征），Android 端自算 fbank/LFR/CMVN。
> 方案 B 必须同时交付该前端的 golden 对照（Python 侧输入/输出），否则 Android 无法自查。

### 2.3 词表、分词与文本规范化

- 词表：`tokens.json`，**8404 项**，**id 顺序即 token 顺序**：
  `0:<blank>`、`1:<s>`、`2:</s>`、其余为**字符级 token**（含英文子词，形如 `and@@`、`do@@`）。
- 分词器：FunASR `CharTokenizer`（`tokenizer_conf.split_with_space: true`）。
  Android 端必须复现同一分词；**英文子词以 `@@` 结尾表示「与下一 token 拼接」**。
- `blank_id = 0`（该模型不用于 CTC，但契约里保留以便统一）。
- 文本规范化：本仓库 `src/autosync/lyrics/normalize.py::normalize_cjk`
  = NFKC → **只保留 CJK 汉字、平/片假名、全角数字与 ASCII 数字** → 其余（含拉丁字母）**全部删除**。
  这正是中文歌里纯拉丁行失败的已知原因（`docs/KNOWN_LIMITATIONS.md`），Android 端要照此实现，
  不要在导出环节"顺手修复"它。
- 模型返回文本是**空格分隔的字符 token**；`stamps[i]` 与第 i 个 token 对应（已核实：10 字 → 10 个时间戳）。

### 2.4 时间戳后处理（Android 端实现，本任务书给出规格）

ONNX 只输出 `us_alphas` / `us_peaks`；逐 token 时间戳由 FunASR 的
`ts_prediction_lfr6_standard` 计算。已核实的常量与算法：

```
START_END_THRESHOLD = 5
MAX_TOKEN_DURATION  = 12
upsample_rate       = 3
TIME_RATE           = 10.0 * 6 / 1000 / 3 = 0.02   # 秒/上采样帧
force_time_shift    = -1.5                         # 单位：上采样帧
```

流程（用 `N = T_enc × 3` 截断 `us_alphas/us_peaks` 后）：

1. `fire_place = where(peaks >= 1.0 - 1e-4) + force_time_shift`（`1.0 - 1e-4` 与 `-1.5` 都必须照抄）。
2. 若 `len(fire_place) != len(tokens) + 1`：把 `alphas` 按 `alphas / (alphas.sum()/(len(tokens)+1))`
   重新归一化，再用同一阈值重算 `peaks` 与 `fire_place`。
3. 首尾：`fire_place[0] > 5` 时补一个 `[0, fire_place[0] * TIME_RATE]` 的起始静音段；
   结尾同理按 `MAX_TOKEN_DURATION=12` 与尾部峰值处理。
4. 逐 token 生成 `[start_ms, end_ms]`（FunASR 返回毫秒整数；`_standard` 内部按 `TIME_RATE` 换算）。

**要求**：交付目录内提供 `postprocess_reference.py`，实现对 FunASR
`funasr.utils.timestamp_tools.ts_prediction_lfr6_standard` 的**独立逐行对照**（同输入 → 同输出），
并给出与官方实现的逐 token `max|Δ|`（目标 0ms；若有差异必须解释原因）。Android 端按此规格实现。

## 3. 量化要求

- 用 **ONNX Runtime 动态量化**（`onnxruntime.quantization.quantize_dynamic`），
  `weight_type=QuantType.QInt8`，默认只量化 MatMul/Gemm；卷积前端与 LayerNorm 保持 fp32。
- 量化后**必须仍是单文件、标准算子、CPU ExecutionProvider 可加载**。
- 参考量级：`model.pt` = 158,469,618 bytes。INT8 目标 **≤ 60 MB**；若超出，报告实际值与原因，
  **不要偷偷换模型**。

## 4. 交付物（放到 `android_export/zh/`）

```
android_export/zh/
├── fa-zh-fp32.onnx                    # 仅用于验证对比
├── fa-zh-int8.onnx                    # ★ Android 端实际使用
├── manifest.json                      # 契约清单，见 4.1
├── test_vectors/                      # 设备端自检向量，见 4.2（gitignore）
│   ├── vector1.wav  vector1.expected.json
│   ├── vector2.wav  vector2.expected.json
│   └── vector3.wav  vector3.expected.json
├── export_model.py                    # 可复跑导出 + 量化
├── validate_onnx.py                   # 生成 REPORT.md 的脚本，可复跑
├── postprocess_reference.py           # §2.4 后处理的独立参照实现 + 官方对照
└── REPORT.md                          # 验证报告，见 7
```

> 大文件不要 `git add`（`.gitignore` 已排除 `android_export/**/*.onnx` 与
> `android_export/**/test_vectors/`；脚本 / manifest / REPORT.md 属于我们自己的产出，可以提交）。
> 交付时报告**绝对路径 + 字节数 + sha256**。

### 4.1 `manifest.json` 格式（字段名不要改）

```json
{
  "schema": "melox.autosync.onnx.1",
  "language": "zh",
  "source_model": "funasr/fa-zh",
  "source_model_alias": "iic/speech_timestamp_prediction-v1-16k-offline",
  "license": "FunASR Model License (verify before redistribution)",
  "sample_rate": 16000,
  "frontend": {
    "baked_into_graph": true,
    "implementation": "kaldi-fbank(hamming,80mel,25ms/10ms,snip_edges) + LFR(7,6) + CMVN(am.mvn)",
    "dither": 0.0,
    "feature_dim": 560
  },
  "frame_hop_samples": 160,
  "encoder_frame_seconds": 0.06,
  "upsample_rate": 3,
  "timestamp_frame_seconds": 0.02,
  "input": { "name": "input_values", "shape": [1, null], "dtype": "float32",
             "preprocessing": "raw 16k mono PCM in [-1,1]; fbank+LFR+CMVN baked into the graph" },
  "aux_inputs": [
    { "name": "token_num", "shape": [1], "dtype": "int64",
      "note": "text token count + 1 (predictor_bias=1)" }
  ],
  "outputs": [
    { "name": "us_alphas", "shape": [1, null], "dtype": "float32", "note": "T3 = 3 * encoder frames" },
    { "name": "us_peaks",  "shape": [1, null], "dtype": "float32" }
  ],
  "blank_id": 0,
  "tokens_file": "tokens.json",
  "tokens_count": 8404,
  "files": {
    "fp32": { "name": "fa-zh-fp32.onnx", "sha256": "...", "bytes": 0 },
    "int8": { "name": "fa-zh-int8.onnx", "sha256": "...", "bytes": 0 }
  }
}
```

### 4.2 自检向量（设备端会用它对拍）

每个 `vectorN.wav`：**16 kHz / 单声道 / PCM16，时长 2–5 秒**，必须是**真实可对齐的中文语音或歌声**
（不要用纯静音）。可用仓库现有素材（`experiments/audio/fleurs_zh_sample.wav`、
`experiments/audio/synthetic_zh.wav` 均为本地忽略文件），**不得使用第三方歌词文本**。

`vectorN.expected.json` 由 `validate_onnx.py` 生成，至少包含：

```json
{
  "text_raw": "原始输入文本（ASCII/Unicode 均可）",
  "text_normalized": "normalize_cjk 之后、真正喂给模型文本",
  "tokens": ["逐 token 列表（与 token_num - 1 对应）"],
  "token_num": 0,
  "int8": {
    "us_alphas_shape": [1, 0], "us_peaks_shape": [1, 0],
    "us_alphas_mean": 0.0, "us_alphas_std": 0.0,
    "timestamps_ms": [[0, 0]],
    "peak_count": 0
  },
  "fp32": { "同样字段": true },
  "notes": "timestamps_ms 由 postprocess_reference.py 在 FP32 图上生成；INT8 用同一后处理"
}
```

每个向量还应记录 `frontend_features_sha256`（方案 A 时可留空）与 `encoder_shape`，便于 Android 定位偏差。

## 5. 验证要求（全部要跑，数字写进 REPORT.md）

任何一项不达标，**如实报告**并给出诊断，不要粉饰。

1. **前端等价性（方案 A 必须做）**
   - ONNX 图内前端（`input_values` 之后、编码器之前）取出的特征 vs FunASR `extract_fbank`
     （`dither=0`）：`max|Δ|` 目标 **≤ 1e-3**（若实现同构应可达 ≤ 1e-4），报告真实数字。
   - ≥3 段、合计 ≥30 秒真实音频，逐段独立跑。
2. **编码器 + 预测器正确性（FP32 vs PyTorch）**
   - `us_alphas` / `us_peaks` vs PyTorch：`max|Δ| ≤ 5e-4`、余弦相似度 `≥ 0.999`。
   - **端到端时间戳**（经 §2.4 后处理）：逐 token `start/end` 与 Python 参考比较：
     - 完全一致（0 ms）比例 **≥ 95%**
     - `max|Δ|` **≤ 40 ms**（2 个时间戳帧）
     - 峰值数 `fire_place` 与 Python 不一致的样本数 = 0（否则必须逐例分析）
3. **INT8 精度（INT8 vs FP32）**
   - `us_alphas` / `us_peaks` 逐帧余弦：**mean ≥ 0.995，min ≥ 0.97**。
   - 端到端时间戳：`start/end MAE ≤ 20 ms`，`max|Δ| ≤ 60 ms`，
     token 数/峰值数不一致的行 = 0；在 **≥20 条真实歌词行、≥3 段素材**上比较。
   - 若本机没有真实歌曲素材，用仓库现有测试素材并**明确标注素材来源**。
4. **健壮性**
   - 静音（全 0）输入：无 NaN / Inf。
   - 同一输入连跑 3 次：INT8 输出**逐位一致**。
   - 极短输入（400 采样）与 ≥30 秒长输入各跑一次，不报错。
   - **`token_num` 边界**：token 数 1、50、200 各跑一次（CIF 对 `token_num` 敏感，必须覆盖）。
5. **可加载性**
   - `onnx.checker.check_model` 通过。
   - `onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])` 通过，且
     **输入/输出名与形状**与 `manifest.json` 完全一致。
   - 无 external data（单文件自包含）、opset ≤ 17、无 `com.microsoft` 算子（用 `onnx` 遍历确认）。
6. **性能与内存（桌面参考值）**
   - 12 秒窗口、单线程 CPU：推理耗时中位数与 P90（ms）。
   - 进程峰值 RSS（加载 + 连续推理 10 个窗口）。
   - INT8/FP32 文件字节数、sha256。
7. **可复跑**：`validate_onnx.py` 一条命令即可重跑并重新生成 REPORT.md，命令写进报告。

## 6. 参考实现骨架（可用，但必须自己验证）

```python
# android_export/zh/export_model.py（示例，需按实际情况调整与验证）
import torch, torch.nn as nn
from funasr import AutoModel

MODEL_DIR = "models/funasr/fa-zh"

class FaZhOnnx(nn.Module):
    """raw PCM -> (us_alphas, us_peaks); dither must be 0 and deterministic."""
    def __init__(self, auto_model, token_num_default: int):
        super().__init__()
        self.frontend = auto_model.kwargs["frontend"]
        self.net = auto_model.model
        self.frontend.dither = 0.0            # ★ 必须固定，否则导出不确定
        self.token_num_default = token_num_default

    def forward(self, input_values, token_num):
        feats, lens = self.frontend(input_values, [input_values.shape[1]])
        encoder_out, encoder_lens = self.net.encode(feats, lens)
        _, _, us_alphas, us_peaks = self.net.calc_predictor_timestamp(
            encoder_out, encoder_lens, token_num
        )
        return us_alphas, us_peaks

auto = AutoModel(model=MODEL_DIR, device="cpu", disable_update=True,
                 disable_pbar=True, disable_log=True)
m = FaZhOnnx(auto, 10).eval()
torch.onnx.export(
    m, (torch.zeros(1, 16000), torch.tensor([8], dtype=torch.int64)),
    "android_export/zh/fa-zh-fp32.onnx",
    input_names=["input_values", "token_num"],
    output_names=["us_alphas", "us_peaks"],
    dynamic_axes={"input_values": {1: "n"}, "us_alphas": {1: "t3"}, "us_peaks": {1: "t3"}},
    opset_version=17, do_constant_folding=True,
)
# 然后 quantize_dynamic -> android_export/zh/fa-zh-int8.onnx
```

> ⚠ 上面把 `WavFrontend.forward` 直接放进图里。实测该 `forward` 是**逐样本 Python 循环**
> （`for i in range(batch_size)`），batch=1 时可用，但导出时若遇到不支持的算子请改用
> `forward_fbank` + `forward_lfr_cmvn` 自行拼装，并**保持逐值一致**。
> ⚠ `calc_predictor_timestamp` 内部走 `CifPredictorV3.get_upsample_timestamp`（`upsample_type=cnn_blstm`），
> 含 `torch.where` / 动态归一化，导出前请确认算子全部可导出；不可导出时按 §5.2 如实报告。

## 7. 报告模板（REPORT.md）

```markdown
# AutoSync Android ONNX 导出报告（中文 / fa-zh）
## 1. 结论（是否可交付 / 未达标项）
## 2. 交付文件（路径 / 字节 / sha256）
## 3. 契约核对（I/O 名、形状、opset、算子、external data、前端等价性、dither 固定证据）
## 4. 验证数字（按第 5 节 1–7 逐项列表，含素材来源与样本数）
## 5. 一键复跑命令
## 6. 已知限制与风险（FunASR 许可、量化在歌声上的表现、内存、桌面≠设备、token_num 敏感性）
## 7. 未做/做不了的部分（如实说明）
```

## 8. 明确不做的事

- ❌ 不要用启发式/平均分配伪造逐字时间轴
- ❌ 不要试图导出英文或日文链路（各自有独立任务书）
- ❌ 不要为了达标而换模型
- ❌ 不要把权重或测试音频提交进 Git
- ❌ 不要修改 `src/autosync/` 里现有算法与默认行为
- ❌ 不要"顺手修复" `normalize_cjk` 删拉丁字母的行为（那是已知且有意的保守行为）

完成后请回复：**报告路径 + 交付文件路径 + 是否所有验收项通过 + 未通过项清单**。
