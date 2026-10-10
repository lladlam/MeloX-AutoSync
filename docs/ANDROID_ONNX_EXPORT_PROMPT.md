# 任务：把英文 AutoSync 模型导出为 Android 可用的 ONNX / INT8

> 这份文档是一份**可直接交给 AI 执行的任务书**。
> 目标：为 MeloX-Android 的「AutoSync 自动逐字歌词」功能，交付一个**能在 Android 端
> 用 ONNX Runtime 本地推理**的英文强制对齐模型。
>
> 执行者需要**真实运行**所有命令、**真实验证**所有数字，并**如实报告**失败项。
> 禁止编造验证结果，禁止用启发式（平均分配）冒充模型输出。

---

## 0. 背景与范围

- 仓库：`/home/lladlam/项目/MeloX-AutoSync`（Python 端 AutoSync 引擎，已完成 V1.0）。
- 消费方：MeloX-Android。Android 端会自己实现 **CTC Viterbi 强制对齐**
  （与 `src/autosync/alignment/ctc.py::forced_align` 同算法、blank=0）与**分词/分词组**，
  因此本次只要求交付**声学模型 + 词表 + 契约清单**，不要求交付对齐算法。
- **本文档只覆盖英文**（`facebook/wav2vec2-base-960h`）。三语言是**三份独立任务书**，可并行执行：
  - 中文 → `ANDROID_ONNX_EXPORT_ZH.md`（fa-zh 是 MonotonicAligner，非 CTC，前端与解码契约不同）
  - 日文 → `ANDROID_ONNX_EXPORT_JA.md`（HuBERT 音素 CTC，需设备端 G2P，成本最高）
  - 未覆盖语言：MeloX 端**保守回退**到原始逐行歌词，不生成逐字。
- 英文这份是整个 Android 管线的「跑通基线」：CTC Viterbi 与设置/UI 接入都先在这里验证，
  之后中文只需替换模型与解码器。
- 不要修改 `src/autosync/` 下现有引擎的默认行为；验证脚本请新建在交付目录内。

## 1. 环境

```bash
cd /home/lladlam/项目/MeloX-AutoSync
.venv/bin/python -V            # 已有：torch 2.14.1+cpu / transformers 5.19.0 / numpy 2.5.3
.venv/bin/python -c "import onnx, onnxruntime"   # 当前缺失，需要你先安装
```

需要补装（用 uv 或 pip 均可，写进报告）：

```bash
.venv/bin/python -m pip install "onnx>=1.17" "onnxruntime>=1.20"
```

模型权重**已经下载在本地**，不要重复下载，离线使用即可：

```
models/hf/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156/
  ├── model.safetensors
  ├── config.json
  ├── vocab.json
  └── preprocessor_config.json
```

## 2. 硬性契约（Android 端已按此实现，不要改）

### 2.1 张量接口

| 项 | 值 |
|---|---|
| 输入名 | `input_values` |
| 输入 dtype / shape | `float32` / `[1, N]`，**N 动态**（任意长度，≥ 400 采样） |
| 输入内容 | **原始 16kHz 单声道 PCM 浮点，范围 [-1, 1]，不做任何归一化** |
| 输出 1 | `logits`，`float32`，`[1, T, 32]` |
| 输出 2 | `log_probs`，`float32`，`[1, T, 32]`（= `log_softmax(logits, -1)`，fp32） |
| 帧率 | 每帧 **20 ms**（320 采样 @16k）；`T` 由模型自身步长决定，Android 端从输出形状读取 |

**归一化必须烘焙进图内**：导出模型内部要对输入做与 `Wav2Vec2FeatureExtractor`
（`do_normalize=true`）**逐值一致**的 zero-mean unit-variance 归一化，
即 `(x - mean(x)) / sqrt(var(x) + 1e-7)`（对整段输入求均值/方差，population variance）。

> 为什么要烘焙：Android 端只做「解码音频 → 喂原始 PCM → 拿 log_probs → Viterbi」，
> 少一个环节就少一处漂移。请务必用测试波形**实测比对**你的烘焙实现与
> `AutoProcessor` 输出一致（逐步比对归一化后的张量，报告 max|Δ|），不要只凭公式。

模型必须以 **eval 模式、float32、无 attention mask** 导出；图内不允许出现
`com.microsoft` 贡献算子，只允许标准 ONNX 算子；**opset = 17**；单文件、无 external data。

### 2.2 词表与特殊 token

`vocab.json` 共 32 项，**id 顺序即 token 顺序**：

```
0:<pad>  1:<s>  2:</s>  3:<unk>  4:|  5:E  6:T  ...  31:Z
```

- `blank_id = 0`（CTC blank）
- 词分隔符 id = 4（token 文本为 `|`）
- 字母为**大写** A–Z，另有 `'`；空格在 tokenizer 里映射为 `|`

### 2.3 文本规范化（Android 端会照此实现，供你核对）

对应 `src/autosync/lyrics/normalize.py::normalize_en`：
NFKC → 弯引号 `’ ‘` 转 `'` → 删除 `[^A-Za-z0-9' ]`（替换为空格）→ 折叠空白 → **转大写**。
分词后按 `|`/空格切词做 CTC 对齐，再按 `[A-Za-z0-9']+` 还原到原词。
（你不需要实现它，但请确认词表能覆盖它产生的所有字符。）

## 3. 量化要求

- 用 **ONNX Runtime 动态量化**（`onnxruntime.quantization.quantize_dynamic`），
  `weight_type=QuantType.QInt8`，默认只量化 MatMul/Gemm；卷积前端与 LayerNorm 保持 fp32。
- 量化后**必须仍是单文件、标准算子、CPU ExecutionProvider 可加载**。
- INT8 文件体积目标 **≤ 120 MB**；若超出，报告实际值与原因，不要偷偷换模型。

## 4. 交付物（放到 `android_export/en/`）

```
android_export/en/
├── wav2vec2-base-960h-fp32.onnx       # 仅用于验证对比
├── wav2vec2-base-960h-int8.onnx       # ★ Android 端实际使用
├── manifest.json                       # 契约清单，见 4.1
├── test_vectors/                       # 设备端自检向量，见 4.2
│   ├── vector1.wav  vector1.expected.json
│   ├── vector2.wav  vector2.expected.json
│   └── vector3.wav  vector3.expected.json
├── validate_onnx.py                    # 生成 REPORT.md 的脚本，可复跑
└── REPORT.md                           # 验证报告，见 5
```

> 不要把这些大文件 `git add` 进仓库（保持在磁盘上即可，可用 `.gitignore` 排除
> `android_export/`）。交付时报告**绝对路径 + 字节数 + sha256**。

### 4.1 `manifest.json` 格式（字段名不要改）

```json
{
  "schema": "melox.autosync.onnx.1",
  "language": "en",
  "source_model": "facebook/wav2vec2-base-960h",
  "license": "Apache-2.0",
  "sample_rate": 16000,
  "frame_hop_samples": 320,
  "frame_hop_seconds": 0.02,
  "input": { "name": "input_values", "shape": [1, null], "dtype": "float32",
             "preprocessing": "raw 16k mono PCM in [-1,1]; zero-mean unit-var normalization is baked into the graph" },
  "outputs": [
    { "name": "logits",    "shape": [1, null, 32], "dtype": "float32" },
    { "name": "log_probs", "shape": [1, null, 32], "dtype": "float32" }
  ],
  "blank_id": 0,
  "word_delimiter_id": 4,
  "tokens": ["<pad>", "<s>", "</s>", "<unk>", "|", "E", "... 完整 32 项 ..."],
  "files": {
    "fp32": { "name": "wav2vec2-base-960h-fp32.onnx", "sha256": "...", "bytes": 0 },
    "int8": { "name": "wav2vec2-base-960h-int8.onnx", "sha256": "...", "bytes": 0 }
  }
}
```

### 4.2 自检向量（设备端会用它对拍）

每个 `vectorN.wav`：**16 kHz / 单声道 / PCM16，时长 2–5 秒**，必须是**真实可对齐的语音或歌声片段**
（不要用纯静音；可用仓库现有测试素材，注意不要提交第三方歌词文本）。
`vectorN.expected.json` 由 `validate_onnx.py` 生成，至少包含：

```json
{
  "text": "用于对齐的短句（ASCII，大写或小写均可）",
  "int8": {
    "log_probs_shape": [1, 0, 32],
    "log_probs_mean": 0.0, "log_probs_std": 0.0,
    "argmax_run_lengths_first20": [0, 0],
    "blank_ratio": 0.0,
    "align_spans": [ { "token_id": 0, "start_frame": 0, "end_frame": 0 } ]
  },
  "fp32": { "同样字段": true },
  "notes": "token_id -> 字符见 manifest.tokens；align_spans 用 src/autosync/alignment/ctc.py::forced_align(blank=0) 生成"
}
```

## 5. 验证要求（全部要跑，数字写进 REPORT.md）

必须先跑通，再交付。任何一项不达标，**如实报告**并给出诊断，不要粉饰。

1. **导出正确性（FP32 vs PyTorch）**
   同一批输入（≥3 段、合计 ≥30 秒真实音频，逐段独立跑）：
   - `max|Δlogits|` ≤ **5e-4**，`mean|Δlogits|` ≤ **5e-6**
   - argmax 一致率 100%（仅统计 PyTorch 端 max softmax > 0.5 的帧）
   - 烘焙归一化 vs `AutoProcessor` 归一化：`max|Δ|` ≤ **1e-5**

2. **INT8 精度（INT8 vs FP32）**
   - `log_probs` 逐帧余弦相似度：**mean ≥ 0.995，min ≥ 0.97**
   - 端到端强制对齐（用仓库自己的 `forced_align`，blank=0）：
     在 **≥20 条真实歌词行、≥3 首歌**上比较逐字/逐词起止时间：
     - 起始时间 **MAE ≤ 30 ms**
     - 覆盖率变化 **≤ 2 个百分点**
     - 可行性（feasible）判定翻转 **≤ 1 行**
   - 若本机没有真实歌曲素材，用仓库现有测试素材并在报告中**明确标注素材来源**。

3. **健壮性**
   - 静音（全 0）输入：无 NaN / Inf
   - 同一输入连跑 3 次：INT8 输出**逐位一致**（动态量化是确定性的）
   - 极短输入（400 采样）与 ≥30 秒长输入各跑一次，不报错

4. **可加载性**
   - `onnx.checker.check_model` 通过
   - `onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])` 通过，且
     **输入/输出名与形状**与 `manifest.json` 完全一致
   - 无 external data（单文件自包含）、opset ≤ 17、无 `com.microsoft` 算子（用 `onnx` 遍历确认）

5. **性能与内存（桌面参考值）**
   - 12 秒窗口、单线程 CPU：推理耗时中位数与 P90（ms）
   - 进程峰值 RSS（加载 + 连续推理 10 个窗口）
   - INT8/FP32 文件字节数、sha256

6. **可复跑**：`validate_onnx.py` 一条命令即可重跑并重新生成 REPORT.md，命令写进报告。

## 6. 参考实现骨架（可用，但必须自己验证）

```python
# android_export/en/export_model.py（示例，需按你的实际情况调整与验证）
import torch, torch.nn as nn
from transformers import Wav2Vec2ForCTC

class Wav2Vec2CTCNorm(nn.Module):
    """bakes Wav2Vec2FeatureExtractor(do_normalize=True) into the graph"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_values):                      # [1, N] raw PCM
        mean = input_values.mean(dim=1, keepdim=True)
        var = input_values.var(dim=1, keepdim=True, unbiased=False)
        x = (input_values - mean) / torch.sqrt(var + 1e-7)
        logits = self.model(x).logits                      # [1, T, 32]
        return logits, torch.log_softmax(logits.float(), dim=-1)

model = Wav2Vec2ForCTC.from_pretrained(SNAPSHOT).eval()
wrapper = Wav2Vec2CTCNorm(model).eval()
torch.onnx.export(
    wrapper, torch.zeros(1, 16000),
    "wav2vec2-base-960h-fp32.onnx",
    input_names=["input_values"], output_names=["logits", "log_probs"],
    dynamic_axes={"input_values": {1: "n"}, "logits": {1: "t"}, "log_probs": {1: "t"}},
    opset_version=17, do_constant_folding=True,
)
# 然后 quantize_dynamic -> wav2vec2-base-960h-int8.onnx
```

> ⚠ `input_values.var(unbiased=False)` 与 `Wav2Vec2FeatureExtractor` 内部使用的
> numpy `.var()`（population）一致；但**务必实测比对**（见 5.1），不要只凭公式推断。

## 7. 报告模板（REPORT.md）

```markdown
# AutoSync Android ONNX 导出报告
## 1. 结论（是否可交付 / 未达标项）
## 2. 交付文件（路径 / 字节 / sha256）
## 3. 契约核对（I/O 名、形状、opset、算子、external data、归一化比对）
## 4. 验证数字（按第 5 节 1–5 逐项列表，含素材来源与样本数）
## 5. 一键复跑命令
## 6. 已知限制与风险（尤其：量化在歌声上的实际表现、内存、桌面≠设备）
## 7. 未做/做不了的部分（如实说明）
```

## 8. 明确不做的事

- ❌ 不要用启发式/平均分配伪造逐字时间轴
- ❌ 不要尝试导出中文 `fa-zh` 或日文链路（本轮范围外）
- ❌ 不要为了达标而换大模型（仓库已验证 large 更差）
- ❌ 不要把权重提交进 Git
- ❌ 不要修改 `src/autosync/` 里现有算法与默认行为

完成后请回复：**报告路径 + 交付文件路径 + 是否所有验收项通过 + 未通过项清单**。
