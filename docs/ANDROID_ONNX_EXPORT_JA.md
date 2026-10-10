# 任务：把日文 AutoSync 模型（HuBERT 音素 CTC）导出为 Android 可用的 ONNX / INT8

> 这份文档是一份**可直接交给 AI 执行的任务书**，与 `ANDROID_ONNX_EXPORT_PROMPT.md`（英文）、
> `ANDROID_ONNX_EXPORT_ZH.md`（中文）并列。三语言是三份独立任务书，可并行执行。
>
> 执行者需要**真实运行**所有命令、**真实验证**所有数字，并**如实报告**失败项。
> 禁止编造验证结果，禁止用启发式（平均分配）冒充模型输出。

---

## 0. 背景与范围

- 仓库：`/home/lladlam/项目/MeloX-AutoSync`。
- 消费方：MeloX-Android。
- **日文链路是三条链路里设备端成本最高的一条**，因为它是**音素级 CTC**：
  | 环节 | 谁来算 | 难度 |
  |---|---|---|
  | 声学模型（音频 → 48 类音素后验） | **ONNX（本任务交付）** | 低（标准 HuBERT） |
  | **G2P**（日文文本 → 音素序列） | Android 端（需 OpenJTalk 级 G2P） | **最高** |
  | 字符 ↔ 音素归属映射 | Android 端（对照 `ja_pronunciation.py` 移植） | 中高 |
  | CTC Viterbi（blank=0） | Android 端（与 `ctc.py::forced_align` 同算法） | 中 |
- Android 端需要的是**声学模型 + 词表 + G2P/映射/Viterbi 的规格与自检向量**；
  本任务不要求交付解码算法，但**必须交付它的可对拍规格**。
- 模型：`prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`（Apache-2.0，已通过 HF API 核实：
  `cardData.license = apache-2.0`）。**日文当前仍是 experimental 状态**
  （见 `docs/V1_RELEASE_READINESS.md`），导出成功不等于可上线。
- 不要修改 `src/autosync/` 下现有引擎的默认行为；导出与验证脚本请建在交付目录内。

## 1. 环境

```bash
cd /home/lladlam/项目/MeloX-AutoSync
.venv/bin/python -V                       # 已验证：Python 3.12 / torch 2.14.1+cpu / numpy 2.5.3
.venv/bin/python -c "import onnx, onnxruntime; print(onnx.__version__, onnxruntime.__version__)"
# 已验证：onnx 1.23.2 / onnxruntime 1.31.0
```

G2P 与映射参照实现需要 pyopenjtalk：

```bash
uv sync --extra dev --extra shiro --extra onnx       # 已验证：pyopenjtalk 0.3+ 可用
uv run --extra dev --extra shiro --extra onnx python -c "import pyopenjtalk; print(pyopenjtalk.g2p('こんにちは', kana=True))"
```

模型权重**已经在本地**，不要重复下载，离线使用：

```
models/hf/models--prj-beatrice--japanese-hubert-base-phoneme-ctc-v4/snapshots/f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99/
├── model.safetensors
├── config.json                # HubertForCTC, 12 层, hidden 768, vocab_size 48
├── vocab.json                 # 48 音素 token（id 顺序即 token 顺序，见 2.2）
├── preprocessor_config.json   # do_normalize = false  ★
├── tokenizer_config.json
└── special_tokens_map.json
```

## 2. 硬性契约

### 2.1 张量接口

| 项 | 值 |
|---|---|
| 输入名 | `input_values` |
| 输入 dtype / shape | `float32` / `[1, N]`，**N 动态**（任意长度，≥ 400 采样） |
| 输入内容 | **原始 16kHz 单声道 PCM 浮点，范围 [-1, 1]，不做任何归一化** |
| 输出 1 | `logits`，`float32`，`[1, T, 48]` |
| 输出 2 | `log_probs`，`float32`，`[1, T, 48]`（= `log_softmax(logits, -1)`，fp32） |
| 帧率 | 每帧 **20 ms**（320 采样 @16k） |

**与英文的关键差异（已实测核实）**：本模型的 `preprocessor_config.json` 为
`do_normalize = false`，`AutoProcessor` 对 `input_values` **原样透传**
（实测：`max|Δ| = 0.0`，均值/方差保持不变）。HuBERT 前端自带 `feat_extract_norm="group"` 组归一化，
**因此图内不要再烘焙 zero-mean/unit-var 归一化**——那样反而会与 Python 参考不一致。

**帧数公式（已实测核实）**：`T = floor((N - 400) / 320) + 1`

```
N=16000 -> T=49      N=32000 -> T=99      N=8000 -> T=24
```

模型必须以 **eval 模式、float32、无 attention mask** 导出；图内不允许出现 `com.microsoft`
贡献算子，只允许标准 ONNX 算子；**opset = 17**；单文件、无 external data。

### 2.2 词表与特殊 token（48 项，id 顺序即 token 顺序）

```
0:PAD   1:UNK   2:SOS   3:EOS   4:a    5:i    6:u    7:e    8:o    9:I    10:U
11:k    12:g    13:s    14:z   15:t   16:d   17:n   18:h   19:b   20:p
21:m    22:y    23:r    24:w   25:f   26:j   27:v   28:N   29:cl  30:sh
31:ch   32:ts   33:ky  34:gy  35:hy  36:by  37:py  38:my  39:ny  40:ry
41:fy   42:ty   43:dy  44:kw  45:gw  46:pau  47:sil
```

- `blank_id = 0`（CTC blank = PAD）
- `pau = 46`（停顿）、`sil = 47`（静音）
- 注意：这是**音素**词表，**不是字符词表**。文本必须先经 G2P 变音素，否则无法对齐。
- `config.json` 里 `word_delimiter_token_id = 1`（UNK）。**该模型不做词分隔符对齐**，
  不要照搬英文的 `|` 逻辑。

### 2.3 设备端 G2P 规格（必须对拍，最高风险项）

Python 参考链（`src/autosync/lyrics/ja_pronunciation.py` + `models/phoneme_ctc_aligner.py`）：

```
文本 --NFKC/overrides--> run_frontend 分词(surface, pron) --片假名转平假名--> kana
kana --pyopenjtalk.g2p(kana)--> 音素列表 --前后各补一个 "pau"--> 完整音素序列
```

- 参照实现返回的完整序列**首尾各有一个 `pau`**，即
  `["pau", *phonemes, "pau"]`（见 `phoneme_ctc_aligner.py::pyopenjtalk_phonemes`）。
- 音素序列中**不在 48 词表内的音素会被跳过**（`PhonemeCTCAligner._phoneme_ids`）。
- 已实测核对（4 条样例，含英文混入）：G2P 输出音素**全部命中词表，OOV = 0**：

| 文本 | kana | 音素数 | OOV |
|---|---|---:|---|
| みなさん、こんにちは | ミナサン、コンニチワ | 17 | 0 |
| 東京の街を歩いてる | トーキョーノマチヲアルイテル | 21 | 0 |
| シュビデュビ　娑婆Lover | シュビデュビ　シャバエルオーブイイーアール | 29 | 0 |
| きょうはいい天気ですね | キョーワイイテンキデスネ | 18 | 0 |

  （注意：拉丁字母会被逐字母读成片假名，如 `Lover` → `エルオーブイイーアール`。）
- **交付要求**：`g2p_reference.py` 必须为每个测试向量导出
  `text → (surface, kana) 分词 → 完整音素序列（含 pau）→ 逐字符音素归属`，
  作为 Android 端 G2P 的**唯一对拍真值**。
- **必须报告 G2P 覆盖率**：在 ≥50 条真实日文歌词行（或仓库可用素材）上统计
  词表 OOV 比例与「某字符拿不到任何音素」的比例，如实报告。

**设备端选项（必须给出结论，不是建议）**：

- **A（推荐）**：随 App 打包 OpenJTalk（BSD）+ 词典（CC-BY-SA，注意署名义务）经 JNI 调用，
  与 pyopenjtalk 同源，可达一致性。必须报告 ABI/体积/许可结论。
- **B**：纯 Java/Kotlin 自行实现 G2P。必须用 `g2p_reference.py` 的真值逐条对拍，
  报告字符级一致率；**一致率不达标就必须放弃 B**。
- **C**：设备端无法提供 G2P → **日文链路本轮不可交付**，必须在报告中明确写出
  「JA 只能保持实验状态 / 或仅支持预生成音素」，**不得假装可用**。

### 2.4 字符 ↔ 音素归属与 Viterbi（规格，Android 端实现）

1. **CTC Viterbi**：与 `src/autosync/alignment/ctc.py::forced_align(logits, targets, blank=0)`
   **同算法**，`targets` = 词表内音素 id（去掉 OOV 后的序列）。不可行时必须显式报不可行，
   不得回退到平均分配。
2. **音素 → 字符**：照 `phoneme_ctc_aligner.py::map_spans_to_char_units`
   （`locate_inner_offset` 先把逐字符音素序列定位到含 `pau` 的完整序列内部，**避免首字符吞掉
   前导 `pau` 的时间**；无音素的字符给零宽区间并标 `uncertain`）。
3. **帧 → 时间**：`start = start_frame × 0.02`，`end = (end_frame + 1) × 0.02`。
4. 逐字符结果必须能**无损重建原文**（字符一个都不能丢）。

## 3. 量化要求

- 用 **ONNX Runtime 动态量化**（`onnxruntime.quantization.quantize_dynamic`），
  `weight_type=QuantType.QInt8`，默认只量化 MatMul/Gemm；卷积前端与 LayerNorm 保持 fp32。
- 量化后**必须仍是单文件、标准算子、CPU ExecutionProvider 可加载**。
- INT8 文件体积目标 **≤ 120 MB**；若超出，报告实际值与原因，不要偷偷换模型。

## 4. 交付物（放到 `android_export/ja/`）

```
android_export/ja/
├── hubert-ja-phoneme-ctc-fp32.onnx     # 仅用于验证对比
├── hubert-ja-phoneme-ctc-int8.onnx     # ★ Android 端实际使用
├── manifest.json                        # 契约清单，见 4.1
├── test_vectors/                        # 设备端自检向量，见 4.2（gitignore）
│   ├── vector1.wav  vector1.expected.json
│   ├── vector2.wav  vector2.expected.json
│   └── vector3.wav  vector3.expected.json
├── export_model.py                      # 可复跑导出 + 量化
├── validate_onnx.py                     # 生成 REPORT.md 的脚本，可复跑
├── g2p_reference.py                     # pyopenjtalk G2P + 映射真值导出（设备端对拍）
└── REPORT.md                            # 验证报告，见 7
```

> 大文件不要 `git add`（`.gitignore` 已排除 `android_export/**/*.onnx` 与
> `android_export/**/test_vectors/`；脚本 / manifest / REPORT.md 属于我们自己的产出，可以提交）。
> 交付时报告**绝对路径 + 字节数 + sha256**。

### 4.1 `manifest.json` 格式（字段名不要改）

```json
{
  "schema": "melox.autosync.onnx.1",
  "language": "ja",
  "source_model": "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4",
  "license": "Apache-2.0",
  "sample_rate": 16000,
  "frame_hop_samples": 320,
  "frame_hop_seconds": 0.02,
  "input": { "name": "input_values", "shape": [1, null], "dtype": "float32",
             "preprocessing": "raw 16k mono PCM in [-1,1]; NO external normalization (preprocessor do_normalize=false; HuBERT uses internal group norm)" },
  "outputs": [
    { "name": "logits",    "shape": [1, null, 48], "dtype": "float32" },
    { "name": "log_probs", "shape": [1, null, 48], "dtype": "float32" }
  ],
  "blank_id": 0,
  "phoneme_tokens": ["PAD", "UNK", "SOS", "EOS", "a", "... 完整 48 项 ..."],
  "g2p": {
    "reference": "pyopenjtalk (OpenJTalk) via ja_pronunciation.py",
    "wrap_pau": true,
    "device_option": "A|B|C",
    "device_option_evidence": "见 REPORT.md 第 4 节"
  },
  "files": {
    "fp32": { "name": "hubert-ja-phoneme-ctc-fp32.onnx", "sha256": "...", "bytes": 0 },
    "int8": { "name": "hubert-ja-phoneme-ctc-int8.onnx", "sha256": "...", "bytes": 0 }
  }
}
```

### 4.2 自检向量（设备端会用它对拍）

每个 `vectorN.wav`：**16 kHz / 单声道 / PCM16，时长 2–5 秒**，必须是**真实可对齐的日文语音或歌声**
（不要用纯静音）。可用仓库现有素材（`experiments/audio/fleurs_ja_sample.wav`、
`experiments/audio/synthetic_ja.wav` 均为本地忽略文件），**不得使用第三方歌词文本**。

`vectorN.expected.json` 由 `validate_onnx.py` 生成，至少包含：

```json
{
  "text": "日文歌词短句（原始文本）",
  "phonemes": ["pau", "k", "o", "..."],
  "phoneme_ids": [46, 11, 8],
  "char_phoneme_spans": [{ "char": "こ", "phoneme_index_start": 1, "phoneme_index_end": 2 }],
  "int8": {
    "log_probs_shape": [1, 0, 48],
    "log_probs_mean": 0.0, "log_probs_std": 0.0,
    "argmax_run_lengths_first20": [0, 0],
    "blank_ratio": 0.0,
    "align_spans": [ { "token_id": 0, "start_frame": 0, "end_frame": 0 } ],
    "char_units": [ { "char": "こ", "start": 0.0, "end": 0.0, "uncertain": false } ]
  },
  "fp32": { "同样字段": true },
  "notes": "phoneme_ids 按 manifest.phoneme_tokens；align_spans 用 forced_align(blank=0) 生成；char_units 由 map_spans_to_char_units 生成"
}
```

> `char_units` 是三条链路里**唯一**需要设备端实现「音素→字符」的向量，务必完整、可直接对拍。

## 5. 验证要求（全部要跑，数字写进 REPORT.md）

任何一项不达标，**如实报告**并给出诊断，不要粉饰。

1. **导出正确性（FP32 vs PyTorch）**
   同一批输入（≥3 段、合计 ≥30 秒真实音频，逐段独立跑）：
   - `max|Δlogits| ≤ 5e-4`，`mean|Δlogits| ≤ 5e-6`
   - argmax 一致率 100%（仅统计 PyTorch 端 max softmax > 0.5 的帧）
   - **归一化核对**：确认 ONNX 图**没有**额外归一化（`input_values` 逐值透传，
     `max|Δ| = 0.0` 对 `AutoProcessor` 输出）
2. **INT8 精度（INT8 vs FP32）**
   - `log_probs` 逐帧余弦相似度：**mean ≥ 0.995，min ≥ 0.97**
   - 端到端强制对齐（用仓库自己的 `forced_align`，blank=0）：
     在 **≥20 条真实歌词行、≥3 段素材**上比较逐字符起止：
     - 起始时间 **MAE ≤ 30 ms**
     - 覆盖率变化 **≤ 2 个百分点**
     - 可行性（feasible）判定翻转 **≤ 1 行**
   - 若本机没有真实歌曲素材，用仓库现有测试素材并**明确标注素材来源**。
3. **G2P 与映射**
   - `g2p_reference.py` 必须能独立复跑，并与 `ja_pronunciation.py` 的
     `map()` 输出**逐字段一致**（分词、kana、音素序列、逐字符归属）。
   - 报告 ≥50 行的 OOV 比例与「无音素字符」比例。
   - 报告设备端选项 A/B/C 的结论与证据；若选 B，报告字符级一致率。
4. **健壮性**
   - 静音（全 0）输入：无 NaN / Inf
   - 同一输入连跑 3 次：INT8 输出**逐位一致**
   - 极短输入（400 采样）与 ≥30 秒长输入各跑一次，不报错
5. **可加载性**
   - `onnx.checker.check_model` 通过
   - `onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])` 通过，且
     **输入/输出名与形状**与 `manifest.json` 完全一致
   - 无 external data（单文件自包含）、opset ≤ 17、无 `com.microsoft` 算子（用 `onnx` 遍历确认）
6. **性能与内存（桌面参考值）**
   - 12 秒窗口、单线程 CPU：推理耗时中位数与 P90（ms）
   - 进程峰值 RSS（加载 + 连续推理 10 个窗口）
   - INT8/FP32 文件字节数、sha256
7. **可复跑**：`validate_onnx.py` 一条命令即可重跑并重新生成 REPORT.md，命令写进报告。

## 6. 参考实现骨架（可用，但必须自己验证）

```python
# android_export/ja/export_model.py（示例，需按实际情况调整与验证）
import torch, torch.nn as nn
from transformers import HubertForCTC

SNAPSHOT = ("models/hf/models--prj-beatrice--japanese-hubert-base-phoneme-ctc-v4/"
            "snapshots/f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99")

class HubertCTC(nn.Module):
    """raw PCM -> (logits, log_probs); NO external normalization (do_normalize=false)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_values):                      # [1, N] raw PCM
        logits = self.model(input_values).logits          # [1, T, 48]
        return logits, torch.log_softmax(logits.float(), dim=-1)

model = HubertForCTC.from_pretrained(SNAPSHOT).eval()
wrapper = HubertCTC(model).eval()
torch.onnx.export(
    wrapper, torch.zeros(1, 16000),
    "android_export/ja/hubert-ja-phoneme-ctc-fp32.onnx",
    input_names=["input_values"], output_names=["logits", "log_probs"],
    dynamic_axes={"input_values": {1: "n"}, "logits": {1: "t"}, "log_probs": {1: "t"}},
    opset_version=17, do_constant_folding=True,
)
# 然后 quantize_dynamic -> android_export/ja/hubert-ja-phoneme-ctc-int8.onnx
```

> ⚠ G2P/映射真值请**直接调用仓库实现**（`JapanesePronunciationMapper` +
> `PhonemeCTCAligner._phoneme_ids` + `map_spans_to_char_units`），不要另写一份：
> 那是 Android 端要对拍的**同一份规格**。若需要脱离 funasr/引擎运行，请把它们当纯函数调用，
> 不要改动 `src/autosync/`。

## 7. 报告模板（REPORT.md）

```markdown
# AutoSync Android ONNX 导出报告（日文 / HuBERT 音素 CTC）
## 1. 结论（是否可交付 / 未达标项 / 设备端 G2P 选项 A|B|C 结论）
## 2. 交付文件（路径 / 字节 / sha256）
## 3. 契约核对（I/O 名、形状、opset、算子、external data、无额外归一化的证据）
## 4. 验证数字（按第 5 节 1–7 逐项列表，含素材来源与样本数）
   - 含 G2P OOV 率、无音素字符比例、映射一致性
## 5. 一键复跑命令
## 6. 已知限制与风险（G2P 移植成本、量化在歌声上的表现、内存、桌面≠设备、JA 仍属 experimental）
## 7. 未做/做不了的部分（如实说明）
```

## 8. 明确不做的事

- ❌ 不要用启发式/平均分配伪造逐字时间轴
- ❌ 不要试图导出英文或中文链路（各自有独立任务书）
- ❌ 不要为了达标而换模型
- ❌ 不要把权重或测试音频提交进 Git
- ❌ 不要修改 `src/autosync/` 里现有算法与默认行为
- ❌ 不要在设备端 G2P 未验证的情况下宣称日文可用

完成后请回复：**报告路径 + 交付文件路径 + 是否所有验收项通过 + 未通过项清单 + 设备端 G2P 选项结论**。
