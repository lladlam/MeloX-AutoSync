# 日语歌声模型对比（JAPANESE_MODEL_COMPARISON）

> 评测协议：V0.3 的 **`lrc_only` 盲评**（只用音频 + 歌词文本 + LRC 行起始；Ground Truth 仅用于评分），
> 同一批 30 段 GTSinger 日语歌声、同一评分器（`benchmarks/metrics.py`）。
> 复现：`benchmarks/run_singing_eval.sh`（新增引擎见下）。原始报告 `experiments/reports/singing_ja.json`。

## 0. 结论速览

| 引擎 | 覆盖率 | 失败率 | sMAE | s中位 | sP90 | eMAE | e中位 | eP90 | 不确定 | 加载 | 体积 | 许可 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heuristic（基线） | 0.997 | 0% | 1.556 | – | 3.822 | 1.615 | – | 3.950 | 0.00 | – | – | – |
| **reazon wav2vec2-base**（V0.1 起） | 0.528 | **46.7%** | 1.248 | 0.386 | 4.121 | 1.291 | 0.457 | 4.121 | 0.35 | ~5s | 369 MB | Apache-2.0 |
| **pyshiro v2**（A） | **1.000** | **0%** | **0.628** | **0.118** | 2.184 | **0.626** | **0.135** | 2.067 | 0.01 | ~0.05s | **0.49 MB** | GPLv3 代码 / 权重许可不明 |
| pyshiro v1（A） | 1.000 | 0% | 0.628 | 0.131 | 1.988 | 0.647 | 0.152 | 2.049 | 0.01 | ~0.05s | 0.13 MB | 同上 |
| **HuBERT 音素 CTC**（C） | 1.000 | 0% | 0.643 | 0.340 | **1.399** | 0.748 | 0.431 | **1.763** | 0.01 | 50s | 361 MB | **Apache-2.0** |
| TIFA（B） | 未完成端到端（磁盘配额，见 §4） | | | | | | | | | | 151 MB | MIT（代码） |

（秒数为 30 段 × 2 模式的合计推理时间；`lrc_only`。）

**相比原 ReazonSpeech 模型**：
- 失败率 46.7% → **0%**；覆盖率 0.528 → **1.000**；起始中位误差 386 ms → **118 ms**。
- 但 **MAE 未达 200 ms 目标**（0.628 s），主要由 JA-Tenor-1 的长音段落（さんぽ）拉动。

## 1. A. pyshiro（SHIRO / HSMM）— 本轮最优

- 仓库：https://github.com/wavtechyukky/pyshiro；**代码 GPLv3**。
  权重在仓库 `checkpoint/`（v1/v2）与 submodule `intunist/SHIRO-Models-Japanese`，**无明确权重许可**。
- 面向日语**歌声**训练（v1：多库合计 19.4h；v2：`nmix 4`，未见过歌手的人审音素边界平均误差 2.05 帧 ≈ 10 ms）。
- 采用两遍 HMM→HSMM 对齐，帧移 5 ms；我们只需**音素序列**（由 §G2P 提供），无需训练。
- 在本盲评中：**覆盖率 100%、零失败、中位 118 ms**，是三者中 MAE/中位最优。
- v1 vs v2：v2 中位略优（0.118 vs 0.131），v1 的 P90 略好（1.99 vs 2.18）——v1 对 `br`（气息）更稳。

## 2. C. HuBERT 音素 CTC — 许可干净、P90 最优

- 模型：`prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`（**Apache-2.0**，~361 MB，48 音素）。
  训练数据为 ReazonSpeech（**语音**），但以**音素 CTC** 形式复用我们自研的 Viterbi 强制对齐。
- 我们只把 G2P 得到的**已知音素序列**做强制对齐（不做自由识别），因此比自由转写稳健。
- 结果：覆盖 100%、零失败，**P90 最优**（start 1.40 s），但中位（340 ms）与 MAE（0.643）劣于 pyshiro。
- 价值：许可友好 + 可与英文共用 CTC/Viterbi 代码路径；适合作为**第二方案/交叉校验**。

## 3. 分歌手 / 分曲目失败情况

| 引擎 | JA-Soprano-1（女高） | JA-Tenor-1（男高） |
|---|---|---|
| pyshiro v2 | sMAE 0.436，中位 0.085 | sMAE **1.476**，中位 0.935 |
| HuBERT CTC | sMAE 0.479，中位 0.301 | sMAE 1.363，中位 0.689 |

分曲目（pyshiro）：`さんぽ` 1.476 ≫ `first love` 0.702 > `时雨` 0.368 > `heartful song` 0.363。

**失败样本（保留）**：`さんぽ/Vibrato/0000`，参考把长音记在 `こ`（1.03–3.21s），
pyshiro 把长音落到 `う`（预测 2.26–4.69s）→ 单个长拖音的**モーラ归属**错误。
这与 pyshiro 模型卡「对男声（男高/男低）表现较差」的说明一致，属**长音/男声**局限，
**不是**窗口或评测口径问题（`lrc_only` 与 `oracle` 结果几乎一致）。

## 4. B. TIFA — 资源与部署评估

- 代码 **MIT**（openvpi），模型 `TIFA-1.0-ST`（GitHub release，151 MB zip），多语言（zh/ja/en），
  G2P 走 `g2pflow`（日语依赖 UniDic 词典）。
- 具备**完整 ONNX 部署管线**（`spectrogram/prepare/model/score/select.onnx`）→ 对 Android 有利。
- 依赖较重（`lightning`、`dask`、`onnx`、`praat-parselmouth`、`textgrid`、`numba` …）。

**实测状态（未完成端到端）**：本轮在本环境尝试运行，步骤与结果如下：

1. 在独立环境 `/tmp/opencode/tifa_env` 安装 `torch`(CPU) + `requirements.txt` → **成功**。
2. 下载并解压 `TIFA-1.0-ST`（150.8 MB，含 `model.pt`/`config.yaml`/`vocabulary.json`）→ **成功**。
3. 首次推理：`G2P failed: japanese-mecab requires optional dependencies` → 安装 `g2pflow[ja]`，
   开始下载 **UniDic v3.1.0（约 526 MB）**。
4. 解压 UniDic 时失败：`G2P failed: [Errno 122] Disk quota exceeded`（本机可用磁盘约 4.6 GB，
   不足以同时容纳独立环境 + UniDic + 已缓存的其它模型权重）。

因此 **TIFA 的日语歌声对齐未能给出实测精度**。这是**客观资源限制**，非评测口径问题；
已保留失败证据（命令与报错）。后续可在更充裕的机器上补做。

## 5. 评测说明与限制

- **同一测试集/评分器**，未使用人工边界辅助推理（`lrc_only`）。
- 若未来用歌声数据微调，必须**按歌手/歌曲隔离**；本评测集只用于评测。
- GTSinger 标签核查：在模型间分歧最大的长音片段（さんぽ），参考把长音记在前一モーラ、
  模型记在后一モーラ。我们**遵循 GTSinger 标注**，并把这些分歧作为失败样本保留，不修改评测口径。
- 权重许可：pyshiro 权重许可不明 → **不得打包发布**；HuBERT/TIFA 许可清晰（Apache-2.0 / MIT）。
