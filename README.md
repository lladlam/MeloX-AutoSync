# MeloX-AutoSync

**MeloX-AutoSync** 是一个**完全离线**的多语言（中文 / 日文 / 英文）**歌词逐字对齐引擎**。

给定：

- 一首歌的音频（`song.wav` / `song.mp3` / ...）
- 对应的 **LRC 逐行歌词**（`song.lrc`）

它输出：尽可能准确的**逐字 / 逐词时间轴 JSON**，供 MeloX 播放器做卡拉 OK 式高亮。

> 本项目只做 **forced alignment（强制对齐）**，不做从零开始的语音转写。
> 已有歌词文本是强先验，我们用它把音频的声学帧对齐到歌词单元上。

## 设计原则

1. **完全离线**：模型推理不联网、不调用云端语音 API。
2. **按需加载**：三种语言各自独立模型，只有被请求的语言才会下载 / 加载。
3. **不伪造**：真实推理 vs 启发式估算（heuristic）在输出中显式区分；失败就报告失败。
4. **小模型优先**：CPU 可跑，可量化、可移植到 Android（见 `docs/ANDROID_FEASIBILITY.md`）。

## 安装

```bash
# 需要 uv (https://docs.astral.sh/uv/)
uv sync                 # 基础依赖：英文 / 日文 wav2vec2 链路
uv sync --extra funasr  # 可选：中文 FunASR fa-zh 链路
uv sync --extra dev     # 开发 / 测试
```

模型权重会被下载到 `models/`（已 gitignore，不会提交到 Git）。

## 使用

```bash
# 英文：单词级对齐
uv run autosync align --audio song.wav --lyrics song.lrc --language en --output result.json

# 日文：发音单元 -> 原文字符映射
uv run autosync align --audio song.wav --lyrics song.lrc --language ja --output result.json

# 中文：汉字级时间
uv run autosync align --audio song.wav --lyrics song.lrc --language zh --output result.json

# 不加载任何模型，仅用启发式基线（结果标记为 heuristic）
uv run autosync align --audio song.wav --lyrics song.lrc --language en --engine heuristic

# V0.2：能量感知的结束边界细化
uv run autosync align --audio song.wav --lyrics song.lrc --language en --output result.json --refine energy
```

## 项目结构

```
src/autosync/audio/       音频加载、裁剪、重采样
src/autosync/lyrics/      LRC 解析与文本规范化
src/autosync/models/      独立语言模型适配器（en / ja / zh）
src/autosync/alignment/   CTC Viterbi 强制对齐、分段、启发式基线
src/autosync/export/      时间轴 JSON 导出
benchmarks/               模型实验与误差评估
tests/                    自动化测试
docs/                     调研 / 实验日志 / 技术报告
```

## 状态

V0.2 加入了**歌声评测**、CTC 塌缩/不可行检测、日文原文映射与能量边界细化。
请阅读 `docs/EXPERIMENT_RESULTS.md` 了解**已经真实运行验证**的内容，
`docs/SINGING_EVAL.md` 了解歌声数据集与许可，
以及 `docs/KNOWN_LIMITATIONS.md` 了解**尚未完成 / 存在风险**的部分。

### 新增（V0.2）

- `--refine energy`：能量感知的结束边界细化（改善 end MAE）。
- CTC Viterbi 现在会**检测并报告**帧数不足、塌缩、非单调等异常，而不是输出垃圾。
- 日文多字符 token 保留 `uncertain` 标记，并校验原文可重建。
- 新评测：`benchmarks/singing_eval.py`、`benchmarks/accompaniment_eval.py`（覆盖率/漏匹配/错误率/失败样本）。

### 新增（V0.3）— 修正评测 + 歌声模型研究

- **评测泄漏修复**：`singing_eval.py` 新增 **`lrc_only` 盲评**模式（只用音频+歌词+LRC 行起始），
  `oracle` 模式仅作研究，二者分开报告；报告含按歌手/歌曲/技巧拆分与盲评-oracle 差距。
- 修复 `accompaniment_eval.py` 的引擎混合汇总缺陷。
- 逐段诊断工具 `benchmarks/diagnosis.py`（CTC CER、blank/塌缩分布、窗口对比、有符号起始偏差）。
- 研究结论：**日语语音模型在歌声上不可用**（加大模型反而更差）；**英文 base 已够用、large 无益**；
  能量起始回吸与能量分段重对齐**均被否决**。详见 `docs/EXPERIMENT_RESULTS.md` 与 `docs/ACCEPTANCE.md`。

### 新增（V0.4）— 日语歌声模型替换

- 新增 **pyshiro（SHIRO/HSMM 歌声对齐）** 适配器：覆盖率 0.528→**1.000**、失败率 46.7%→**0%**、
  起始中位 386ms→**118ms**（`--extra shiro`）。
- 新增 **HuBERT 音素 CTC** 适配器（Apache-2.0，复用自研 CTC Viterbi）。
- 新增 **`JapanesePronunciationMapper`**：汉字/假名 G2P、促音/长音/拨音/拗音、音素↔原文映射、
  读音覆盖、不确定性、无损重建。
- 许可结论：pyshiro 代码 GPLv3、权重许可不明（不可打包）；HuBERT/TIFA 许可清晰。
  详见 `docs/JAPANESE_MODEL_COMPARISON.md`、`docs/JAPANESE_G2P_DESIGN.md`、`docs/JAPANESE_ANDROID_FEASIBILITY.md`。

### 新增（V0.5）— 日语正确性修复与长音分析

- 修复 **HuBERT 音素→文字索引偏移**（前导 `pau` 污染首字）：起始 sMAE 0.643→**0.365**，P90 1.399→**1.121**。
- 修复 **G2P 文字↔假名映射**：改用 MeCab 词级上下文读音（`は→わ`），修正 `私→わたし` 等。
- pyshiro 按**实际回溯状态序列**聚合音素区间（`pskip` 健壮性；v1/v2 当前无 `pskip`）。
- 根因分析：**句子内部非歌词区间**（`<AP>`/`<SP>`，29/30 段，非静音）是日语误差主因；
  见 `docs/JAPANESE_LONG_VOWEL_ANALYSIS.md`。
- 许可审计（含 MeloX-Android GPLv3 更正）：`docs/LICENSE_AUDIT.md`。
