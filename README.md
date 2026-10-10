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
- 根因分析：句子内部非歌词区间与低声学后验；见 `docs/JAPANESE_LONG_VOWEL_ANALYSIS.md`。
- 许可审计（含 MeloX-Android GPLv3 更正）：`docs/LICENSE_AUDIT.md`。

### 新增（V0.6）— 暂停感知对齐（研究结论：无收益，保留 V0.5）

- 新增 `alignment/pause_aware.py`（blank-aware / optional-pause / gap-state 半马尔可夫）与
  `benchmarks/pause_eval.py`、`benchmarks/pause_diagnosis.py`。
- **消融结论**：三方案均无稳定收益（gap-state 略差）；**V0.5 仍为默认算法**。
- **根因更正**：极端误差来自**部分段落声学后验无信息**（blank≈1），与内部停顿时长无关
  （corr=−0.09）；`pau`/`sil` 后验≈0。
- 详见 `docs/PAUSE_AWARE_ALIGNMENT.md`。

### 新增（V0.7）— 声学置信度与安全回退

- 新增 `src/autosync/quality/`：`AlignmentQualityAnalyzer` 特征、三状态 `RELIABLE/UNCERTAIN/FAILED`、
  `player_timeline`（非可靠行**整行回退到原始 LRC**）、JSON/CLI 集成（schema `0.2`）。
- `benchmarks/confidence_eval.py`：分组交叉验证（leave-one-singer/song-out）与风险—覆盖率曲线。
- **结论（否定）**：质量门控判别力不足（最佳 AUC 0.665；接受 50% 行后严重行率仍 0.47）→
  仅作保守回退，**V0.5 时间轴保持默认**；详见 `docs/ALIGNMENT_CONFIDENCE.md`、
  `docs/SELECTIVE_ALIGNMENT_REPORT.md`。

### 新增（V0.9）— LRC 非演唱行识别

- 新增 `lyrics/classify.py`：**声学推理前**把每行分为 `SINGING / METADATA / INSTRUMENTAL / UNKNOWN`。
- 制作者署名（中/日/英）与间奏/前奏/尾奏等提示**不参与音素对齐**，JSON 标记 `kind` 与 `skip_reason`。
- **不删除**原始行/时间/文本；跳过的行**不截断**相邻歌词的音频窗口（窗口延伸到下一个演唱行）。
- 播放器：`kind != singing` 的行按**原始时间**整行展示，**不生成虚假逐字高亮**。
- JSON schema → `0.3`（新增 `line_kind_summary`）；`tests/test_line_classify.py`。

### 新增（V0.8）— 引擎正确性、回退契约与 CLI 整合

- 显式**后端注册表**：ja 默认改为 **HuBERT Phoneme CTC**（experimental），ReazonSpeech 保留为可选基线，
  pyshiro 仅本地研究；`--backend` / `--mode` / `autosync info`。
- **原始 LRC 时间与模型时间彻底分离**（`lrc_start`/`lrc_end`/`model_start`/`model_end`/`display_end`）；
  回退使用原始 LRC 的时间与文字。见 `docs/LRC_FALLBACK_CONTRACT.md`。
- **未评估 ≠ 可靠**：门控启用时未评估模型行安全回退；`quality_summary` 在决策后统计。
- 修复交叉验证**特征选择泄漏**：更正后 LOSO F1 0.58 / LOSO-song 0.59（V0.7 偏乐观作废）。
- 新增 `docs/ENGINE_INTEGRATION_AUDIT.md`、`docs/LRC_FALLBACK_CONTRACT.md`、`docs/CLI_END_TO_END.md`；
  `tests/test_cli_integration.py`、`tests/test_fallback_timestamps.py`。

### 新增（V1.0）— 六首真实歌曲验收与混合语言支持

- **六首真实歌曲全部完成真实 CLI 推理**（本地授权素材，未进仓库）：时间轴 0 跨行/0 越界/0 非单调，
  原文与原始 LRC 时间 6/6 保留；逐字覆盖 production **301/348 行（86.5%）**。
- **窗口保守上限**（12s，长音可延伸到 30s 硬上限）与**启发式绝对时间偏移**修复；
  文本重建失败**强制回退**，不再进入播放器逐字时间轴。
- **逐行混合语言路由**（`--route-lines`）：zh 歌英文段落改由英文模型对齐，失败 15→5、9→4；
  拉丁字母不无条件判英文；模型惰性加载并复用；每行记录语言/模型/原因。
- 新增 `benchmarks/full_song_eval.py`、`tools/make_annotation_template.py`（人工标注模板 + 评分器）。
- 文档：`docs/FULL_SONG_EVALUATION.md`、`docs/MIXED_LANGUAGE_ROUTING.md`、`docs/V1_RELEASE_READINESS.md`。
- **真实歌曲精度仍需人工标注**（模板已生成 300 行）；本轮不声称任何真实歌曲精度数字。
