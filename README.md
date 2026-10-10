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

V0.1 是研究原型。请阅读 `docs/EXPERIMENT_RESULTS.md` 了解**已经真实运行验证**的内容，
以及 `docs/KNOWN_LIMITATIONS.md` 了解**尚未完成 / 存在风险**的部分。
