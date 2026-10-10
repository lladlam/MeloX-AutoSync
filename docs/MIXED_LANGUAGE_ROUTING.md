# 单曲混合语言路由（MIXED_LANGUAGE_ROUTING） — V1.0 P2

> 需求：保留「每语言独立模型包」，但允许**同一首歌内逐行**选择正确模型；
> 拉丁字母**不等于**英语；低置信度**不得**悄悄选错模型；无法可靠分割则**整行回退**。
> 代码：`src/autosync/lyrics/language.py`、`src/autosync/models/router.py`；CLI：`--route-lines`。

## 1. 决策矩阵

对每一行（非演唱行已由 `classify.py` 先行跳过）：

| 行的文字构成 | 路由 | 理由（写入 `route_reason`） |
|---|---|---|
| 含假名 | **ja** | `kana present` |
| 含假名 + 拉丁，且 CJK 占比 ≥0.5 | **ja** | `kana-dominant mixed; ja G2P covers Latin tokens` |
| 仅汉字（无假名/拉丁） | 歌曲主语言（zh 歌→zh，ja 歌→ja） | `kanji-only; routed by song primary=…` |
| 汉字 + 拉丁，汉字占比 ≥0.5 | 主语言（zh/ja） | `CJK-dominant mixed; …` |
| 纯拉丁，主语言=en | **en** | `Latin text in an English-primary song` |
| 纯拉丁，含英语虚词证据 | **en** | `Latin text with English function words` |
| 纯拉丁，无英语证据 | **回退**（不路由） | `latin_ambiguous: may be romanised or stylised` |
| 拉丁占比更高的混合（zh 歌） | **回退**（不路由） | `Latin-dominant mixed; not reliably separable` |
| 拉丁占比更高的混合（**ja 歌**） | **ja** | `…; ja G2P reads Latin tokens (approximate)` |
| 无字母 | 回退 | `no letters` |

**为什么 ja 例外**：日语 G2P（pyopenjtalk）会把拉丁词读成片假名，因此日文歌曲可以吸收拉丁词
（V0.9 基线即如此，0 失败）。**中文却是 `normalize_cjk` 直接丢弃拉丁字符**，所以 zh 歌中的
拉丁主导行必须回退，绝不能假装对齐。

**英语虚词证据**是一份约 120 个常见功能词的清单（the/you/and/to/is/we/…）；**没有**证据时判为
`latin_ambiguous` → 回退。该清单是**保守启发式**，故意不追求高召回。

## 2. 模型加载：惰性 + 复用

- `LanguageRouter` 按需加载被路由到的语言模型，并在整首歌内**复用**（每语言只 `load()` 一次）。
- 不可路由的行**不加载任何模型**（单测覆盖）。
- `unload()` 释放全部语言包。

## 3. 审计（JSON 字段）

每行记录：`line_language`、`route_model`（真实模型名）、`route_reason`、`error`（失败/回退原因）；
`diagnostics.route_lines=true` 与 `routed_models` 列出实际加载的语言包。

## 4. 实测（6 首真实歌曲，`--route-lines`）

| 歌曲 | 主语言 | 行语言分布（演唱行） | 路由到 en 的行 | 回退行 | 失败行（修复前） |
|---|---|---|---|---|---|
| DAMIDAMI | en | en 51 | — | 0 | 0（0） |
| 预言 Prophecy | en | en 62 | — | 0 | 0（0） |
| SAN値直葬 | ja | ja 52 / mixed 2 / latin_amb 2 | 0 | 0 | 0（0） |
| 歌姫失格 | ja | ja 71 | 0 | 0 | 0（0） |
| **不安灵魂收容所** | zh | zh 34 / **en 15** / mixed 5 | **15** | 5 | **5（基线 15）** |
| **提瓦特民谣** | zh | zh 45 / **en 5** / latin_amb 4 | **5** | 4 | **4（基线 9）** |

- 中文歌的英文段落（`If you don't wanna fall asleep…`）现在由**英文模型**对齐，`failed` 15→5、9→4。
- `提瓦特民谣` 的 `(Bora sana, Bora sana)` 等 **4 行**判为 `latin_ambiguous` → **显式回退**，
  **不生成逐字时间**（这是纠正而非“消除失败”：基线这 9 行是 zh 归一化静默清空导致的误导性失败）。

## 5. 限制（诚实）

- 英语虚词清单是启发式；过短或风格化的拉丁行（`lie`、`Bora sana`、`tulu tulu`）在 zh 歌中会回退。
- ja 歌中的拉丁主导行使用 ja 模型（G2P 近似读法），对齐是**近似**的，已在 `route_reason` 标注。
- 中文歌中的“汉字+拉丁”若拉丁占多数 → 整行回退；若中文占多数 → 中文模型（拉丁 token 无逐字时间，
  但**行文本完整保留**，行级时间仍由原 LRC 提供）。
- 单行内的**可靠分割**（把一行拆成中文段+英文段分别对齐）本轮**未实现**；按需求改为整行回退。

## 6. 测试与复现

- `tests/test_language_routing.py`（13 项）：各脚本组合、虚词证据、ja 例外、惰性加载与复用、
  不可路由不加载模型、审计字段、异常类型。
- 复现：`.venv/bin/python benchmarks/full_song_eval.py --tag v1b --modes production --routing`
