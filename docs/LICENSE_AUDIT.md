# 许可证审计（LICENSE_AUDIT） — V0.5 §4

> 核实对象：MeloX-Android、pyshiro（代码/权重）、HuBERT 音素 CTC、G2P 词典与预处理资源、
> 以及本项目已用的其它模型。结论用于决定**哪些资源可以放进可公开下载的语言包**。

## 0. 关键更正：MeloX-Android 是 **GPL-3.0**，不是闭源

`/MeloX-Android/LICENSE` 为 **GNU General Public License v3.0**（README 徽章亦为 GPLv3）。
`MeloX-Windows` 同为 GPLv3。

**因此：GPLv3 的 pyshiro 代码与 GPLv3 的 MeloX-Android 在许可上**兼容**（不存在「GPL 不能进闭源 App」的问题）。
真正的障碍是**权重授权**，不是代码许可。

## 1. 审计表

| 组件 | 版本/来源 | 许可 | 可否再分发 | 备注 |
|---|---|---|---|---|
| **MeloX-Android** | 本地 `LICENSE` | **GPL-3.0** | 是（须 GPLv3） | 与 GPLv3 组件兼容 |
| MeloX-Windows | 本地 `LICENSE` | GPL-3.0 | 是 | 同上 |
| **pyshiro 代码** | wavtechyukky/pyshiro | **GPL-3.0** | 是（须 GPLv3 + 提供源码） | 与 MeloX-Android 兼容 |
| **pyshiro 权重** v1/v2 | 仓库 `checkpoint/` | **未声明许可** | **否** | 训练数据含多个第三方歌声库；不得放入公开语言包 |
| pyshiro 权重（子模块） | intunist/SHIRO-Models-Japanese | **未声明许可** | **否** | 同上 |
| pyshiro 假名→音素表 | `pyshiro/data/kana2phonemes.table`，源自 ENUNU | ENUNU 为 **MIT** | 是（附 MIT 声明） | 需连同来源声明 |
| **HuBERT 音素 CTC** | prj-beatrice/japanese-hubert-base-phoneme-ctc-v4 | **Apache-2.0** | 是（附声明） | base：rinna/japanese-hubert-base（Apache-2.0） |
| transformers / torch | PyPI | Apache-2.0 / BSD | 是 | 运行时依赖 |
| **pyopenjtalk** | r9y9/pyopenjtalk | 代码 **MIT** | 是 | GitHub API 返回 NOASSERTION；以仓库 README/LICENSE 为准 |
| Open JTalk 词典 | open_jtalk_dic_utf_8 | **modified BSD** | 是（附声明） | pyopenjtalk 首次运行下载（约 22 MB） |
| g2pflow（TIFA 的 G2P） | openvpi/g2pflow | **MIT** | 是 | |
| UniDic（TIFA 日语后端） | unidic | 混合（BSD 系 / 部分 CC） | 需逐项核实 | 约 526 MB；本轮因磁盘配额未完成 TIFA |
| TIFA 代码/模型 | openvpi/TIFA | **MIT** | 是 | 模型 151 MB |
| funasr / fa-zh（中文） | FunASR | 见 §5：权重 **Apache-2.0**（已核实） | **是**（附声明 + 修改说明） | 工具链为 MIT；权重来自 `iic/speech_timestamp_prediction-v1-16k-offline` |
| GTSinger（评测数据） | GTSinger/GTSinger | **CC BY-NC-SA 4.0** | 仅非商业、ShareAlike | 仅本地评测，未提交原始音频 |

## 2. 结论

1. **代码许可不再是障碍**：MeloX-Android 与 pyshiro 同为 GPLv3，可合法集成；
   发布时需提供对应源码并保留许可声明（GPLv3 的安装信息/反 DRM 条款需在发布流程中评估）。
2. **权重授权是唯一硬阻塞**：pyshiro 预训练权重**没有明确许可**，**不得**放进可公开下载的语言包。
   允许继续用于研究与评测；若要分发，需获得作者授权，或用**许可清晰的数据自行训练**。
3. **Apache-2.0 的 HuBERT 音素 CTC 是当前唯一「代码与权重都可再分发」的日语方案**，
   适合作为公开语言包的默认（配 pyopenjtalk MIT + Open JTalk BSD 词典）。
4. G2P 词典（Open JTalk）可随包分发，但需附带 modified BSD 声明；UniDic（TIFA 用）需逐项核实。

## 3. 待办（发布前）

- 向 pyshiro 作者确认 `.hsmm` 权重授权；否则只用 HuBERT 路线。
- 记录所有随包资源的版本与许可文本。
- 复核 GPLv3 与目标应用商店分发条款的兼容性（反 DRM 条款）。

## 4. 公开仓库的内容政策（V1.0）

公开 Git **只包含**：实现代码、测试、文档与**汇总指标**。

- 评测报告（`experiments/reports/*.json`）中的**歌词文本字段**已统一替换为 `<redacted>`
  （`tools/redact_reports.py`）；数值指标、结构、样本数保持不变。
- **保留**的数据集标识符：歌曲名/歌手名/技术标签（如 GTSinger 的 `edelweiss`、`JA-Soprano-1`）
  与 segment id。这些是**数据集元数据**，用于 CC BY-NC-SA 的署名与按歌曲拆分报告，不是歌词摘录。
- 第三方音频与完整歌词（`Test/`、逐曲完整结果 `fullsong_*.json`、人工标注工作文件）
  **全部 gitignore**，仅本地保留。
- 每次发布前应运行：`uv run python tools/redact_reports.py experiments/reports/*.json`，
  并用关键词抽查确认无歌词残留。

## 5. Android 模型包（Release 资产）许可核实 — 2026-10-11

发布 `android-models-v0.1.0-beta.1` 之前的逐项核实，**证据可复现**：

| 资产 | 上游 | 声明的许可 | 证据 |
|---|---|---|---|
| en 权重 `wav2vec2-base-960h` | `facebook/wav2vec2-base-960h` | Apache-2.0 | HF API `cardData.license=apache-2.0`；tag `license:apache-2.0` |
| ja 权重 `japanese-hubert-base-phoneme-ctc-v4` | `prj-beatrice/...` | Apache-2.0 | HF API `cardData.license=apache-2.0`；tag `license:apache-2.0` |
| zh 权重 `speech_timestamp_prediction-v1-16k-offline`（别名 `funasr/fa-zh`） | ModelScope `iic/...` | **Apache License 2.0** | ModelScope API 字段 `License`；仓库 `README.md` front matter `license: Apache License 2.0`；README 全文无 `non-commercial` / `research only` / `不得用于` 等限制关键词 |
| FunASR 工具链（**不随包分发**） | `modelscope/FunASR` | MIT | GitHub API `license.spdx_id=MIT` |
| pyopenjtalk（**不随包分发**） | `r9y9/pyopenjtalk` | MIT（Expat） | 包元数据 `License` 字段 |
| Open JTalk 词典 `open_jtalk_dic_utf_8-1.11`（**不随包分发**，日文 G2P 依赖） | NAIST | modified BSD（三条款） | 词典目录内 `COPYING`：版权归 NAIST，允许源码/二进制再分发，须保留版权声明与免责声明，且不得用 NAIST 名义背书。体积 107.3 MB |
| HTS voice `mei_normal.htsvoice`（**不分发**） | 第三方 | 独立许可 | 仅用于语音合成，与 G2P 无关，因此**不包含**在模型包中 |

**结论**：三个模型权重均可公开再分发（Apache-2.0）。发布时履行的义务：

1. 随 Release 附 **Apache-2.0 全文**与 **NOTICE**；
2. NOTICE 中声明**修改**（动态 INT8 量化、按通道权重量化、中文前端烘焙进图、
   英文归一化烘焙进图）——Apache-2.0 §4(b) 要求对修改后的文件作显著说明；
3. 不改变上游版权归属，不使用上游名称为本项目背书；
4. `pyshiro`（权重无许可）与 GTSinger（CC BY-NC-SA）**继续排除**在任何公开语言包之外；
5. 日文 G2P 词典若日后随 App 打包，须一并附上 NAIST 的 `COPYING` 声明
   （MeloX-Android 为 GPLv3，与 modified BSD 兼容）。
