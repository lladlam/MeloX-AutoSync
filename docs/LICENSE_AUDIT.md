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
| funasr / fa-zh（中文） | FunASR | **FunASR Model License**（自定义） | 需复核 | 商业使用前确认 |
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
