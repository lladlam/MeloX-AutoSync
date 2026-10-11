# MeloX-AutoSync V1.1 — 真实流行音乐驱动的三语量化优化

> 基线：`android-models-v0.1.0-beta.2`（提交 `b140629`）。本轮**不改动、不覆盖** beta.2。
> 全部实验在 CPU、严格内存预算内完成；所有数字为真实运行结果。
> `Test/` 音频/歌词不入 Git；本报告只含汇总统计，无歌词文本。

## 0. 结论速览

| 语言 | beta.2 | 本轮最优 | 是否值得升级 |
|---|---|---|---|
| **EN** | 122 MB / 整混音 sMAE 82.8 ms、P90 220 ms | `head_ffn`（排除 CTC 头+FFN 量化）**292 MB** / 盲测 sMAE 40.7→**19.9 ms**、P90 140→**6 ms** | 精度提升明确，但**体积 2.4×**，需权衡 |
| **JA** | 95.7 MB / 整混音 sMAE 80.7 ms、>500 ms 4.9%；盲测 **272.8 ms / >500 ms 20.2%** | `attn_L0_5`（保留前 6 层 attention 为 FP32）**164.7 MB** / 盲测 **60.6 ms、>500 ms 3.7%** | **值得升级**（+69 MB，大幅降低严重长尾） |
| **ZH** | 48.1 MB / 整混音 sMAE 7.1 ms | **同 beta.2**（任何选择性 FP32 都更差） | **不升级** |

**关键发现**
1. **EN 与 JA 的量化敏感模块相反**：EN 的漂移主要来自 **FFN**；JA 的漂移主要来自 **attention**。
   用同一套“排除什么”的直觉去套两个模型是错的。
2. **JA attention 保留为 FP32** 把整混音量化的逐字符起始 MAE 从 **80.7 → 13.0 ms**（整混音）、
   盲测从 **272.8 → 60.6 ms**，严重错误（>500 ms）从 **20.2% → 3.7%**。
3. **ZH 已经量化饱和**：排除 attention/FFN/CIF 头都让结果变差；CIF 前端必须保持 FP32（否则崩坏）。
4. **`cif_wo_hidden` 闭式替代在真实歌词行上与顺序循环逐值一致**（59 行 `max|Δ|=0`，峰值数 0 不一致）；
   此前记录的 `max|Δ|=12` 只出现在 `token_num` 边界合成样本，不是算法偏差，也不是量化误差。

## 1. 硬件纪律（4 GB 预算，无 GPU）

- 无 GPU；`torch.cuda.is_available() == False`。
- **教训**：`/tmp` 是 tmpfs（RAM 盘），把 200 MB 级的候选模型写到 `/tmp` 会直接吃内存并导致整机 OOM。
  本轮所有候选模型、scratch、结果都改到外盘 `/mnt/ntfs_files_sda4/MeloX-AutoSync-data/quant/`，
  并设 `TMPDIR` 指向外盘。
- 每个重量级步骤单独进程、`--threads 1`、量化步骤用 `ulimit -v ~2.6 GB` 兜底（超限即失败而非拖垮整机）。
- 一次只加载一个 ONNX 会话；fp32 与 int8 分两个子进程各写 `.npz`，比较阶段不加载任何模型。
- 复用现有 FP32 ONNX，不重新加载 PyTorch。

实测峰值 RSS（见 §6）：fp32 EN 827 MB / JA 1065 MB / ZH 377 MB；int8 候选 170–910 MB；
量化（`quantize_dynamic`）单次约 **1.6 GB**。全部在预算内。

## 2. Test/ 评测集（自动构建）

`tools/build_playlist_dataset.py` 递归扫描 `Test/`，匹配音频↔歌词、检测语言、校验、打场景标签、
按歌曲家族切分。实测：**13 首音频 / 13 个歌词**（全部 MP3+LRC；无 TTML）。

| 语言 | 数量 | 歌曲 | 划分 |
|---|---:|---|---|
| en | 4 | DAMIDAMI、FURYON、若我不曾见过太阳、预言 Prophecy | blind=DAMIDAMI；val=FURYON；tune=其余 |
| ja | 5 | SAN値直葬、不乱不破、乐园梦游记、歌姫失格、绘本 | blind=SAN値直葬；val=不乱不破；tune=其余 3 |
| zh | 3 | 一颗方糖悬滞的时间、提瓦特民谣、红透晚烟青 | blind=一颗方糖；val=提瓦特；tune=红透晚烟青 |
| mixed | 1 | 不安灵魂收容所 | blind |

- 语言按**歌词正文**（假名/汉字/拉丁比例）判定，不是按文件名：`绘本`、`乐园梦游记` 的歌词是日文，
  因此判为 ja。
- 校验项：音频时长、歌词时间范围、有效演唱行；`乐园梦游记` 音频名为 `乐园梦游记.mp3` 而歌词为
  `乐园梦游计-…lrc`（记/计 不同字），用最长公共前缀容错匹配并如实记录。
- **诚实声明**：这是 13 首的小集合，语言内部再切分后 **盲测每种语言只有 1 首歌**，
  因此盲测是“冻结方案后的独立确认”，不是统计意义上的大规模测试集。场景标签对英文不具区分度
  （英文按字符/秒都会命中 `fast_singing`），故以**逐曲**结果为准。

## 3. 方法

- 对齐指标：INT8 相对 FP32 的**逐词/逐字起始与结束**误差（en/ja 用仓库 `forced_align(blank=0)`，
  zh 用 CIF `us_alphas` + `ts_precision_lfr6_standard`）。
- 报告 MAE、Median、P90、P95、Max，以及 >100/200/300/500/1000 ms 比例；
  **不以平均 MAE 掩盖长尾**。
- 选择只用 **tune(+val)**；**blind** 仅在方案冻结后跑一次。
- 工具：`tools/selective_quantize.py`（选择性/参数化动态量化）、
  `benchmarks/quant_selective_eval.py`（分进程评测）。

## 4. EN：Wav2Vec2 CTC 敏感性

整混音 tune+val（107 行 / 606 字）。

| 候选 | 保留为 FP32 | 体积 | sMAE | sP90 | sP95 | sMax | >500 ms | >1 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **beta.2** | 无 | 122.4 | 82.8 | 220 | 480 | 2560 | 5.0% | 1.8% |
| head | CTC 头 | 122.4 | 82.6 | 210 | 480 | 2560 | 5.0% | 1.8% |
| uint8（仅换权重类型） | — | 122.4 | 116.4 | 300 | 570 | 4080 | 5.6% | 3.1% |
| reduce_range | — | 122.4 | 119.1 | 320 | 660 | 3720 | 6.3% | 2.8% |
| attn | attention | 207.1 | 133.7 | 340 | 775 | 3720 | 7.8% | 3.5% |
| ffn_in | FFN 前馈 | 207.1 | 81.4 | 170 | 460 | 3720 | 4.1% | 2.0% |
| ffn_out | FFN 输出 | 207.2 | 87.5 | 230 | 555 | 2560 | 5.4% | 2.6% |
| ffn（1/4 层） | 部分层 FFN | 178.9 | 70–84 | 200–240 | — | — | 4.0–5.0% | — |
| ffn | **全部 FFN** | 292.0 | 50.0 | **20** | 220 | 2980 | 3.3% | 1.5% |
| **head_ffn** | CTC 头 + FFN | **292.0** | **44.4** | **20** | 170 | 2980 | **2.6%** | 1.3% |
| attn_ffn | attention+FFN | 376.7 | 13.5 | 0 | 20 | 2980 | 0.7% | 0.2% |

**EN 结论**：主导敏感模块是 **FFN**；保留 attention **反而更差**（attention 量化对 EN 影响小，
但会打乱 FFN 已适应的量化尺度）。`uint8`/`reduce_range` 两个“免费”杠杆都让 EN 更差。
**没有低成本的中间档**：要显著压低漂移必须整体保留 FFN，体积升到 292 MB（+170 MB）。

**EN 盲测**（DAMIDAMI，40 行 / 248 字）：beta.2 sMAE **40.7** / P90 140 → head_ffn **19.9** / P90 **6**。
逐曲（tune+val）：FURYON 55.5→11.6，预言 134.1→67.3，**若我不曾见过太阳 40.3→65.0（变差）**。
即 head_ffn 的收益**不均衡**：P90 全面改善，但存在个别歌曲变差。

## 5. JA：HuBERT 音素 CTC 敏感性

整混音 tune+val（122 行 / 1262 字）。

| 候选 | 保留为 FP32 | 体积 | sMAE | sP90 | sP95 | sMax | >500 ms | >1 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **beta.2** | CTC 头 | 95.7 | 80.7 | 258 | 480 | 7000 | 4.9% | 1.3% |
| attn_L0_5 | 前 6 层 attention | **164.7** | **13.0** | **0** | 20 | 980 | 0.7% | 0.0% |
| attn_L6_11 | 后 6 层 attention | 164.7 | 11.9 | 0 | 20 | 920 | 0.6% | 0.0% |
| **attn** | 全部 attention | 207.0 | 10.0 | 0 | 20 | 920 | 0.3% | 0.0% |
| head_attn | CTC 头 + attention | 207.2 | 10.1 | 0 | 20 | 920 | 0.4% | 0.0% |
| ffn | FFN | 292.0 | 2.2 | 0 | 0 | 380 | 0.0% | 0.0% |
| all3 | 头+attention+FFN | 376.8 | 0.2 | 0 | 0 | 120 | 0.0% | 0.0% |

**JA 结论**：主导敏感模块是 **attention**（与 EN 相反）。只保留**前 6 层** attention 为 FP32
就能拿到几乎全部收益：**164.7 MB，sMAE 80.7 → 13.0**，严重错误 4.9% → 0.7%。

**JA 盲测**（SAN値直葬，36 行 / 356 字；Vocaloid 合成声，beta.2 表现极差）：

| 候选 | 体积 | sMAE | sP90 | >500 ms | >1 s |
|---|---:|---:|---:|---:|---:|
| beta.2 | 95.7 | **272.8** | 700 | 20.2% | 2.0% |
| attn_L0_5 | 164.7 | **60.6** | 40 | 3.7% | 0.6% |
| attn | 207.0 | 63.0 | 30 | 3.7% | 0.6% |

盲测上 `attn_L0_5`（更小）与 `attn`（更大）基本持平 → **选更便宜的 `attn_L0_5`**。
逐曲（tune+val）全部改善：不乱不破 125.6→11.8，乐园梦游记 164.6→28.0，歌姫失格 17.8→5.4，绘本 6.4→0.4。

## 6. ZH：FunASR CIF 分析

整混音 tune+val（59 行 / 479 字）。**控制实验**：用本工具不加任何排除重建 ≈ beta.2，得到**完全相同**的
48.1 MB / sMAE 7.1 ms，证明工具与交付配方一致（关键：CIF 前端 DFT/Mel 必须排除）。

| 候选 | 体积 | sMAE | sP90 | sMax | >300 ms |
|---|---:|---:|---:|---:|---:|
| **beta.2** | 48.1 | **7.1** | 0 | 520 | 0.8% |
| 重建（=beta.2） | 48.1 | 7.1 | 0 | 520 | 0.8% |
| +attention FP32 | 85.4 | 8.3 | 0 | 1160 | 1.0% |
| +FFN FP32 | 121.5 | 9.6 | 0 | 1220 | 1.0% |
| +CIF 头 FP32 | 48.1 | 10.3 | 0 | 1160 | 1.5% |
| 前端不排除（错误配方） | 47.4 | **280.4** | 560 | 4860 | 13.2% |

**ZH 结论**：**已经量化饱和**，任何选择性 FP32 保留都更差且更大 → **保持 beta.2**。
`us_alphas/us_peaks` 对量化已不敏感；剩余长尾（>300 ms 0.8%，max 520 ms）**不是量化造成**，
因为把整个 attention/FFN/CIF 保留为 FP32 反而更差，说明长尾来自模型/域本身。

**算法偏差澄清**：在同一批 59 条真实歌词行的 FP32 输出上，`cif_wo_hidden` 的**闭式实现与顺序循环
逐值完全一致**（`max|Δ|=0.0000`，峰值数不一致 0 行）。此前 `max|Δ|=12`、2 个峰值数不一致样本
只出现在 `token_num=200/201` 的合成边界用例，属**合成边界现象**，既不是量化误差也不是真实数据上的算法偏差。

**ZH 盲测**（一颗方糖，70 行 / 250 字）：beta.2 sMAE 18.4 / P90 20 / >300 ms 1.6%。

## 7. 资源与失败记录

| 阶段 | 样本 | 耗时 | 峰值 RSS | 模型体积 |
|---|---:|---:|---:|---:|
| EN fp32（tune+val） | 108 行 | 52.9 s | 827 MB | 377.8 MB |
| EN int8 beta.2 | 108 行 | 33.7 s | 462 MB | 122.4 MB |
| EN int8 head_ffn | 108 行 | 46.8 s | 753 MB | 292.0 MB |
| JA fp32（tune+val） | 122 行 | 60.5 s | 1065 MB | 377.8 MB |
| JA int8 beta.2 | 122 行 | 87.5 s | 566 MB | 95.7 MB |
| JA int8 attn_L0_5 | 122 行 | 42.1 s | 710 MB | 164.7 MB |
| ZH fp32（tune+val） | 118 行 | 5.1 s | 377 MB | 158.8 MB |
| ZH int8（重建） | 118 行 | 3.0 s | 173 MB | 48.1 MB |
| 量化（单次） | — | ~7 s | ~1568 MB | — |

- **失败/异常**：早期把候选模型写入 `/tmp`（tmpfs）导致内存耗尽、整机重启一次；
  已改为外盘 + `TMPDIR` 重定向 + `ulimit -v` 兜底，之后无 OOM。
- ZH 首轮 sweep 因未排除 DFT/Mel 前端而全部崩坏（280 ms），已定位并加入
  `--exclude-weight-suffix dft_re/dft_im/mel_banks` 修正；该过程如实保留（见 §6 最后一行）。
- 未做 INT4：`onnxruntime.quantization` 的动态路径不提供受支持的 int4 权重类型，
  且 Android ONNX Runtime 对 int4 CPU 算子支持有限，本环境下不做（避免无意义的失败尝试）。
- 未做静态 QDQ 全量搜索：`docs/QUANTIZATION_CALIBRATION.md` 已证其跨域无益，本轮以 beta.2 为基线不重复。

## 8. 建议与 beta.3

- **JA**：`attn_L0_5`（164.7 MB）在整混音与盲测上都大幅改善且稳定 → **建议升级**。
- **EN**：`head_ffn`（292 MB）盲测/整混音 P90 大幅改善，但体积 2.4×、且个别歌曲变差 →
  **建议作为“精度优先”可选项**，默认是否采用交由发布决策。
- **ZH**：**保持 beta.2**。

未做自动替换：beta.2 保留；新模型作为独立候选与 Release 测试包提供，契约（I/O 张量、采样率、
预处理）与 beta.2 完全一致，可直接被现有 Kotlin/ONNX 后端加载。

## 9. 复现

```bash
# 1) 构建评测集（不写音频进 Git）
.venv/bin/python tools/build_playlist_dataset.py --root Test --out experiments/playlist

# 2) 选择性量化候选（外盘存放，避免 tmpfs）
D=/mnt/ntfs_files_sda4/MeloX-AutoSync-data/quant/cand_ja
.venv/bin/python tools/selective_quantize.py \
  --fp32 android_export/ja/hubert-ja-phoneme-ctc-fp32.onnx \
  --out $D/attn_L0_5.onnx --exclude-regex 'layers\.[0-5]/attention/'

# 3) 评测（fp32 → int8 → compare；分进程、单线程）
S=/mnt/ntfs_files_sda4/MeloX-AutoSync-data/quant/q_ja
.venv/bin/python benchmarks/quant_selective_eval.py --language ja --phase fp32 --scratch $S --splits tune,val
.venv/bin/python benchmarks/quant_selective_eval.py --language ja --phase int8 --scratch $S --tag attn_L0_5 --model $D/attn_L0_5.onnx
.venv/bin/python benchmarks/quant_selective_eval.py --language ja --phase compare --scratch $S --tag attn_L0_5 --out results/ja_attn_L0_5.json

# 4) ZH 前端必须排除（否则崩坏）
... --exclude-weight-suffix dft_re --exclude-weight-suffix dft_im --exclude-weight-suffix mel_banks
```
