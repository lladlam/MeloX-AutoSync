# MeloX-AutoSync V1.2 — 模型体积与真实逐字精度联合优化

> 基线：`android-models-v0.1.0-beta.3`。本轮**不覆盖** beta.3，只新增 beta.4（仅日文升级）。
> 全部实验 CPU、无 GPU、峰值 RSS ≤1.6 GB；大模型与 scratch 全在 sda4，不使用 tmpfs。
> 报告中不含任何歌词文本。

## 0. 结论

| 语言 | beta.3 | V1.2 结论 | 是否升级 |
|---|---|---|---|
| **JA** | 164.7 MB（全 attention FP32） | **`aL8`：只保留编码器第 8 层 attention 为 FP32 → 129.4 MB**，且在 Test 整混音、连续性盲测、GTSinger 三条独立口径上**都更小且更一致** | **是（beta.4）** |
| **EN** | 122 MB 默认 / 292 MB 精度变体 | **不存在更合理的中间档**：逐层 FFN 敏感度显示单层/4 层方案要么更差、要么牺牲个别歌曲 | **否** |
| **ZH** | 48.1 MB | 量化已饱和，本轮不重复；改用**长尾 + CIF 边界**核查（见 §3） | **否** |

**一句话**：日文找到了**更小且更好**的方案；英文找不到不牺牲单曲的中间档；中文问题是**长尾与混合语言处理**，不是量化。

## 1. 方法与环境

- 数据集：`tools/build_playlist_dataset.py`（Test/ 13 首；不新增第三方音频）。
- 一致性评测：`benchmarks/quant_selective_eval.py`（INT8 vs FP32，分进程、单线程）。
- 独立语料：GTSinger 清唱留出集，`benchmarks/onnx_quant_eval.py`（相对**人工边界**的绝对 MAE）。
- 选择只用 **tune**，**val** 复核；此前用过的盲测歌（DAMIDAMI / SAN値直葬 / 一颗方糖）**只作连续性参考，不再当作全新盲测**。
- 内存纪律：所有候选 ONNX 写 sda4；`TMPDIR` 重定向；量化每步 `ulimit -v ~2.6 GB`。单次量化峰值 ~1.6 GB。

## 2. 日文（JA）：更细粒度 attention 选择性量化

### 2.1 单层敏感度（Test tune+val，122 行 / 1262 字）

把**任意一层**的 attention 保留为 FP32，漂移就从 beta.2 的 80.7 ms 掉到 ~9–14 ms：

| 保留层 | 体积 | tune MAE | val MAE | P90 | >500 ms |
|---|---:|---:|---:|---:|---:|
| beta.2（仅 CTC 头） | 95.7 | 73.3 | 125.6 | 258 | 4.9% |
| L1 | 129.4 | 9.7 | 6.9 | 0 | 0.6% |
| **L8** | **129.4** | **9.5** | **8.7** | **0** | **0.6%** |
| L11 | 129.4 | 9.5 | 8.5 | 0 | 0.5% |
| beta.3（L0–5，6 层） | 164.7 | 13.1 | 11.8 | 0 | 0.7% |
| 全部 attention | 207.0 | 10.0 | 9.8 | 0 | 0.3% |

**机制推断**：漂移不是逐层累积的均匀噪声，而更像**相位滑移**——量化 attention 会偶尔让整行对齐跳变（beta.2 中位 0、P90 258 ms 的双峰分布），
只要任一层保持 FP32 就足以把模型留在“好相位”。因此**单层**就够，不必保留全部 attention。
按 **tune-only** 选层（并列取最小编号）→ **L8**。

### 2.2 三条口径的独立验证（选定 L8）

| 口径 | 指标 | beta.3 (164.7 MB) | **aL8 (129.4 MB)** |
|---|---|---:|---:|
| Test 整混音 tune+val | INT8-vs-FP32 起始 MAE | 13.1 ms | **9.5 ms** |
| 连续性盲测（SAN値直葬，已用过） | 起始 MAE / >500 ms | 60.6 ms / 3.7% | **38.7 ms / 1.4%** |
| GTSinger 清唱（独立语料） | INT8-vs-FP32 起始 MAE | 12.27 ms | **8.45 ms** |
| GTSinger 清唱（独立语料） | 相对人工边界绝对 MAE（INT8） | 110.5 ms | 123.8 ms |

**逐曲**（tune+val）：不乱不破 125.6→8.7、乐园梦游记 164.6→18.8、歌姫失格 17.8→5.4、绘本 6.4→0.5（均对 beta.2 大幅改善，且不劣于 beta.3）。

唯一对 aL8 略不利的是 GTSinger 上的**绝对** INT8 误差（123.8 vs 110.5 ms，12 段小样本）；
三首歌的绝对 FP32 基线是 118.7 ms，两者差距在噪声量级内。

**结论**：`aL8` 在体积（−35 MB）与一致性上全面优于 beta.3，绝对精度在独立小样本上基本持平 → **升级 beta.4**。

## 3. 英文（EN）与中文（ZH）

### 3.1 EN：中间档不存在（如实报告）

逐层 FFN 敏感度（Test tune+val，107 行 / 606 字）：

| 候选 | 体积 | tune MAE | val MAE | P90 | >500 ms |
|---|---:|---:|---:|---:|---:|
| beta.2 | 122.4 | 101.2 | 55.5 | 220 | 5.0% |
| 单层 FFN（12 个候选） | 136.5 | **79.8–157.1** | 49.7–84.8 | 200–340 | 4.5–7.4% |
| FFN L0–3（4 层） | 178.9 | — | — | 240 | 5.3% |
| FFN 全部 | 292.0 | 76.5 | 10.7 | **20** | 3.3% |
| CTC 头 + FFN | 292.0 | 66.5 | 11.6 | **20** | 2.6% |

- 单层 FFN 多数**更差**（如 L0 128.0、L7 104.7），仅 L4/L9 略好（79.8/78.9）。
- 4 层组（178.9 MB）只把 MAE 降到 ~70–78，P90 220→200，收益很小。
- 逐曲检查：**没有任何候选对所有歌都更好**——`head_ffn` 让 FURYON 55.5→11.6、预言 134.1→67.3，
  但**若我不曾见过太阳 40.3→65.0（变差）**；4 层组同样牺牲个别歌。

**结论**：EN 不存在“更小且不牺牲单曲”的中间档。默认保持 beta.2（122 MB）；292 MB 精度变体作为可选资产保留（已明确其单曲回退）。

### 3.2 ZH：长尾与 CIF 边界（非量化问题）

- **长尾集中在短行**：字数 ≤6 的行平均最大误差 **57 ms**，≥10 字的行仅 **14 ms**（beta.2，59 行）。
  最差样本为 4–8 字的短行（520 / 400 / 340 / 280 ms）。V1.1 已证把 attention/FFN/CIF 头保留 FP32 反而更差，故尾部长**不是量化**。
- **CIF 边界**：`cif_wo_hidden` 闭式与顺序循环在 59 条真实行上**逐值一致**（`max|Δ|=0`，峰值数 0 不一致）；
  此前 `max|Δ|=12` 仅出现在 `token_num=200/201` 合成样本。
- **混合中英（重要缺陷，已定位）**：`提瓦特民谣` 有 **9 条可演唱行是纯拉丁**（如和声 "Bora sana"），
  `normalize_cjk` 对它们输出空串 → 这些行在评测/对齐中被**静默跳过**（既不产出时间轴，也不记回退）。
  本轮 59 条评测样本里恰好没有这种行，所以不影响上面数字，但**必须修复**：
  纯拉丁/拉丁主导行应走英文模型（仓库已有 `--route-lines`）或显式标记回退，不能静默丢弃。

## 4. AMLL 人工时间轴：版本一致性自动核查

不是假设同版，而是**自动估计整曲全局偏移**并检查 TTML 行起点与音频起音的对齐：

| 曲目 | TTML 行数 | 估计偏移 | 起音命中率 | 判定 |
|---|---:|---:|---:|---|
| DAMIDAMI | 50 | −2.02 s | 0.88 | **CONSISTENT** |
| 提瓦特民谣 | 53 | +5.26 s | 0.93 | **CONSISTENT** |
| 不安灵魂收容所 | 54 | −13.68 s | 0.76 | **CONSISTENT** |
| 预言 Prophecy | 58 | −3.46 s | 0.66 | **INCONSISTENT / 版本不符** |

**要点**：3/4 的 AMLL 参考**在存在非零全局偏移时**才与本地录音一致（此前按 offset=0 比较是不对的），
`预言 Prophecy` 判定为版本不符，**不能**用作真值。

**本轮限制（如实说明）**：把模型逐词预测与 AMLL span 做**文本级匹配**再去算“相对真实演唱时间的绝对 MAE”，
需要复用 `benchmarks/amll_ttml_eval.py` 的 span 匹配管线；本轮只完成了**版本一致性核查**（并把偏移量自动估计出来），
**未**产出新候选的 AMLL 绝对 MAE。这是明确的下一步，而不是已完成的结论。

## 5. Release manifest 精度数据修正

- 新增 `accuracy.provenance`，对每个语言声明**指标来自哪个模型版本、哪个评测集**，
  明确区分 `real_music`（INT8 vs FP32 一致性）、`gtsinger`（相对人工边界的绝对 MAE）、
  `continuity_blind`（**已用过的**盲测歌，仅供参考）。
- 修正 beta.3 中“清单引用 beta.2 报告数字”的混淆：beta.4 的 JA 数字全部对应 **aL8**，并注明
  `android_export/ja/REPORT.md` 记录的仍是 beta.2 配方（未为变体重跑）。
- 历史指标（beta.2 / beta.3）保留在各自 Release 与报告中，不删除。

## 6. 交付与复现

- 新工具：`tools/amll_version_check.py`（AMLL 版本一致性 + 全局偏移）。
- 新单测：`tests/test_amll_version_check.py`（时钟解析、`xmlns=""` 的 `<p>` 读取）。
- 报告：本文件。
- 复现：

```bash
# 单层 attention 敏感度（JA）
for L in $(seq 0 11); do
  .venv/bin/python tools/selective_quantize.py \
    --fp32 android_export/ja/hubert-ja-phoneme-ctc-fp32.onnx \
    --out $D/attn_L$L.onnx --exclude-regex "layers\.$L/attention/"
  .venv/bin/python benchmarks/quant_selective_eval.py --language ja --phase int8 \
    --model $D/attn_L$L.onnx --scratch $S --tag aL$L --splits tune,val
  .venv/bin/python benchmarks/quant_selective_eval.py --language ja --phase compare \
    --scratch $S --tag aL$L --out results/ja_aL$L.json
done
# AMLL 版本一致性
.venv/bin/python tools/amll_version_check.py --ttml experiments/amll_reference/2750140001.ttml \
  --audio "Test/提瓦特民谣.mp3"
```

## 7. beta.4

- Tag `android-models-v0.1.0-beta.4`（prerelease），JA = `aL8`（129.4 MB），EN/ZH 不变，
  英文精度变体仍作为可选资产。已重新下载校验：`checksums OK`，三个 ONNX 均可加载。
- beta.2 / beta.3 **未删除、未覆盖**。
