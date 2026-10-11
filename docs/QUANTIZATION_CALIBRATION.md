# 量化校准与真实测评报告（GTSinger 歌声集）

> 本报告只记录**真实运行**的结果。所有音频/标注来自 GTSinger（CC BY-NC-SA 4.0），
> 仅存放在仓库外 `/mnt/ntfs_files_sda4/MeloX-AutoSync-data/`，不入 Git、不再分发。
> 复现命令见文末。

## 1. 为什么要做这一步

`android_export/*/REPORT.md` 测的是 **INT8 相对 FP32 的一致性**，素材主要是 `Test/`
的**整混音流行歌**。它回答不了两个问题：

1. 模型相对**人工边界**到底差多少（绝对精度）？此前没有任何真实数字。
2. 那些一致性差异里，有多少来自**量化**，有多少来自**域不匹配**（整混音 vs 清唱）？

本轮用 GTSinger（专业录音棚清唱，带**人工词/音素边界**）回答这两个问题。

## 2. 数据与隔离

GTSinger（`AaronZ345/GTSinger`，CC BY-NC-SA 4.0）本地池，**按歌手/歌曲隔离**：

| 池 | 语言 | 内容 | 用途 |
|---|---|---|---|
| `gtsinger/` | en | EN-Alto-1（edelweiss / let it go / my love）、EN-Tenor-1（Shallow / Rolling in the Deep） | 校准候选 |
| `gtsinger/` | ja | JA-Soprano-1（heartful song / 时雨 / first love）、JA-Tenor-1（さんぽ） | 校准候选 |
| `gtsinger/` | zh | ZH-Alto-1（大鱼 / 南山南 / 十年 / 成都）、ZH-Tenor-1（单身情歌 / 最长的电影） | 校准候选 |
| `gtsinger_val/` | en/ja/zh | EN-Alto-2·EN-Tenor-1、JA-Tenor-1·JA-Soprano-1、ZH-Tenor-1 的**其它歌** | 留出验证 |
| `gtsinger_extra/` | en/ja/zh | 追加 10 段（见 `manifest.json`） | 验证补充 |

`gtsinger_val`（36 段）与校准池**无同曲重叠**；JA 静态校准实验进一步做到**歌手隔离**
（校准用 JA-Soprano-1，验证用 JA-Tenor-1）。

## 3. 真实绝对精度（GTSinger 留出集，n=12/语言）

用**交付的 ONNX 图**（fp32 与 int8）跑仓库自己的强制对齐（en/ja：CTC Viterbi，blank=0；
zh：CIF `us_alphas` + `ts_prediction_lfr6_standard`），与 GTSinger 人工**起始边界**比较：

| 语言 | 引擎 | FP32 起始 MAE | INT8 起始 MAE | **INT8 vs FP32 MAE** |
|---|---|---:|---:|---:|
| en | ctc-wav2vec2 | 100.11 ms | 99.36 ms | **2.78 ms** |
| ja | ctc-hubert-phoneme | 118.73 ms | 175.18 ms | **105.90 ms** |
| zh | monotonic-aligner-cif | 181.38 ms | 181.04 ms | **0.82 ms** |

对照 `Test/` 整混音（en，45 行 / 230 词）：

| 场合 | INT8 vs FP32 起始 MAE |
|---|---:|
| GTSinger 清唱（本轮） | **2.78 ms** |
| `Test/` 整混音（既有） | **121.30 ms** |

**结论（重要）**：
- 英文的“121 ms 量化漂移”几乎全部来自**整混音域**，不是量化本身的普遍代价；
  在清唱上量化影响仅 **2.8 ms**。
- 中文 CIF 链路量化影响**可忽略**（0.82 ms）。
- **日文是唯一在清唱上也明显受量化损伤的链路**（105.9 ms），是需要解决的点。
- 绝对精度（相对人工边界）：清唱上 en≈100 ms、ja(fp32)≈119 ms、zh≈181 ms；
  这些是**真实数字**，此前报告从未给出。

## 4. 静态 QDQ 校准实验（任务书建议的路线）

用 `tools/quantize_static_calib.py` 以**真实歌声**做校准，构建静态 QDQ INT8（MatMul/Gemm，
QInt8 权重、QUInt8 激活、per-channel），与交付的动态 INT8 对比：

| 实验 | 校准集 | 验证集 | 动态 INT8 | 静态 QDQ INT8 | 结论 |
|---|---|---|---:|---:|---|
| JA（歌手隔离） | JA-Soprano-1 清唱 | **JA-Tenor-1** 清唱 | 174.2 ms | 171.5 ms | **无改善** |
| JA（同歌手不同曲） | JA-Soprano-1 清唱 | JA-Soprano-1 其它曲 | 37.6 ms | 28.9 ms | 略好，但不干净 |
| EN（跨域） | GTSinger 清唱 | `Test/` 整混音 | 121.30 ms | **246.78 ms** | **明显变差** |

**结论**：静态 QDQ + 真实歌声校准**没有**改善交付模型的 INT8 一致性；
跨域校准（清唱校准 → 整混音验证）使英文翻倍恶化。因此**维持现有动态 INT8 配方**，
不因“通过发布检查”而更换。

## 5. 关于“训练”

本轮**没有**做声学模型微调，原因如实说明：

- 本机**无 GPU**（`torch.cuda.is_available() == False`），可用内存约 3 GB；
  Wav2Vec2/HuBERT 微调在此环境不可行。
- 更关键的是，能显著改善的是**域**（整混音/副歌/和声），而不是量化；
  用 CC BY-NC-SA 数据微调得到的权重还受**非商业 + ShareAlike**约束，
  不宜直接并入公开可发布的模型包。
- 因此本轮把“训练”落在**可交付、可验证**的量化校准上，并把结果如实报告。

## 6. 建议

1. **英文/中文**：现有动态 INT8 可用；整混音的 121 ms 主要来自域，应靠**质量门控 +
   逐行回退**兜底，而不是继续压量化。
2. **日文**：清唱上 INT8 漂移 106 ms 是真实缺陷；静态 QDQ 已验证无效。
   下一步应评估 (a) 保留 FP32 日文回退，(b) 量化感知微调（需 GPU 与合规数据），
   或 (c) 仅对日文降低 INT8 使用面（行级时间）。
3. **验收口径**：发布说明应引用本报告的**绝对 MAE**（清唱），不要用 INT8-vs-FP32
   一致性数字冒充精度。

## 7. 复现

```bash
# 数据（外盘，CC BY-NC-SA 4.0；不入 Git）
python tools/fetch_gtsinger_subset.py --out /mnt/.../gtsinger_val --max-segments 3 \
  --spec "Japanese/JA-Tenor-1/Breathy/ブルーバード" ...

# 绝对精度（交付 ONNX）
OMP_NUM_THREADS=4 .venv/bin/python benchmarks/onnx_quant_eval.py --language ja \
  --pool /mnt/.../gtsinger_val --out /tmp/ja.json

# 静态 QDQ 校准 + 对比
OMP_NUM_THREADS=4 .venv/bin/python tools/quantize_static_calib.py \
  --fp32 android_export/ja/hubert-ja-phoneme-ctc-fp32.onnx \
  --calib-dir /mnt/.../gtsinger/Japanese/JA-Soprano-1 \
  --out /tmp/ja_qdq.onnx
OMP_NUM_THREADS=4 .venv/bin/python benchmarks/onnx_quant_eval.py --language ja \
  --int8 /tmp/ja_qdq.onnx --pool /mnt/.../gtsinger_val
```

交付模型未被修改：`android_export/en/wav2vec2-base-960h-int8.onnx` 的 sha256 仍为
`1c8213118da7f555e4b68c0e08e85a4c471ea2a3e0fdf5620d27412800babd3f`。
