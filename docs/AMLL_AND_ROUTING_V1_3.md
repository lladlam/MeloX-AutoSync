# MeloX-AutoSync V1.3 — AMLL 可信评测、混合语言修复与 beta.4 验证

> 基线提交 `1064b4f`，基线 Release `android-models-v0.1.0-beta.4`。
> 本轮**不新增量化模型、不发布 beta.5**；只修缺陷、做真实评测、补契约。
> 全部实验 CPU、无 GPU、峰值 RSS ≤1.6 GB；大文件在 sda4。报告不含歌词文本。

## 0. 结论

1. **AMLL 版本校验器已重写并否定了 V1.2 的结论**：加入负对照、分段稳定性、去除循环移位后，
   4 首 AMLL 参考**全部**为 `UNVERIFIED` 或 `MISMATCH` —— V1.2 报告的偏移（−2.02 / +5.26 / −13.68 / −3.46 s）**不可信**。
   因此在没有人工听音确认前，**不能**用这 4 首计算“相对 AMLL 人工时间轴”的绝对 MAE。
2. **混合语言静默漏行已从底层修复**：pure-Latin 行改走英文模型；CJK 空结果改为**显式报错**；
   新增覆盖核算，**所有演唱行都有唯一结果**（`提瓦特民谣` 9 条拉丁行现在路由到 EN，`unaccounted=0`）。
3. **beta.4 验证**：日文 `aL8`（129.4 MB）在 Test 一致性与连续性盲测上优于 beta.3；
   但在**唯一有人工边界**的 GTSinger 清唱上，绝对精度与 beta.3 基本持平（详见 §4）。
   英文 292 MB 精度变体在 GTSinger 上**没有绝对收益**（a-cappella 99.4 vs 100.1 ms）——它只在整混音上有用，且会牺牲个别歌曲。
4. **契约测试向量**已生成（纯自研合成音频，可随仓库分发），三语张量形状已核对。

## 1. 三种精度口径（严格分离，绝不混用）

| 口径 | 含义 | 本轮状态 |
|---|---|---|
| **A** | INT8 相对 FP32 的一致性误差 | 有（`benchmarks/quant_selective_eval.py`） |
| **B** | INT8/FP32 相对 **AMLL 人工时间轴**的绝对误差 | **不可得**（无 VERIFIED 歌曲，见 §2） |
| **C** | 相对 **GTSinger 人工边界**的绝对误差（清唱） | 有（`benchmarks/onnx_quant_eval.py`） |

## 2. AMLL 版本校验器：重写与负对照

原实现的问题（逐条修复）：

| 原缺陷 | 修复 |
|---|---|
| 宽带能量≠人声起音 | 改为 **200–4000 Hz 人声带**的谱通量（spectral flux） |
| `np.roll` 循环移位产生边界伪匹配 | 偏移扫描**只用有效重叠区**，不 wrap |
| ±30 s 搜索过大 | 默认 ±15 s，并报告是否触界 |
| 只看全局命中率 | **按歌曲三等分**分别估计偏移，报告离散度 |
| 无负对照 | 加入**错误歌曲 / 随机偏移(200 次) / 打乱时间轴**三类对照 |
| 仅凭阈值判“同版” | 输出 **VERIFIED / UNVERIFIED / MISMATCH**；`VERIFIED` **必须**由人工 `--human-verified` 显式确认 |

**实测（每首用另一首歌作错误对照）**：

| 曲目 | n | 最佳偏移 | 命中分 | 随机 P95 | 打乱 | 错误歌曲 | 三段时间偏移 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| DAMIDAMI | 50 | −0.64 s | 0.272 | 0.215 | 0.217 | 0.251 | [15.0, −0.64, 1.20]（离散 15.6 s） | **UNVERIFIED** |
| 提瓦特民谣 | 53 | +3.32 s | 0.469 | 0.393 | 0.342 | 0.279 | [3.32, −11.02, 9.84]（20.9 s） | **UNVERIFIED** |
| 不安灵魂收容所 | 54 | +3.48 s | 0.407 | 0.369 | 0.314 | 0.325 | [12.02, 10.30, −10.24]（22.3 s） | **UNVERIFIED** |
| 预言 Prophecy | 58 | +5.58 s | 0.281 | 0.245 | 0.253 | **0.435** | [−9.7, 0.02, 5.58] | **MISMATCH** |

**要点**：
- 四首的**分段偏移极不稳定**（15–22 s 抖动）→ 单一全局偏移不成立；V1.2 的“CONSISTENT”是统计假象。
- `预言` 的**错误歌曲对照分数（0.435）反而高于真实对齐（0.281）** → 明确 MISMATCH。
- 校验器自身用**合成正对照**做了单测（已知 +3 s 偏移能恢复、噪声不优于随机对照、不稳定参考被判不稳定）。
- 因此 §1 口径 B **本轮不产出数字**：没有证据就不能把脚本时间去当真值（任务明确要求）。
- 人工确认环节保留：只有 `--human-verified --human-note ...` 才会把状态置为 `VERIFIED`。

## 3. 混合语言静默漏行：底层修复

**根因**：中文 `normalize_cjk` 会丢弃拉丁字符；`提瓦特民谣` 有 **9 条纯拉丁演唱行**（和声），
归一化后为空，**既不出时间轴也不记回退**，从覆盖率分母里消失。

**修复（自底向上）**：

1. `src/autosync/lyrics/language.py`：pure-Latin 行（无英文功能词者）由“不可路由”改为
   **路由到英文模型**并标注低置信度原因（不再静默丢弃）。
2. `src/autosync/models/funasr_zh.py`：`normalize_cjk` 为空时**抛错**（带明确原因），
   由 `segment.py` 记为 `failed` + 原因，而非返回空列表。
3. `benchmarks/line_coverage.py`：逐行核算，保证 **每个演唱行恰有一个结果**
   （`model` / 路由语言 / `not_routable`），并把它们**全部计入分母**。

**实测**：

| 歌曲 | 总行 | metadata | instrumental | 演唱行 | 路由 zh | 路由 en | 不可路由 | unaccounted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 提瓦特民谣 | 73 | 9 | 10 | 54 | 45 | **9** | 0 | **0** |
| 不安灵魂收容所 | 58 | 4 | 0 | 54 | 34 | 15 | **5**（拉丁主导混行，原因已记） | **0** |

- `提瓦特民谣` 的 9 条拉丁行**现在有明确去处**（英文模型），不再消失。
- `不安灵魂收容所` 有 5 条**拉丁主导中英混行**，`detect_line_language` 判定“不可靠分离” → 显式 `not_routable` + 原因，
  仍计入分母（54），**未从覆盖率中排除**。
- 回归测试：`tests/test_line_coverage.py`（合成分行、分母不变式、拉丁行路由到 EN、混行显式记录）
  与更新后的 `tests/test_language_routing.py`。

## 4. beta.4 模型验证

### 4.1 日文：beta.3 vs beta.4

| 口径 | beta.3 `attn_L0_5`（164.7 MB） | beta.4 `aL8`（129.4 MB） |
|---|---:|---:|
| A：Test 整混音 tune+val 起始 MAE | 13.1 ms | **9.5 ms** |
| A：连续性盲测（已用过） | 60.6 ms / >500 ms 3.7% | **38.7 ms / 1.4%** |
| A：GTSinger INT8-vs-FP32 | 12.27 ms | **8.45 ms** |
| **C：GTSinger 相对人工边界 绝对 MAE（INT8）** | **110.5 ms** | 123.8 ms |

**结论**：beta.4 在**一致性（A）**上确实更好且更小；但在**绝对精度（C）**上，12 段小样本显示
beta.3 反而略好（110.5 vs 123.8，而 FP32 基线 118.7 ms）。**两者差距在噪声量级**，
因此**不能宣称 beta.4 的逐字绝对精度提高**——只宣称“更小、与 FP32 更一致”。
逐字绝对精度仍为**未验收**状态。

### 4.2 英文：默认版 vs 精度变体（GTSinger 绝对）

| 模型 | 体积 | C：绝对 MAE（INT8） | C：绝对 MAE（FP32） | A：INT8-vs-FP32 |
|---|---:|---:|---:|---:|
| 默认 beta.2 | 122.4 MB | **99.4 ms** | 100.1 ms | 2.78 ms |
| 精度变体 `head_ffn` | 292.0 MB | 100.1 ms | 100.1 ms | 2.09 ms |

**结论**：在**唯一有人工边界**的语料上，292 MB 变体**没有绝对收益**；它的价值只在整混音一致性，
且会牺牲个别歌曲（V1.2 已记录 `若我不曾见过太阳` 40.3→65.0 ms）。→ **维持默认版**，变体仅作可选。

### 4.3 中文短句长尾的成因

- 长尾集中在**短行**（≤6 字平均最大误差 57 ms，≥10 字 14 ms）。
- `cif_wo_hidden` 闭式与顺序循环在 59 条真实行上**逐值一致** → **不是后处理**。
- 把 attention/FFN/CIF 头保留 FP32 反而更差（V1.1）→ **不是量化**。
- 短行音频窗口短 → 编码器帧数少 → CIF 峰值分辨率低，属**模型/窗口分辨率**问题。
  → 结论：**短行应保守回退整行 LRC**，而不是继续压量化。

### 4.4 Android 契约测试向量

`tools/make_contract_vectors.py` 生成**纯自研合成音频**（正弦谐波 + 慢包络，可随仓库分发，
不涉第三方版权），并记录输入规范、输出名/dtype/形状与数值探针：

| 语言 | 模型 | 输出 | 3 s 输入形状 |
|---|---|---|---|
| en | `wav2vec2-base-960h-int8.onnx` | `logits`, `log_probs` | `[1,149,32]` |
| ja | `hubert-ja-phoneme-ctc-int8.onnx` | `logits`, `log_probs` | `[1,149,48]` |
| zh | `fa-zh-int8.onnx` | `us_alphas`, `us_peaks` | `[1,150]` |

产物在 `android_contract/`（wav + `contract.json`），由 `tests/test_contract_and_manifest.py` 校验哈希与形状。

### 4.5 manifest provenance 审计

- 每个语言新增/核对 `accuracy.provenance`：指标对应的**模型版本 + 评测集**（A/B/C 明确区分）。
- beta.4 的日文精度块已改为对应 `aL8`（`model_recipe` 提到 layer 8），删除了对 beta.2 报告的过期引用。
- 新增 `tests/test_contract_and_manifest.py` 守卫：beta.4+ 必须有 provenance、JA 配方必须描述实际变体、
  资源 `sha256` 必须为 64 位十六进制且 URL 与文件名一致。
- 历史包（beta.1–beta.3）**保持原样**，不重写已发布的元数据。

## 5. 已知限制

- **口径 B 缺失**：4 首 AMLL 参考均未通过版本审计，逐字绝对 MAE 无法在真实整混音上给出。
  这是**证据不足**，不是“已完成”。
- GTSinger 是**清唱**且每种语言仅 12 段，样本小；其绝对 MAE 不能外推到整混音。
- 日文 beta.4 的绝对精度与 beta.3 在噪声内，**未验收**。
- `不安灵魂收容所` 的 5 条拉丁主导混行目前**显式回退**（不产出逐字），未实现按能力分段。
- 未做真机（Android/ARM）复测。

## 6. 复现

```bash
# AMLL 版本审计（含错误歌曲对照）
.venv/bin/python tools/amll_version_check.py \
  --ttml experiments/amll_reference/2750140001.ttml \
  --audio "Test/提瓦特民谣.mp3" --control-audio "Test/绘本.mp3" --max-offset 15

# 覆盖核算（每个演唱行都要有结果）
.venv/bin/python benchmarks/line_coverage.py --song 提瓦特民谣 --primary zh

# 契约向量
.venv/bin/python tools/make_contract_vectors.py --out android_contract

# 绝对精度（口径 C）
.venv/bin/python benchmarks/onnx_quant_eval.py --language ja \
  --pool /mnt/ntfs_files_sda4/MeloX-AutoSync-data/gtsinger_val
```

## 7. 交付清单

- 修复：`src/autosync/lyrics/language.py`、`src/autosync/models/funasr_zh.py`
- 工具：`tools/amll_version_check.py`（重写）、`tools/make_contract_vectors.py`、`benchmarks/line_coverage.py`
- 测试：`tests/test_amll_version_check.py`、`tests/test_line_coverage.py`、`tests/test_contract_and_manifest.py`、
  `tests/test_language_routing.py`（更新）
- 契约：`android_contract/`
- ruff 全通过、pytest 全通过。
