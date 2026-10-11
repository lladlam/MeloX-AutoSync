# INT8 量化配方对照实验（英文链路）

> 本文补充 `REPORT.md`。`REPORT.md` 由 `validate_onnx.py` 自动生成，本节是**代理交付之后**
> 追加的对照实验记录，注明来源以便复核。
>
> 背景：`REPORT.md` 的 "Alignment start MAE (INT8 vs FP32)" 实测 219.05 ms，远超任务书阈值
> 30 ms。本节用于回答一个问题：**这是配方可以改善的，还是动态 INT8 的固有代价？**

## 方法

复用 `validate_onnx.py` **自己的**对齐阶段（`_run_align_phase` + `phase_align_compare`，
`forced_align(blank=0)`，29 行真实歌词 / 168 个词 / 4 首本地歌曲），只替换 INT8 模型文件。
除量化配方外，比较路径、素材、fp32 参考缓存完全一致，因此数值可直接与报告中的 219.05 ms 对比。

## 结果

| # | 配方 | 起始 MAE | 结束 MAE | 文件字节 |
|---|---|---:|---:|---:|
| 0 | 交付版：仅 `MatMul/Gemm`，per-tensor | 219.05 ms | 181.79 ms | 121,954,710 |
| 1 | 0 + 排除 CTC 头 `lm_head/MatMul` | 217.74 ms | 186.07 ms | 122,0xx,xxx |
| 2 | 0 + `ActivationSymmetric=True` | 121.19 ms | 129.88 ms | 122,373,796 |
| 3 | **0 + `per_channel=True`（采用）** | **121.19 ms** | **129.88 ms** | **122,373,796** |
| 4 | 变体 1 + `per_channel=True` | 121.19 ms | 129.76 ms | 122,446,070 |
| 5 | 默认算子集合（**连 Conv/Gather 一起量化**） | 425.36 ms | 420.71 ms | — |
| 6 | 变体 5 + `per_channel=True` | 417.50 ms | 434.05 ms | — |

逐曲分解（起始 MAE，ms）：

| 配方 | S1 (54 词) | S3 (50 词) | S4 (15 词) | S6 (49 词) |
|---|---:|---:|---:|---:|
| 0 交付版 | 77.4 | 354.0 | 281.3 | 218.4 |
| 3 per-channel（采用） | 35.2 | 207.2 | 265.3 | 84.1 |

## 结论

1. **`per_channel=True` 是唯一有效杠杆**：起始 MAE 219.05 → 121.19 ms（**-45%**），
   逐曲全部改善，代价仅 **+0.42 MB**。已写入 `export_model.py` 并重新导出。
2. **排除 CTC 头无效**（217.74 vs 219.05 ms）：偏移不来自 logits 头。
3. **`ActivationSymmetric` 无效**：与 per-channel 数值完全相同。
4. **卷积前端必须保持 fp32**：把 Conv/Gather 一起量化后指标**翻倍恶化**（219 → 425 ms）。
   这条验证了任务书 §3 的约束是必要的，不是保守。
5. 差异不是推理噪声：交付版与各变体在 **1 线程与 4 线程下都是逐位确定的**
   （同一输入连跑 4 次哈希一致，`max|Δ|=0`）。
6. **动态 INT8 无法达到 30 ms**：即使最优配方仍为 121 ms。因此：
   - 若 Android 端要**逐字/逐词**精度，动态 INT8 不足；建议保留 FP32 路径，或
   - 改用**静态 QDQ + 校准**（用真实歌曲/歌声做校准集）并**用 AMLL 人工参考复测绝对误差**，或
   - 把 INT8 定位为**行级/短语级**时间轴（现有质量门控与 LRC 回退已能兜底）。

> 说明：本表测量的是 **INT8 与 FP32 的一致性**，不是绝对准确率。121 ms 的"不一致"不等于
> 121 ms 的绝对误差；要判断对用户的实际影响，需要用人工参考打轴（见 `docs/AMLL_COMPARISON_REPORT.md`
> 的流程）分别评测 FP32 与 INT8 的绝对误差。

## 复现

```bash
# 重新量化（采用配方）
.venv/bin/python android_export/en/export_model.py --stage quantize --force
# 重跑完整校验并刷新 REPORT.md
.venv/bin/python android_export/en/validate_onnx.py --phase all
```
