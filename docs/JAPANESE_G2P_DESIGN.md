# 日语 G2P 与原文映射设计（JAPANESE_G2P_DESIGN）

> 代码：`src/autosync/lyrics/ja_pronunciation.py`（`JapanesePronunciationMapper`）
> 回归测试：`tests/test_ja_mapper.py`

## 1. 目标与要求

音素级对齐器（pyshiro / 音素 CTC）都工作在**音素**层，但用户要的是**原始日文歌词**。
映射器必须：

1. 支持汉字、平假名、片假名；促音（っ）、长音（ー）、拨音（ん）、拗音（きゃ）。
2. 建立「音素 ↔ 原文文字」的区间映射。
3. 支持特殊读音覆盖（overrides）。
4. 无法确认读音时标记不确定性。
5. **原始歌词必须能无损重建**。
6. **不允许**把「一个汉字 = 一个音节」。

## 2. 处理流程

```
原始歌词 text
  │  gettext.overrides（特殊读音）
  ▼
NFKC 归一化 → pyopenjtalk.g2p(..., kana=True) → 片假名读音 → 平假名
  │  kana reading
  ▼
DP：把 kana 读音对齐到「原始字符」（汉字≥1 モーラ，假名 1:1，其他 0）
  │  per-char kana span [a,b)
  ▼
前缀音素转换：phonemes(kana[:i])（去掉首尾 pau）
  │
  ▼
每个原文文字获得其音素列表（= prefixes[b] − prefixes[a]）
```

返回：`list[CharReading(char, kana, phonemes, uncertain)]` 与完整音素序列（含首尾 `pau`）。

## 3. 关键设计点

### 3.1 汉字 → 多个モーラ（禁止一字一音节）

DP 允许一个汉字消费 **1..12** 个假名。例如 `最後`（さいご）：
`最`→`さ`（1 モーラ）、`後`→`いご`（2 モーラ）。每个汉字得到其真实读音跨度。

### 3.2 假名 1:1，含助词音变

假名原文与其读音基本 1:1，但助词会发生音变（`は`→`わ`）。DP 对假名按「1 个假名 ↔ 1 个读音假名」匹配，
字符不同只记小代价，因此 `は→わ` 仍能正确落位（测试 `test_align_kana_to_chars_handles_particle_ha`）。

### 3.3 促音 / 长音 / 拗音 / 拨音

- 促音 `っ`：闭音 → 音素 `cl`（pyshiro）或 `cl`（HuBERT 词表），独立单元。
- 长音 `ー`、`う`：由前缀转换自然产生 `O`（pyshiro）/ 重复元音（pyopenjtalk）。
- 拨音 `ん`：`N`。
- 拗音 `きゃ`：读音为 `きゃ`（2 假名），音素 `ky a`。

### 3.4 不确定性（不伪造）

标记 `uncertain=True` 的情形：

- G2P 后端缺失或抛错 → 全部字符不确定，音素列表为空；
- DP 无法把读音对齐到原文（如假名与读音完全不匹配）；
- 某字符**归不到任何音素**（典型：拗音的第二个假名 `ょ` 被并入 `ky`，无法再精确切分）；
- 拉丁/数字/标点等无读音字符（分配到 0 个音素）。

测试 `test_youon_second_mora_marked_uncertain` 覆盖拗音情形。

### 3.5 无损重建与覆盖

- `map()` 返回的字符**始终是原始文本字符**（不做 NFKC 替换），`reconstruct()` 恒等于原文；
  overrides 只影响**读音**，不影响返回文本（测试 `test_reconstruction_is_lossless`、`test_overrides_change_reading`）。
- overrides 支持「按最长匹配」替换读音，用于歌词中的特殊唱读/读法不唯一的词。

## 4. 与对齐器的接口

- pyshiro 适配器（`models/pyshiro_aligner.py`）：用默认后端（pyshiro `kana2phonemes.table`）。
- 音素 CTC 适配器（`models/phoneme_ctc_aligner.py`）：注入 pyopenjtalk 音素转换后端，
  使音素集合与 HuBERT 词表一致。

映射器后端（`table` / `g2p` / `convert`）全部可注入，测试无需真实依赖。

## 5. 局限（诚实）

- pyopenjtalk 使用通用日语词典，**歌唱中常见的特殊读法**未必正确；可通过 overrides 人工修正。
- 同形异读（如「行」いく/ぎょう）依赖上下文，个别歌词可能读错；目前不报告音节级置信度，
  仅报告「能否归到音素」的不确定性。
- 拗音内部（きょ 的 ょ）无法精确切分，已标 `uncertain`，不假装精确。
- 拉丁/数字字符音译依赖 pyopenjtalk，可能被跳过（标 `uncertain`）。
