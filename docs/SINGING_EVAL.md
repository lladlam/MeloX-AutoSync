# Singing evaluation datasets (SINGING_EVAL)

This document records the singing datasets we investigated for V0.2, their
licences, annotation quality, what we actually used, and how to reproduce the
evaluation. It also documents why some datasets were **not** used.

## 1. Dataset survey

| Dataset | Language(s) | Size | Annotations | Licence | Verdict |
|---|---|---|---|---|---|
| **GTSinger** | 9 langs incl. zh/ja/en | 80.6 h, 54.2 GB | Praat TextGrid + per-segment JSON: word & phoneme boundaries, note/pitch, technique flags (vibrato/glissando/breathy…) | **CC BY-NC-SA 4.0** | ✅ **USED** (curated subset) |
| **PJS** | Japanese | phoneme-balanced | phoneme-balanced singing+speech samples | CC BY-SA 4.0 | ❌ not used — isolated phonemes/utterances, not sung lyric phrases; distribution via Google Drive sign-in |
| **OpenCpop** | Chinese | 100 songs, 5.2 h | syllable/word/phoneme boundaries, pitch | **CC BY-NC-ND 4.0** | ❌ not used — *NoDerivatives* clause makes derived alignment outputs legally awkward |
| **M4Singer** | Chinese | 29.8 h, 20 singers | performance MIDI + phoneme alignment | free-to-use (NC) | ❌ not used — GTSinger already covers Chinese with simpler access |
| **MIR-1K** | Chinese | 1000 clips | pitch, lyrics, vocal/non-vocal; **vocal & accompaniment in separate channels** | research use (Zenodo record 3532216) | ❌ not used — public download link (`mirlab.org/.../MIR-1K.rar`) returns **404**; no accessible mirror; licence unclear on Zenodo |

### Notes on licence quality

* **GTSinger** embeds a `dataset_license.md` in the repo stating CC BY-NC-SA 4.0
  (non-commercial, share-alike). We therefore (a) use it only for research,
  (b) never redistribute its raw audio or annotations, and (c) publish only
  aggregate metrics.
* **OpenCpop** is CC BY-NC-**ND** — the "NoDerivatives" term is the reason we
  did not build a derived alignment benchmark on it.
* **PJS** is CC BY-SA 4.0 (permissive-ish) but its content is phoneme-balanced
  *singing syllables*, not connected lyrics, so it cannot validate lyric timing.
* **MIR-1K** would be ideal for the accompaniment experiment, but it is not
  practically obtainable here.

## 2. What we actually used

Curated **42 GTSinger segments** (a-cappella, so the reference is exact):

| Language | Singer | Techniques | Segments |
|---|---|---|---|
| English | EN-Alto-1 | Vibrato, Control | 6 |
| Japanese | JA-Soprano-1 | Vibrato, Glissando, Control | 18 |
| Chinese | ZH-Alto-1 | Vibrato, Glissando, Control | 18 |

Reference granularity: **word-level** for English (`edelweiss`), **character-level**
for Chinese and Japanese (GTSinger annotates CJK per character).

Singing phenomena covered:

* **vibrato** – `*_Vibrato_Group` segments (verified: `vibrato` flags = 1 in the JSON)
* **legato / portamento** – `*_Glissando_Group`
* **held notes** – long character spans (e.g. Chinese `岛` 4.39→6.34 s, Japanese `ス` 4.34→7.08 s)
* **control** – `Control_Group` (normal sustained singing)

We do **not** claim coverage of rap/fast runs or heavy vibrato rates beyond what
these segments contain.

## 3. How to reproduce

```bash
# 1. Download the curated subset (accepts GTSinger licence by doing so).
uv run python tools/fetch_gtsinger.py --out experiments/singing/gtsinger --max-per-group 3

# 2. Run the singing benchmark per language.
uv run python benchmarks/singing_eval.py --languages en --out experiments/reports/singing_en.json
uv run python benchmarks/singing_eval.py --languages ja --out experiments/reports/singing_ja.json
uv run --extra funasr python benchmarks/singing_eval.py --languages zh --out experiments/reports/singing_zh.json

# 3. Accompaniment robustness (controlled synthetic bed).
uv run --extra funasr python benchmarks/accompaniment_eval.py --language zh --out experiments/reports/accompaniment_zh.json
```

The downloaded audio/annotations live under `experiments/singing/` which is
**gitignored**. Only `experiments/reports/*.json` (aggregate metrics) are
committed.

## 4. Licence handling / non-redistribution

* No GTSinger (or any third-party) audio is committed to this repository.
* `tools/fetch_gtsinger.py` only downloads; it does not republish.
* Derived evaluation sets (LRC + reference JSON) are generated locally and are
  gitignored because the underlying lyrics are third-party copyright.
* Any user running these tools is responsible for accepting GTSinger's licence.
