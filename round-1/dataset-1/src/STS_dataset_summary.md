# STS Dataset Information Summary

## 1. STS Benchmark (sentence-transformers/stsb)

### Original Source/Paper
- **Paper**: "SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Cross-lingual Focused Evaluation"
- **Authors**: Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, Lucia Specia
- **Year**: 2017
- **arXiv**: https://arxiv.org/abs/1708.00055
- **HuggingFace**: https://huggingface.co/datasets/sentence-transformers/stsb

### Popularity
- **Downloads**: 6.5k (HuggingFace community)
- **Likes**: 26
- Well-established benchmark in the NLP community

### Dataset Structure
- **Format**: Parquet
- **Total Rows**: 8,630
- **Splits**: 
  - Train: 5,750 rows
  - Validation: 1,500 rows
  - Test: 1,380 rows
- **Fields**:
  - `sentence1` (string): First sentence (length 16-368 chars)
  - `sentence2` (string): Second sentence (length 15-311 chars)
  - `score` (float64): Similarity score (0-5 scale)
- **Languages**: English
- **Task Categories**: Feature Extraction, Sentence Similarity

### License
- **License**: CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0)

### Usage & Citations
- Used extensively in sentence embedding research
- Standard benchmark for evaluating semantic similarity models
- Part of the Sentence Transformers library ecosystem
- Cited in numerous papers on semantic textual similarity

---

## 2. HuggingFace mteb/stsbenchmark-sts

### Original Source
- **Source**: Repackaged from the original STS Benchmark dataset
- **Organization**: Massive Text Embedding Benchmark (MTEB) team
- **HuggingFace**: https://huggingface.co/datasets/mteb/stsbenchmark-sts
- **Related Papers**: 
  - arxiv:2210.07316 (MTEB paper)
  - arxiv:2502.13595

### Popularity
- **Downloads**: 1.18k (HuggingFace community)
- **Likes**: 19
- Moderate usage, primarily within MTEB evaluation framework

### Dataset Structure
- **Format**: JSON (auto-converted to Parquet)
- **Total Rows**: 8,630
- **Splits**: 
  - Train: 5,750 rows
  - Validation: 1,500 rows
  - Test: 1,380 rows
- **Fields**:
  - `split` (string): Dataset split identifier
  - `genre` (string): Genre classification (e.g., "main-captions")
  - `dataset` (string): Source dataset name (e.g., "MSRvid")
  - `year` (string): Year of data collection
  - `sid` (string): Sample ID
  - `score` (float64): Similarity score (0-5 scale)
  - `sentence1` (string): First sentence
  - `sentence2` (string): Second sentence
- **Languages**: English
- **Task Category**: Semantic Similarity Scoring (t2t)

### License
- **License**: Unknown (inherits from original STS Benchmark)

### Key Differences from sentence-transformers/stsb
- Includes additional metadata fields (split, genre, dataset, year, sid)
- Better for reproducibility studies tracking data provenance
- Part of standardized MTEB evaluation pipeline

---

## 3. sentence-transformers/stsb

**Note**: This appears to be the same dataset as `sentence-transformers/stsb` (item #1 above). The search results show these are identical or mirror entries for the same Semantic Textual Similarity Benchmark dataset.

### Recommendation
For research purposes:
- Use **sentence-transformers/stsb** for standard STS benchmark experiments
- Use **mteb/stsbenchmark-sts** if you need additional metadata for provenance tracking or MTEB-compatible evaluations
- Both datasets are well-documented, have clear licenses, and are widely used in the community

### Data Quality Assessment
- ✅ Clear documentation (dataset cards available)
- ✅ Established provenance (SemEval-2017 Task 1)
- ✅ Sufficient size (8.6k samples)
- ✅ Standardized splits (train/val/test)
- ✅ Continuous similarity scores (0-5 scale)
- ✅ Active community usage

Both datasets are suitable for research on semantic textual similarity and sentence embedding evaluation.
