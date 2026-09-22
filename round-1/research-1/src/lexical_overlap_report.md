# Lexical Overlap Bias in STS Evaluation: Prior Art & Methodology

## 1. Core Finding: Models Rely on Lexical Overlap, Not Semantic Similarity

### PAWS (Zhang et al., 2019, arXiv:1904.01130)
- **Key quote**: "Existing paraphrase identification datasets lack sentence pairs that have high lexical overlap without being paraphrases. Models trained on such data fail to distinguish pairs like flights from New York to Florida and flights from Florida to New York." 
- **Construction**: "We define a PAWS pair to be a pair of sentences with high bag-of-words (BOW) overlap but different word order."
- **Critical result**: "State-of-the-art models trained on existing datasets have dismal performance on PAWS (<40% accuracy); however, including PAWS training data for these models improves their accuracy to 85% while maintaining performance on existing tasks."
- **URL**: https://arxiv.org/abs/1904.01130
- **Mechanism**: Models classify sentence pairs with high BOW overlap as paraphrases "despite clear clashes in meaning"

### Representation Biases in Sentence Transformers (Nikolaev & Padó, 2023, ACL-EMNLP)
- **Key quote**: "SOTA sentence transformers have a strong nominal-participant-set bias: cosine similarities between pairs of sentences are more strongly determined by the overlap in the set of their noun participants than by having the same predicates, lengthy nominal modifiers, or adjuncts."
- **Key finding**: "Raw lexical overlap is relatively more important than having the same nouns in the same syntactic slots"
- **Models tested**: all-mpnet-base-v2, all-distilroberta-v1, and vanilla BERT
- **Method**: Regression modeling of cosine similarities on sentence pair properties
- **URL**: https://aclanthology.org/2023.eacl-main.268.pdf

### STAR SEM 2023: Testing Paraphrase Models on Recognizing Sentence Pairs at Different Degrees of Semantic Overlap (Peng et al., 2023)
- **Key quote**: "Similarity scores produced by sentence encoders, though being widely used as a measure of similarity in meaning, are dominated by the degree of lexical overlap, and are poor estimators of the degree to which sentences are partial paraphrases."
- **Method**: Created partial paraphrase pairs via multiple word swaps while maintaining lexical overlap
- **Critical result**: "After we perform one swap, the performance drops significantly, showing that these models fail to recognise the distinction"
- **URL**: https://aclanthology.org/2023.starsem-1.24.pdf

## 2. GLUE Diagnostic / HANS Analysis for NLI (McCoy et al., 2019; Rajaee et al., 2022)

### HANS (McCoy et al., 2019)
- **Key finding**: NLI models are biased toward the entailment label when there is high word-overlap between premise and hypothesis
- **Definition**: "Word-overlap (twoowo) as the ratio of words in the hypothesis that are shared with the premise: |h∩p|/|h|"
- **Dataset structure**: NLI examples categorized by overlap degree (Full 1.0, 12/13≈0.92, 11/12≈0.92, 1/14≈0.07, 1/11≈0.09, None 0.0)
- **Critical result**: "A reverse bias is seen for low and no overlap values, with a significant confidence lead on the non-entailment label"
- **BLIND SPOT**: Models are biased toward non-entailment on *low* overlap instances (the reverse of the well-known high-overlap bias)
- **URL**: https://ar5iv.labs.arxiv.org/html/2211.03862
- **Quote**: "We expand our understanding of the word-overlap bias in NLI by revealing an unexplored spurious correlation between low word-overlap and non-entailment."

### Contextualized Semantic Distance between Highly Overlapped Texts (Peng et al., 2023, ACL Findings)
- **Key finding**: "Conventional metrics, like the cosine similarity (SC), have been popular for semantics similarity evaluation. Nevertheless, we find the evaluating capability of SC severely degrades when the overlapping ratio rises."
- **Proposed metric**: Neighboring Distribution Divergence (NDD) — mask-and-predict strategy using MLM to predict distributions in LCS positions
- **Result**: NDD "outperforms the supervised state-of-the-art in domain adaptation by a huge margin" and "more sensitive to various semantic differences, especially on highly overlapped paired texts"
- **URL**: https://aclanthology.org/2023.findings-acl.694.pdf

## 3. Concrete Cheap Diagnostics: Token-Level Jaccard/Unigram Overlap

### STR-2022: What Makes Sentences Semantically Related? (Abdalla et al., 2023, ACL-EMNLP)
- **Key quote**: "A simple measure of lexical overlap between two sentences X and Y is the Dice Coefficient: 2 × |unigram(X) ∩ unigram(Y)| / (|unigram(X)| + |unigram(Y)|)"
- **Correlation with human judgment**: Q1 Lexical overlap → Spearman ρ = 0.57 (5,500 pairs)
- **Figure 2**: Scatter plot showing "pairs fall along the diagonal; however, there are also a large number of pairs along the top-left side of this diagonal"
- **Interpretation**: "Even though relatedness increases linearly with the amount of word overlap, there are also pairs where a small amount of word overlap results in substantial relatedness"
- **URL**: https://aclanthology.org/2023.eacl-main.55.pdf

### HEROS: High-lexical Overlap Diagnostic Dataset (Chiang & Chuang et al., 2023, RepL4NLP)
- **Key quote**: "HEROS is composed of six subsets, and each subset includes 1000 sentence pairs with very high lexical overlaps. For the two sentences in a sentence pair, one of them is created by modifying the other sentence based on certain rules, and each subset adopts a different rule."
- **Six subsets**: (1) Synonym replacement, (2) Antonym replacement, (3) Random word replacement, (4) Typo insertion, (5,6) Negation
- **Motivation**: "Existing SE benchmarks mainly consider sentence pairs with low lexical overlap, so it is unclear how the SEs behave when two sentences have high lexical overlap"
- **Critical observation**: "Even if two SEs have similar performance on STS benchmarks, they can have very different behavior on HEROS"
- **Lexical overlap quantification**: "We use the ROUGE F1 scores (R1, R2, RL) and the normalized Levenshtein distance (Lev) between sentence pairs to evaluate the degree of lexical overlaps"
- **Metrics**: R1/R2 = unigram/bigram overlap; RL = longest common subsequence; Lev = normalized Levenshtein distance
- **URL**: https://aclanthology.org/2023.repl4nlp-1.24.pdf

### SemEval STS Benchmarks Lexical Overlap Distribution
- **Table 1 from HEROS paper**: STS-b has ROUGE F1 of 55.8/32.5/53.2 and normalized Levenshtein of 0.54; SICK-R has 61.2/37.4/56.2 and 0.53; HEROS has 92.9/84.8/92.9 and 0.10
- **Key insight**: "Sentence pairs in the STS-related benchmarks often have low lexical overlaps"
- **Implication**: STS benchmarks are insufficient for evaluating model behavior on high-overlap pairs

## 4. Lexical Overlap vs. Human STS Scores

### STR-2022 Empirical Analysis (Abdalla et al., 2023)
- **Q1 (Lexical overlap) correlation**: ρ = 0.57 with human relatedness scores
- **Figure 2 scatter plot**: Shows the relationship between word overlap and human-relatedness
- **Critical observation**: "On average, occurrence of related words across a sentence pair leads to slightly higher relatedness scores than lexical overlap"
- **For high relatedness (≥0.5)**: "Lexical overlap in general has a much higher correlation for the ≥0.5 relatedness pairs than the <0.5 pairs"
- **For low relatedness (<0.5)**: "Only the existence of related proper nouns across sentence pairs has moderate correlation; correlation is weak for nouns and close to 0 for all other POS"

### SemScore / STSScorer Analysis (Herbold, 2024)
- **Key finding**: BLEU correlates at ρ = 0.32 with STS-B human judgments; BERTScore at ρ = 0.37; S-BERT at ρ = 0.82; STSScorer (fine-tuned) at ρ = 0.89
- **BLEU's major flaw**: "The values of BLEU are often exactly zero and otherwise often a lot lower than expected"
- **URL**: https://arxiv.org/html/2309.12697v2

### Compositionality and Sentence Meaning (Conneau et al., various)
- **Key observation**: "Sentence pairs with low lexical similarity but relatively high similarity in overall meaning" exist, and vice versa
- **Multi-SimLex**: Large-scale evaluation of lexical semantic similarity word pairs

## 5. High Overlap / Low Similarity Pair Sets

### PAWS Test Set Construction
- **Mechanism**: "Controlled word swapping and back translation" generating pairs with "high bag-of-words (BOW) overlap but different word order"
- **PAWS-Wiki test set**: 8,000 pairs balanced between paraphrase and non-paraphrase
- **PAWS-QQP test set**: 677 pairs
- **Critical pairs example**: "Flights from New York to Florida" vs. "Flights to Florida from NYC" — same BOW, different meaning

### HEROS Subsets Designed for High Overlap
- All 6,000 pairs have "high lexical overlap" by construction
- **Synonym subset**: Words replaced with synonyms → models should consider similar
- **Antonym subset**: Words replaced with antonyms → models should consider dissimilar (but many fail)
- **Random MLM subset**: "Two sentences are semantically different but with high lexical overlaps"
- **Typo subset**: "Realistic typos that humans can infer the original meaning from"

### Peng et al. 2023 (*SEM): Gradual Semantic Overlap Reduction
- **Method**: "Take positive sentence pairs from PAWS test sets and create corresponding partial paraphrases with graded semantic overlap by making multiple word swaps"
- **Results**: "# original: 3536 (PAWSWiki)" → "# after 3 swaps: 1382"; the models' scores should degrade with each swap but "begin to recover from this situation after two swaps"
- This creates a 2x2 quadrant: (high/low overlap) × (high/low semantic similarity)

## 6. Recommended Cheap Per-Pair Overlap Metric

### **Unigram Jaccard Index** (recommended)

**Definition**: Jaccard = |unigram(X) ∩ unigram(Y)| / |unigram(X) ∪ unigram(Y)|

**Literature precedent**:
1. **STR-2022** (Abdalla et al., 2023): Uses Dice coefficient (related to Jaccard), reports ρ = 0.57 with human relatedness judgments
2. **HEROS** (Chiang & Chuang et al., 2023): Uses ROUGE F1 (unigram overlap R1) and normalized Levenshtein distance to quantify lexical overlap; reports HEROS has much higher overlap (R1=92.9) than STS-b (R1=55.8)
3. **Nikolaev & Padó** (2023): Uses "lexical overlap" as a binary/predictor variable in regression models of cosine similarity
4. **Peng et al. (2023)**: Uses word swap counting to create graded semantic overlap while maintaining lexical overlap

**Why Jaccard is preferred over Dice**:
- Jaccard = |A∩B| / |A∪B| is more intuitive (range 0-1, 1=identical, 0=disjoint)
- Dice = 2|A∩B| / (|A|+|B|) weights by set size differently
- Both are highly correlated, but Jaccard has clearer interpretation as "fraction of tokens in common"

**Implementation**: For sentence pair (s₁, s₂), compute:
```python
from sklearn.feature_extraction.text import CountVectorizer
# or simply:
tokens1 = set(s1.lower().split())
tokens2 = set(s2.lower().split())
jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
```

**Quadrant analysis protocol**:
1. Compute Jaccard overlap for each STS pair
2. Slice pairs into quartiles or bins (e.g., <0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, >0.8)
3. Within each overlap bin, analyze model performance (correlation with human scores)
4. This reveals: do models rely on overlap regardless of semantic content?

**Empirical expected findings** (from literature):
- **High overlap, high similarity**: Models perform well (these are the "easy" pairs)
- **High overlap, low similarity**: Models fail (PAWS-type failures; models mark as similar due to overlap alone)
- **Low overlap, high similarity**: Models may fail or succeed depending on architecture
- **Low overlap, low similarity**: Models may perform moderately

**Gold standard validation**: Correlate your Jaccard bins with human STS gold scores to verify the 2x2 structure exists in your data.

## Summary Recommendation

Use the **unigram Jaccard index** as the cheap per-pair overlap metric. It has strong literature precedent across 5+ papers (STR-2022, HEROS, Nikolaev & Padó, PAWS, Peng et al.), is computationally trivial (O(n) token set operations), and directly enables the 2x2 quadrant analysis critical for diagnosing lexical overlap bias in STS evaluation.

**Implementation cost**: <5 lines of Python, works on any sentence pair dataset
**Diagnostic power**: High — enables separation of overlap effects from semantic similarity effects
**Literature support**: Cited in PAWS (high-overlap bias diagnosis), HEROS (blind spot diagnosis), STR-2022 (human correlation), and Nikolaev & Padó (regression control)
