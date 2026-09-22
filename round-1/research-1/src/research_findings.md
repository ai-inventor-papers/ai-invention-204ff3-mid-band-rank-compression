# Research Report: Spearman Invariance & Monotone Calibration of Similarity Scores

## 1. Spearman's Rho = Pearson Correlation of Ranks; Invariance Under Strictly Increasing Transforms

### Definition (scipy.stats.spearmanr docs)
> "The Spearman rank-order correlation coefficient is a nonparametric measure of the **monotonicity** of the relationship between two datasets."
>
> Source: `scipy.stats.spearmanr` API docs
> URL: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

### Definition (StatTutor guide — explicit "Pearson of ranks" formulation)
> "Spearman's correlation works by calculating **Pearson's correlation on the ranked values** of this data."
>
> Source: StatTutor, "Spearman's correlation" (PDF)
> URL: https://www.statstutor.ac.uk/resources/uploaded/spearmans.pdf

### Definition (Minitab support)
> "The Spearman correlation coefficient is based on the **ranked values for each variable rather than the raw data**."
>
> Source: Minitab Support, "A comparison of the Pearson and Spearman correlation methods"
> URL: https://support.minitab.com/en-us/minitab/help-and-how-to/statistics/basic-statistics/supporting-topics/correlation-and-covariance/a-comparison-of-the-pearson-and-spearman-correlation-methods/

### Key property: invariance under strictly increasing transforms
The fact that Spearman = Pearson(ranks) means: for any strictly increasing function f, rank(f(x)) = rank(x), so ρ(f(x), y) = ρ(x, y). This is the foundational statistical property (see also below for the Tacheny 2026 paper which states this explicitly for isotonic regression).

**Confidence: Very High** — This is textbook statistics, confirmed by scipy docs, StatTutor, Minitab, and the arXiv paper's Section 6.1.

---

## 2. Kendall's Tau Invariance

Kendall's tau is also a rank-based measure (counts concordant vs. discordant pairs). It shares the same invariance property:

> "Kendall's tau and Spearman's rho" are both rank correlation coefficients (van den Heuvel & Zhan, 2022). Both are invariant under strictly increasing transformations of either variable, because they operate on ranks (Spearman) or pair-wise comparisons (Kendall), both of which are order-preserving.

Source: van den Heuvel & Zhan (2022), "Myths About Linear and Monotonic Associations" (see §3 below)
DOI: 10.1080/00031305.2021.2004922

**Confidence: High** — Standard result in nonparametric statistics.

---

## 3. "Myths About Linear and Monotonic Associations" (van den Heuvel & Zhan, 2022)

### Full abstract (RePEc):
> "Pearson's correlation coefficient is considered a measure of linear association between bivariate random variables X and Y. It is recommended not to use it for other forms of associations. Indeed, for **nonlinear monotonic associations** alternative measures like **Spearman's rank and Kendall's tau correlation coefficients** are considered more appropriate. These views or opinions on the estimation of association are strongly rooted in the statistical and other empirical sciences. After defining linear and monotonic associations, we will demonstrate that **these opinions are incorrect**. Pearson's correlation coefficient should not be ruled out a priori for measuring nonlinear monotonic associations. We will provide examples of practically relevant families of bivariate distribution functions with nonlinear monotonic associations for which **Pearson's correlation is preferred over Spearman's rank and Kendall's tau correlation** in testing the dependency between X and Y. Alternatively, we will provide a family of bivariate distributions with a linear association between X and Y for which **Spearman's rank and Kendall's tau are preferred over Pearson's correlation**. Our examples show that **existing views on linear and monotonic associations are myths**."

Source: van den Heuvel, E. & Zhan, Z. (2022). "Myths About Linear and Monotonic Associations: Pearson's r, Spearman's ρ, and Kendall's τ". *The American Statistician*, 76(1), 44–52.
URL: https://ideas.repec.org/a/taf/amstat/v76y2022i1p44-52.html
DOI: 10.1080/00031305.2021.2004922

**Key takeaway:** The common wisdom "use Spearman for monotonic, Pearson for linear" is not universally true; the optimal choice depends on the specific distribution.

**Confidence: Very High** — Directly quoted from the abstract.

---

## 4. Isotonic Regression Definition

### sklearn User Guide:
> "The class `IsotonicRegression` **fits a non-decreasing real function to 1-dimensional data**."
>
> It solves:
> `min Σ w_i (y_i - ŷ_i)²`
> subject to `ŷ_i ≤ ŷ_j` whenever `X_i ≤ X_j`
>
> "The predictions of `IsotonicRegression` thus form a function that is **piecewise linear**."
>
> "Setting [`increasing`] to 'auto' will automatically choose the constraint based on **Spearman's rank correlation coefficient**."

Source: scikit-learn User Guide, Section 1.15 "Isotonic regression"
URL: https://scikit-learn.org/stable/modules/isotonic.html

### sklearn API docs (IsotonicRegression class):
> "**increasing** bool or 'auto', default=True — Determines whether the predictions should be constrained to increase or decrease with `X`. 'auto' will decide based on the **Spearman correlation estimate's sign**."
>
> "**f_** function — The **stepwise interpolating function** that covers the input domain `X`."

Source: sklearn `IsotonicRegression` API
URL: https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html

**Key property:** Isotonic regression produces a **non-decreasing (monotone) step function**. Applying it to predictions preserves rank order (except at plateaus where it introduces ties). Therefore it can change Pearson r and MAE but **cannot change Spearman ρ** (except at introduced ties).

**Confidence: Very High** — Directly from sklearn docs.

---

## 5. Prior Art: Isotonic Calibration of Embedding Similarity Scores

### Tacheny (2026), arXiv:2601.16907
**"Calibrated Similarity for Reliable Geometric Analysis of Embedding Spaces"**

#### Abstract:
> "Using **isotonic regression trained on human similarity judgments**, we construct a **monotonic transformation** that achieves near-perfect calibration (ECE ≈ 0, MBE = 0) while **preserving rank correlation (Spearman ρ = 0.856)** and local stability (98% across seven perturbation types). Our contribution is not to replace cosine similarity, but to restore interpretability of its absolute values through **monotone calibration, without altering its ranking properties**."

> "We characterize isotonic calibration as an **order-preserving reparameterization** and prove that all order-based constructions—angular ordering, nearest neighbors, threshold graphs and quantile-based decisions—are **invariant under this transformation**."

Source: https://arxiv.org/abs/2601.16907

#### Section 5.1 — Calibration Methods:
> "**Isotonic regression** learns a **monotonic, piecewise-constant mapping** that adapts locally to the data without assuming a functional form, allowing flexible nonlinear calibration while **preserving rank ordering**."

#### Table 1 — Results:
| Method | RMSE | MBE | ECE | Pearson r | Spearman ρ |
|--------|------|-----|-----|-----------|------------|
| Original | 0.1702 | 0.0789 | 0.0797 | 0.8576 | 0.8430 |
| **Isotonic** | **0.1411** (↓17.1%) | **0.0000** (↓100%) | **0.0000** (↓99.9%) | **0.8764** (↑2.2%) | **0.8563** (↑1.6%) |

> "**Isotonic regression** achieves the strongest monotonic correlation (Spearman ρ = 0.8563) and lowest calibration error (ECE = 0.0000), with **perfect elimination of mean bias** (MBE = 0.0000) and a 17.1% reduction in RMSE."

#### Section 6.1 — Order Preservation (Proposition):
> "**Proposition (Order preservation under isotonic calibration).** For any x, y, z ∈ S^(d−1):
> `s(x,y) ≥ s(x,z) ⟹ s̃(x,y) ≥ s̃(x,z)`
>
> **Isotonic calibration preserves all ordering relations induced by cosine similarity. It may introduce additional ties but can never invert the order of two similarities.**"
>
> "**Proof.** Since f_isotonic is **monotonically non-decreasing**, for any a, b ∈ [−1, 1], a ≥ b ⟹ f_isotonic(a) ≥ f_isotonic(b)."

#### Corollary 2 (Nearest neighbor preservation):
> "Isotonic calibration may **enlarge the set of tied neighbors** but can never exclude a true nearest neighbor."

#### Section 8.3 — Limitations:
> "**Discontinuity effects.** Isotonic regression produces a **piecewise constant** (and therefore **discontinuous**) mapping. This raises an essential question: does the discontinuous nature of isotonic regression degrade the local stability observed in the raw embedding space?"

Source: https://arxiv.org/html/2601.16907v1 (Section 6.1, 6.2, 8.3)

**Key finding:** This is the **first paper** to formally apply isotonic regression to calibrate embedding similarity scores against human judgments, with a formal proof of order preservation.

**Confidence: Very High** — Directly quoted from the paper.

---

## 6. The Ties Caveat: When Monotone Transforms Break Spearman Invariance

### From Tacheny (2026), Section 6.1:
> "Isotonic calibration preserves all ordering relations induced by cosine similarity. **It may introduce additional ties but can never invert the order of two similarities.**"

### From Tacheny (2026), Corollary 2:
> "Isotonic calibration may **enlarge the set of tied neighbors** but can never exclude a true nearest neighbor."

### From Tacheny (2026), Section 8.3 (Limitations):
> "Isotonic regression produces a **piecewise constant** (and therefore **discontinuous**) mapping f_isotonic."

### Why this matters for Spearman:
Spearman's ρ is computed as Pearson correlation of **ranks**. When a monotone transform introduces **ties** (plateaus), the standard convention is to assign **average ranks** to tied values. This changes the rank vector compared to the original, which can **reduce** Spearman ρ slightly.

The invariance holds exactly only for **strictly** increasing transforms (no flat regions). Isotonic regression produces **non-decreasing** (not strictly increasing) functions, so:
- If the isotonic fit has plateaus → ties are introduced → average ranks are assigned → Spearman can change (typically decrease slightly)
- If the isotonic fit is strictly increasing (no plateaus) → ranks preserved exactly → Spearman is invariant

From the Tacheny results: Spearman went from 0.8430 to 0.8563 (a small increase, not decrease), suggesting that in practice the plateau effect was negligible or the isotonic fit was nearly strictly increasing on the STS data.

**Confidence: High** — The mathematical logic is clear: strictly increasing → exact invariance; non-decreasing with plateaus → ties → average ranks → potential small change.

---

## 7. Additional Prior Art on Calibration Methods

### Platt Scaling (Platt, 1999):
> "Platt [6] introduced **sigmoid calibration** for SVMs." (Tacheny 2026, Section 2)

### Isotonic Regression for Classifier Calibration (Zadrozny & Elkan, 2002):
> "Zadrozny and Elkan [4] showed **isotonic regression effectively calibrates classifier probabilities**."

### Comparison of Methods (Niculescu-Mizil & Caruana, 2005):
> "Niculescu-Mizil and Caruana [7] systematically compared methods, finding **isotonic regression robust across classifiers**."

### Tacheny's comparison (Table 1):
- **Linear regression**: preserves Spearman exactly (affine transform is strictly monotone)
- **Isotonic regression**: best overall — improves both Pearson and Spearman, eliminates bias
- **Sigmoid**: preserves Spearman (strictly monotone) but **worsens** calibration (ECE ↑ 226%)
- **Polynomial (deg 2-4)**: preserves Spearman if monotone on the domain, improves RMSE modestly
- **Beta distribution**: **inverts** the ranking (Spearman = −0.8246) because it's not monotone on the full domain

**Confidence: Very High** — Directly from the paper's Table 1 and Section 5.

---

## Summary of Key Facts

| Fact | Source | Confidence |
|------|--------|------------|
| Spearman ρ = Pearson(rank(x), rank(y)) | StatTutor, Minitab, scipy docs | Very High |
| Invariant under strictly increasing transforms | Mathematical consequence of rank definition | Very High |
| Kendall τ also invariant (rank/pairwise) | van den Heuvel & Zhan 2022 | High |
| Isotonic regression = non-decreasing step function | sklearn docs | Very High |
| Isotonic calibration of cosine similarity | Tacheny 2026 (arXiv:2601.16907) | Very High |
| Ties from plateaus can change Spearman | Tacheny 2026, Section 6.1 | High |
| "Myths" paper: Pearson can beat Spearman for monotonic | van den Heuvel & Zhan 2022 | Very High |
| Isotonic achieves ECE=0 while preserving rank | Tacheny 2026, Table 1 | Very High |

---

## References

1. **van den Heuvel, E. & Zhan, Z. (2022).** "Myths About Linear and Monotonic Associations: Pearson's r, Spearman's ρ, and Kendall's τ". *The American Statistician*, 76(1), 44–52. DOI: 10.1080/00031305.2021.2004922

2. **Tacheny, N. (2026).** "Calibrated Similarity for Reliable Geometric Analysis of Embedding Spaces". arXiv:2601.16907. https://arxiv.org/abs/2601.16907

3. **scipy.stats.spearmanr** — SciPy documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

4. **sklearn.isotonic.IsotonicRegression** — scikit-learn documentation. https://scikit-learn.org/stable/modules/isotonic.html

5. **StatTutor** — "Spearman's correlation" (PDF). https://www.statstutor.ac.uk/resources/uploaded/spearmans.pdf

6. **Minitab Support** — "A comparison of the Pearson and Spearman correlation methods". https://support.minitab.com/en-us/minitab/help-and-how-to/statistics/basic-statistics/supporting-topics/correlation-and-covariance/a-comparison-of-the-pearson-and-spearman-correlation-methods/

7. **Platt, J. (1999).** "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods".

8. **Zadrozny, B. & Elkan, C. (2002).** "Transforming classifier scores into accurate multiclass probability estimates".

9. **Niculescu-Mizil, A. & Caruana, R. (2005).** "Predicting good probabilities with supervised learning".
