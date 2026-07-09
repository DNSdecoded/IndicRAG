# IndicRAG — Evaluation Report

_Generated: 2026-07-09T22:03:09 · k=5 · 3 queries_

> Retrieval metrics computed from ranked document lists against manually labeled relevance judgments.
> Citation grounding uses token Jaccard similarity (threshold 0.15). Claims with no citation
> (system correctly acknowledged absent context) are excluded from the grounding denominator.

---

## Aggregate Metrics

| Metric | Score | Bar |
|---|---|---|
| Precision@5         | **1.000** | `████████████████████` |
| Recall@5            | **0.889** | `██████████████████░░` |
| MRR                   | **1.000** | `████████████████████` |
| nDCG@10               | **0.908** | `██████████████████░░` |
| Recall@20             | **0.889** | `██████████████████░░` |
| Citation Grounding    | **0.917** | `██████████████████░░` |
| **Retrieval Score**   | **0.963** | `███████████████████░` |
| **Generation Score**  | **0.917** | `██████████████████░░` |
| **Overall**           | **0.940** | `███████████████████░` |

---

## Per-Language Breakdown

> Mean per-query overall score, grouped by query language. Indic vs English quality is shown, not averaged away.

| Language | Queries | Mean Overall | Bar |
|---|---|---|---|
| `en` | 3 | **0.943** | `███████████████████░` |

---

## Per-Query Results

### Query A

> What is the role of smooth thresholding in tandem neural network antenna design?

| Metric | Score |
|---|---|
| Precision@5      | 1.000 |
| Recall@5         | 1.000 |
| Reciprocal Rank    | 1.000 |
| nDCG@10            | 1.000 |
| Recall@20          | 1.000 |
| Citation Grounding | 0.750 (3/4 claims) |

<details>
<summary>Per-claim grounding detail</summary>

| Claim | Similarity | Status |
|---|---|---|
| The smooth thresholding function promotes the discrete nature of design paramete | 0.091 | ❌ |
| The ST function works in conjunction with crucial regularization terms in the ne | 0.867 | ✅ |
| The resulting antennas can be up to 50% more compact in area and up to 18% thinn | 0.389 | ✅ |
| The framework enables the synthesis of custom microstrip antennas in less than o | 1.000 | ✅ |

</details>

### Query B

> Compare Bayesian optimization and PPO-based RL methods for antenna optimization.

| Metric | Score |
|---|---|
| Precision@5      | 1.000 |
| Recall@5         | 1.000 |
| Reciprocal Rank    | 1.000 |
| nDCG@10            | 0.949 |
| Recall@20          | 1.000 |
| Citation Grounding | 1.000 (4/4 claims) |

<details>
<summary>Per-claim grounding detail</summary>

| Claim | Similarity | Status |
|---|---|---|
| BO-STP-EST demonstrated superior performance by finding a minimum value of 2.88e | 1.000 | ✅ |
| BO-GP-EST required 8 iterations to find its minimum. | 1.000 | ✅ |
| The PPO algorithm obtains the probability distribution of design parameters base | 0.469 | ✅ |
| RL is specifically noted for its utility in dynamic environments where antennas  | 0.833 | ✅ |

</details>

### Query C

> How do surrogate-model-based methods differ from policy-based RL in sample efficiency and convergence behavior?

| Metric | Score |
|---|---|
| Precision@5      | 1.000 |
| Recall@5         | 0.667 |
| Reciprocal Rank    | 1.000 |
| nDCG@10            | 0.774 |
| Recall@20          | 0.667 |
| Citation Grounding | 1.000 (2/2 claims) |

_1 claim(s) had no citation — system correctly acknowledged absent context._

<details>
<summary>Per-claim grounding detail</summary>

| Claim | Similarity | Status |
|---|---|---|
| GPR has strong data-fitting capabilities with small sample sizes, performing bes | 0.688 | ✅ |
| As sample size increases to 1000 and 1600, Decision Tree Regression slightly out | 0.900 | ✅ |
| The context contains no mention of policy, reinforcement learning, gradients in  | — | ⚠️ absent |

</details>

---

## Methodology

**Precision@5** — fraction of top-5 retrieved documents in the relevant set.

**Recall@5** — fraction of relevant documents retrieved in top 5.

**MRR** — reciprocal rank of first relevant document. 1.0 = top result was relevant.

**Citation Grounding** — Jaccard similarity between each answer claim and its cited chunk. Threshold 0.15. Claims where the system stated no source exists are excluded from the denominator (correct epistemic behavior, labeled 'absent').

**Retrieval Score** — mean of Precision, Recall, MRR.

**Generation Score** — mean citation grounding across queries.

**nDCG@10** — Normalized Discounted Cumulative Gain at 10. Uses graded relevance (0-3) when `relevance_grades` is present in judgments; defaults to binary (3 for relevant).

**Recall@20** — fraction of relevant documents retrieved in top 20.

**Limitations** — Jaccard similarity undercounts grounding for paraphrased claims. Single annotator for relevance judgments.
