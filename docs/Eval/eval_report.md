# IndicRAG — Computed Evaluation Report

_Generated: 2026-02-26T15:18:56 · k=5 · 4 queries_

> Metrics computed automatically from retrieved document lists and cited chunk texts.
> Relevance judgments are manually labeled (see `relevance_judgments.json`).
> Citation grounding uses token Jaccard similarity (threshold ≥ 0.15) as a proxy for semantic overlap.

---

## Aggregate Metrics

| Metric | Score | Visual |
|---|---|---|
| Precision@5 | **1.000** | `████████████████████` |
| Recall@5    | **0.917** | `██████████████████░░` |
| MRR           | **1.000**   | `████████████████████` |
| Citation Grounding Accuracy | **0.938** | `███████████████████░` |
| **Composite** | **0.964** | `███████████████████░` |

---

## Per-Query Results

### Query A

> What is the role of smooth thresholding in tandem neural network antenna design?

| Metric | Score |
|---|---|
| Precision@5 | 1.000 |
| Recall@5    | 1.000 |
| Reciprocal Rank | 1.000 |
| Citation Grounding | 0.750 (3/4 claims) |

**Retrieved (top 5):** tandem_nn, tandem_nn, tandem_nn, tandem_nn, tandem_nn

**Expected relevant:** tandem_nn

<details>
<summary>Per-claim grounding detail</summary>

| Claim (truncated) | Cited Paper | Similarity | Grounded |
|---|---|---|---|
| The smooth thresholding function promotes the discrete nature of design paramete... | tandem_nn | 0.091 | ❌ |
| The ST function works in conjunction with crucial regularization terms in the ne... | tandem_nn | 0.867 | ✅ |
| The resulting antennas can be up to 50% more compact in area and up to 18% thinn... | tandem_nn | 0.389 | ✅ |
| The framework enables the synthesis of custom microstrip antennas in less than o... | tandem_nn | 1.000 | ✅ |

</details>

### Query B

> Compare Bayesian optimization and PPO-based RL methods for antenna optimization.

| Metric | Score |
|---|---|
| Precision@5 | 1.000 |
| Recall@5    | 1.000 |
| Reciprocal Rank | 1.000 |
| Citation Grounding | 1.000 (4/4 claims) |

**Retrieved (top 5):** bayesian_techniques, bayesian_stp, ppo_pixel, bayesian_stp, ppo_wireless

**Expected relevant:** bayesian_techniques, ppo_pixel, bayesian_stp, ppo_wireless

<details>
<summary>Per-claim grounding detail</summary>

| Claim (truncated) | Cited Paper | Similarity | Grounded |
|---|---|---|---|
| BO-STP-EST demonstrated superior performance by finding a minimum value of 2.88e... | bayesian_stp | 1.000 | ✅ |
| BO-GP-EST required 8 iterations to find its minimum. | bayesian_stp | 1.000 | ✅ |
| The PPO algorithm obtains the probability distribution of design parameters base... | ppo_pixel | 0.469 | ✅ |
| RL is specifically noted for its utility in dynamic environments where antennas ... | ppo_wireless | 0.833 | ✅ |

</details>

### Query C

> How do surrogate-model-based methods differ from policy-based RL in sample efficiency and convergence behavior?

| Metric | Score |
|---|---|
| Precision@5 | 1.000 |
| Recall@5    | 0.667 |
| Reciprocal Rank | 1.000 |
| Citation Grounding | 1.000 (2/2 claims) |

_1 claim(s) had no citation — system correctly acknowledged absent context._

**Retrieved (top 5):** surrogate_lowcost, surrogate_framework, surrogate_framework, surrogate_lowcost, surrogate_framework

**Expected relevant:** ppo_pixel, surrogate_framework, surrogate_lowcost

<details>
<summary>Per-claim grounding detail</summary>

| Claim (truncated) | Cited Paper | Similarity | Grounded |
|---|---|---|---|
| GPR has strong data-fitting capabilities with small sample sizes, performing bes... | surrogate_framework | 0.688 | ✅ |
| As sample size increases to 1000 and 1600, Decision Tree Regression slightly out... | surrogate_framework | 0.900 | ✅ |
| The context contains no mention of policy, reinforcement learning, gradients in ... | none | — | ⚠️ N/A |

</details>

### Query D

> What hyperparameters control PPO training stability in the cited antenna design paper?

| Metric | Score |
|---|---|
| Precision@5 | 1.000 |
| Recall@5    | 1.000 |
| Reciprocal Rank | 1.000 |
| Citation Grounding | 1.000 (1/1 claims) |

**Retrieved (top 5):** ppo_pixel, ppo_pixel, ppo_pixel, ppo_pixel, ppo_pixel

**Expected relevant:** ppo_pixel

<details>
<summary>Per-claim grounding detail</summary>

| Claim (truncated) | Cited Paper | Similarity | Grounded |
|---|---|---|---|
| REPLACE THIS: paste a claim from your actual Test D answer here. | ppo_pixel | 0.182 | ✅ |

</details>

---

## Methodology Notes

**Precision@5** — fraction of top-5 retrieved documents that appear in the manually labeled relevant set.

**Recall@5** — fraction of relevant documents that were retrieved in the top 5.

**MRR** — reciprocal rank of the first relevant document in the retrieved list. Score of 1.0 means the most relevant paper ranked first.

**Citation Grounding Accuracy** — for each factual claim in the answer, token Jaccard similarity is computed between the claim and its cited chunk. Claims above the threshold (0.15) are marked grounded. Claims where the system explicitly stated no source exists are excluded from the denominator (correct behavior).

**Limitations** — Jaccard similarity is a weak proxy for semantic overlap; it will undercount grounding for paraphrased claims. A stronger evaluation would use a cross-encoder or an LLM judge. Relevance judgments are from a single annotator (the developer).
