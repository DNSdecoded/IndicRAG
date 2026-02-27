# Evaluation Report — IndicRAG

This document has two parts. The first section contains **automatically computed retrieval and grounding metrics** produced by `evaluate.py`. The second section contains **manual qualitative assessments** of answer quality for each query.

To reproduce or update the computed metrics:

```bash
python evaluate.py --k 4 --output eval_report.json
```

Edit `relevance_judgments.json` to update ground-truth relevance labels. Edit `answers_and_citations.json` to add new query results.

---

## Computed Metrics (automated)

_Full per-query breakdown: [eval_report.json](eval_report.json)_

| Metric | Score |
|---|---|
| Precision@4 | 1.000 |
| Recall@4 | 0.854 |
| MRR | 1.000 |
| Citation Grounding Accuracy | 0.938 |
| **Retrieval Score** | **0.951** |
| **Generation Score** | **0.938** |
| **Overall** | **0.944** |

**Methodology:** Precision, Recall, and MRR are computed from ranked retrieval lists against manually labeled relevance judgments. Citation Grounding Accuracy uses token Jaccard similarity (threshold ≥ 0.15) between each answer claim and its cited chunk — claims where the system explicitly stated no source exists are excluded from the denominator (correct epistemic behavior). See `evaluate.py` for full details and limitations.

---

## Manual Qualitative Assessment

This document reports the results of a manual evaluation of the IndicRAG system across four benchmark queries. Each query was designed to stress a different system capability. Scores were assigned by the developer based on inspection of retrieved chunks, generated answers, and citation accuracy.

> **Caveat:** These scores reflect human assessment on a small, controlled query set — not a statistically sampled benchmark. They should be interpreted as a qualitative signal of system behavior, not a formal research evaluation.

---

## Evaluation Methodology

### Scoring Dimensions

Each query was scored on nine dimensions (0–1 scale):

| Metric | What It Measures |
|---|---|
| Retrieval Precision | Fraction of retrieved chunks that were materially relevant |
| Retrieval Recall | Fraction of necessary sources that were actually retrieved |
| Faithfulness | Fraction of claims in the answer that were grounded in citations |
| Attribution Accuracy | Correct mapping of methods/results to their source papers |
| Technical Depth | Quality of mechanism-level explanation |
| Convergence Reasoning | Accuracy of optimization/update-rule analysis |
| Cross-Document Discipline | Absence of forced or unsupported cross-paper synthesis |
| Hallucination Rate | Rate of unsupported or fabricated claims (lower is better) |
| Formatting Compliance | Structural adherence to output template |

### Query Categories

| Test | Category | Purpose |
|---|---|---|
| A | Single-source mechanistic | Focused retrieval + gradient reasoning discipline |
| B | Cross-method comparison | Structured synthesis + convergence contrast |
| C | Convergence / sample efficiency | Mechanistic rigor under theory-heavy query |
| D | Insufficient-context constraint | Epistemic honesty + non-extrapolation |

---

## Test A — Single-Source Mechanistic

**Query:** *What is the role of smooth thresholding in tandem neural network antenna design?*

**Summary:** The system correctly identified this as a single-source question and retrieved exclusively from the tandem neural network paper. It explained the ST function's role in enabling discrete-variable handling within a pixel-based geometry, noted its integration with regularization terms in the loss function, and cited specific performance claims (50% area reduction, 18% substrate height reduction). It correctly acknowledged that the gradient-flow behavior of the ST function was not specified in the retrieved context rather than speculating.

**Where it fell short:** The answer acknowledged the absence of gradient-propagation details but did not explain *why* smoothness is necessary for backpropagation — a mechanistic gap that a stronger answer would address explicitly.

| Metric | Score |
|---|---|
| Retrieval Precision | 0.95 |
| Retrieval Recall | 0.90 |
| Faithfulness | 0.97 |
| Attribution Accuracy | 0.97 |
| Technical Depth | 0.85 |
| Convergence Reasoning | 0.82 |
| Cross-Document Discipline | 0.98 |
| Hallucination Rate | < 2% |
| Formatting Compliance | 0.98 |
| **Composite** | **0.93** |

---

## Test B — Cross-Method Comparison

**Query:** *Compare Bayesian optimization and PPO-based RL methods for antenna optimization.*

**Summary:** The system retrieved from four relevant sources covering BO-GP, BO-STP, PPO for pixel antennas, and PPO for wireless coverage. It correctly contrasted the two paradigms: BO using surrogate models (Gaussian and Student's T processes) for efficient global search in low-iteration regimes, and PPO using actor-critic networks for high-dimensional discrete topology problems. Specific iteration counts (3, 8, 41) and objective values were cited correctly. The comparison table was appropriate given the comparative nature of the query.

**Where it fell short:** The answer did not explicitly contrast the update mechanisms — posterior Bayesian update vs. clipped policy gradient step — which would have elevated the mechanistic depth. Exploration-exploitation differences were mentioned but not analyzed.

| Metric | Score |
|---|---|
| Retrieval Precision | 0.90 |
| Retrieval Recall | 0.92 |
| Faithfulness | 0.97 |
| Attribution Accuracy | 0.97 |
| Technical Depth | 0.88 |
| Convergence Reasoning | 0.85 |
| Cross-Document Discipline | 0.95 |
| Hallucination Rate | < 2% |
| Formatting Compliance | 0.98 |
| **Composite** | **0.94** |

---

## Test C — Convergence and Sample Efficiency

**Query:** *How do surrogate-model-based methods differ from policy-based RL in sample efficiency and convergence behavior?*

**Summary:** The retrieved context contained surrogate-model papers (GPR, DTR, RBFN with evolutionary algorithms) but no policy-based RL material. The system explicitly stated this absence rather than fabricating a comparison. It then provided a detailed analysis of sample efficiency *within* the surrogate domain — showing how GPR outperforms at small sample sizes (S=600) while DTR takes over at larger sizes (S≥1000). The answer correctly labeled this as a surrogate-only analysis and declined to infer RL behavior.

This is the strongest result in the evaluation set. Refusing to hallucinate a comparison when context is absent is the correct behavior for a grounded scientific assistant.

| Metric | Score |
|---|---|
| Retrieval Precision | 0.96 |
| Retrieval Recall | 0.93 |
| Faithfulness | 0.99 |
| Attribution Accuracy | 0.98 |
| Technical Depth | 0.90 |
| Convergence Reasoning | 0.88 |
| Cross-Document Discipline | 0.98 |
| Hallucination Rate | ~0% |
| Formatting Compliance | 0.99 |
| **Composite** | **0.96** |

---

## Test D — Strict Grounding (Hyperparameter Precision)

**Query:** *What hyperparameters control PPO training stability in the cited antenna design paper?*

This test was designed to probe whether the system would extrapolate PPO hyperparameter knowledge from general ML literature when the specific paper did not provide it. Results from this test are incorporated into the aggregate scores below.

---

## Aggregate Results

| Metric | Score (avg across tests) |
|---|---|
| Retrieval Precision | 0.93 |
| Retrieval Recall | 0.91 |
| Faithfulness | 0.98 |
| Attribution Accuracy | 0.97 |
| Technical Depth | 0.88 |
| Convergence / Mechanistic Reasoning | 0.86 |
| Cross-Document Discipline | 0.95 |
| Hallucination Rate | < 2% |
| Formatting Compliance | 0.98 |
| **Final Composite** | **0.95** |

---

## Key Observations

**Strengths**

The system maintains strict citation discipline across all test types. No fabricated equations, invented hyperparameters, or unsupported performance claims were observed. The insufficient-context case (Test C) demonstrated that the system correctly refuses to synthesize comparisons when relevant material is absent — a critical property for scientific use.

**Remaining Gaps**

Mechanistic depth is the lowest-scoring dimension (0.86–0.88). When retrieved context is thin on gradient-flow or update-rule specifics, the system acknowledges the gap but does not always explain *why* the gap limits the analysis. This is a prompt-engineering target for future iterations, not a retrieval failure.

**What changed between early and current versions**

An earlier version of the system produced unsolicited cross-paper comparison tables, included irrelevant medical disclaimers in engineering queries, and occasionally synthesized across papers without sufficient grounding. The current prompt revision eliminated these behaviors across all four tests.

---

## Limitations of This Evaluation

- Four queries is a small sample; results may not generalize across all domain areas or query types.
- Scores were assigned by the developer, not an independent evaluator.
- There is no automated metric (e.g., RAGAS, TruLens) comparison; this is a purely qualitative assessment.
- The query set was designed to test known system behaviors rather than to discover unknown failure modes.

A more rigorous evaluation would involve an independent annotator, a larger query set drawn from held-out documents, and automated faithfulness scoring against a ground-truth answer set.
