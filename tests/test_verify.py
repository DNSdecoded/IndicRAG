"""Regression tests for verify.check_claims (faithfulness scoring).

Covers a real bug found in production: a [N] marker left in the NLI
hypothesis text collapses entailment probability toward 0 regardless of how
well the source actually supports the claim, and a marker placed right after
a sentence-ending period gets split into its own citation-only "sentence"
with no claim text at all. Both silently forced faithfulness scores toward
0 on nearly every real answer.
"""

from unittest.mock import MagicMock, patch

import numpy as np

import verify


def _fake_model(entailment_logit=5.0, contradiction_logit=-3.0, neutral_logit=-2.0):
    """CrossEncoder stand-in returning fixed NLI logits.

    Label order matches the default multilingual model (mDeBERTa-xnli):
    index 0 = entailment, 1 = neutral, 2 = contradiction — i.e.
    config.NLI_ENTAILMENT_INDEX == 0.
    """
    m = MagicMock()
    m.predict = MagicMock(
        side_effect=lambda pairs: np.array([[entailment_logit, neutral_logit, contradiction_logit]] * len(pairs))
    )
    return m


def test_citation_marker_after_period_is_merged_not_orphaned():
    """[N] right after the period must not become its own claimless fragment."""
    answer = "The framework uses deep Q-networks for optimization. [1]"
    chunks = ["irrelevant chunk text"]

    with patch("verify._load", return_value=_fake_model()):
        results = verify.check_claims(answer, chunks)

    assert len(results) == 1
    assert results[0]["claim"].strip().endswith("[1]")


def test_citation_marker_stripped_from_nli_hypothesis():
    """The literal '[N]' text must not reach the NLI model as part of the hypothesis."""
    answer = "The framework uses deep Q-networks for optimization. [1]"
    chunks = ["The framework uses deep Q-networks for optimization."]

    fake_model = _fake_model()
    with patch("verify._load", return_value=fake_model):
        verify.check_claims(answer, chunks)

    called_pairs = fake_model.predict.call_args[0][0]
    assert len(called_pairs) == 1
    premise, hypothesis = called_pairs[0]
    assert "[" not in hypothesis
    assert "[" not in premise


def test_faithfulness_threshold_is_reachable():
    """The grounded bar must sit inside the range the NLI model actually produces.

    Measured on this corpus with the shipped int8 mDeBERTa-xnli: a sentence copied
    verbatim out of its own chunk scores median 0.226 / max 0.428; an unrelated
    paper's sentence scores median 0.099, p90 0.158 — the two distributions overlap,
    so 0.15 trades a ~0.10-0.15 false-positive rate for 0.70 recall. The old 0.5 was
    above every positive, so `grounded` was always False and faithfulness read ~0 for
    every answer. Guards against a future model/threshold change reintroducing that.
    """
    import config

    assert 0.099 < config.FAITHFULNESS_THRESHOLD < 0.428, (
        "threshold outside the measured positive/negative separation — recalibrate "
        "before changing NLI_MODEL_NAME or FAITHFULNESS_THRESHOLD"
    )


def test_chunks_per_citation_are_capped():
    """One paper contributing many chunks must not cost one NLI pair each — the
    per-citation cap is what keeps the faithfulness pass inside the agent budget."""
    import config

    answer = "The framework uses deep Q-networks for optimization. [1]"
    chunks = [f"chunk {i} about deep Q-networks" for i in range(8)]
    metas = [{"title": "One Paper"} for _ in chunks]  # all 8 chunks = citation [1]

    fake_model = _fake_model()
    with patch("verify._load", return_value=fake_model):
        verify.check_claims(answer, chunks, metas)

    called_pairs = fake_model.predict.call_args[0][0]
    assert len(called_pairs) == config.NLI_MAX_CHUNKS_PER_CITATION


def test_high_entailment_logit_yields_high_grounded_score():
    answer = "The framework uses deep Q-networks for antenna optimization. [1]"
    chunks = ["The proposed framework uses deep Q-networks to optimize antenna parameters."]

    with patch("verify._load", return_value=_fake_model(entailment_logit=5.0)):
        results = verify.check_claims(answer, chunks)

    assert results[0]["support"] > 0.9
    assert results[0]["grounded"] is True


def test_low_entailment_logit_yields_ungrounded():
    answer = "The framework achieves 99% accuracy on unrelated benchmark X. [1]"
    chunks = ["The proposed framework uses deep Q-networks to optimize antenna parameters."]

    with patch("verify._load", return_value=_fake_model(entailment_logit=-5.0, neutral_logit=5.0)):
        results = verify.check_claims(answer, chunks)

    assert results[0]["support"] < 0.1
    assert results[0]["grounded"] is False


def test_sentence_with_no_citation_is_skipped():
    answer = "This sentence has no citation marker at all."
    chunks = ["some chunk"]

    with patch("verify._load", return_value=_fake_model()):
        results = verify.check_claims(answer, chunks)

    assert results == []


def test_citation_number_maps_to_correct_paper_not_chunk_index():
    """[N] is a per-paper number: it must score against that paper's chunk(s),
    not chunks[N-1]. Regression for the faithfulness citation/chunk-index bug."""
    # PaperA contributes 2 chunks, PaperB 1 chunk. [2] == PaperB.
    chunks = ["A intro text", "A methods text", "B results text"]
    metadatas = [{"title": "PaperA"}, {"title": "PaperA"}, {"title": "PaperB"}]
    answer = "The result was significant. [2]"

    fake_model = _fake_model()
    with patch("verify._load", return_value=fake_model):
        verify.check_claims(answer, chunks, metadatas)

    premises = [p[0] for p in fake_model.predict.call_args[0][0]]
    assert premises == ["B results text"]          # scored PaperB
    assert "A methods text" not in premises         # not chunks[1]


def test_metadatas_none_falls_back_to_chunk_index():
    """Without metadatas, [N] -> chunks[N-1] (historical behaviour, still valid
    when every chunk is a distinct document)."""
    chunks = ["first", "second"]
    answer = "Claim. [2]"

    fake_model = _fake_model()
    with patch("verify._load", return_value=fake_model):
        verify.check_claims(answer, chunks)

    premises = [p[0] for p in fake_model.predict.call_args[0][0]]
    assert premises == ["second"]


def _fake_model_per_pair(rows):
    """CrossEncoder stand-in returning a distinct logit row per input pair, in order."""
    m = MagicMock()
    m.predict = MagicMock(side_effect=lambda pairs: np.array(rows[:len(pairs)]))
    return m


def test_check_claims_returns_argmax_supporting_chunk():
    """The winning (highest-entailment) chunk must be surfaced, not just its score."""
    chunks = ["chunk with weak overlap", "chunk that strongly supports the claim"]
    metadatas = [{"title": "PaperA"}, {"title": "PaperA"}]
    answer = "The claim is well supported. [1]"

    fake_model = _fake_model_per_pair([
        [-2.0, 0.0, 1.0],   # low entailment for chunk 0
        [5.0, -2.0, -3.0],  # high entailment for chunk 1
    ])
    with patch("verify._load", return_value=fake_model):
        results = verify.check_claims(answer, chunks, metadatas)

    assert results[0]["supporting_chunk_index"] == 1
    assert results[0]["supporting_chunk"] == chunks[1]


def test_supporting_chunk_truncated_for_payload_size():
    chunks = ["x" * 1000]
    answer = "Claim. [1]"

    with patch("verify._load", return_value=_fake_model()):
        results = verify.check_claims(answer, chunks)

    assert len(results[0]["supporting_chunk"]) == 500


def test_not_found_marker_also_merges_as_citation_only_fragment():
    answer = "No information was available on this topic. [NOT FOUND: topic] Second sentence. [1]"
    chunks = ["relevant chunk"]

    with patch("verify._load", return_value=_fake_model()):
        results = verify.check_claims(answer, chunks)

    # Only the [1] sentence is scorable; [NOT FOUND] has no chunk index to check.
    assert len(results) == 1
    assert "Second sentence" in results[0]["claim"]
