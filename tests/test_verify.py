"""Regression tests for verify.check_claims (faithfulness scoring).

Covers a real bug found in production: a [Cite:N] marker left in the NLI
hypothesis text collapses entailment probability toward 0 regardless of how
well the source actually supports the claim, and a marker placed right after
a sentence-ending period gets split into its own citation-only "sentence"
with no claim text at all. Both silently forced faithfulness scores toward
0 on nearly every real answer.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import verify


def _fake_model(entailment_logit=5.0, contradiction_logit=-3.0, neutral_logit=-2.0):
    """CrossEncoder stand-in returning fixed (contradiction, entailment, neutral) logits."""
    m = MagicMock()
    m.predict = MagicMock(
        side_effect=lambda pairs: np.array([[contradiction_logit, entailment_logit, neutral_logit]] * len(pairs))
    )
    return m


def test_citation_marker_after_period_is_merged_not_orphaned():
    """[Cite:N] right after the period must not become its own claimless fragment."""
    answer = "The framework uses deep Q-networks for optimization. [Cite:1]"
    chunks = ["irrelevant chunk text"]

    with patch("verify._load", return_value=_fake_model()):
        results = verify.check_claims(answer, chunks)

    assert len(results) == 1
    assert results[0]["claim"].strip().endswith("[Cite:1]")


def test_citation_marker_stripped_from_nli_hypothesis():
    """The literal '[Cite:N]' text must not reach the NLI model as part of the hypothesis."""
    answer = "The framework uses deep Q-networks for optimization. [Cite:1]"
    chunks = ["The framework uses deep Q-networks for optimization."]

    fake_model = _fake_model()
    with patch("verify._load", return_value=fake_model):
        verify.check_claims(answer, chunks)

    called_pairs = fake_model.predict.call_args[0][0]
    assert len(called_pairs) == 1
    premise, hypothesis = called_pairs[0]
    assert "[Cite:" not in hypothesis
    assert "[Cite:" not in premise


def test_high_entailment_logit_yields_high_grounded_score():
    answer = "The framework uses deep Q-networks for antenna optimization. [Cite:1]"
    chunks = ["The proposed framework uses deep Q-networks to optimize antenna parameters."]

    with patch("verify._load", return_value=_fake_model(entailment_logit=5.0)):
        results = verify.check_claims(answer, chunks)

    assert results[0]["support"] > 0.9
    assert results[0]["grounded"] is True


def test_low_entailment_logit_yields_ungrounded():
    answer = "The framework achieves 99% accuracy on unrelated benchmark X. [Cite:1]"
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


def test_not_found_marker_also_merges_as_citation_only_fragment():
    answer = "No information was available on this topic. [NOT FOUND: topic] Second sentence. [Cite:1]"
    chunks = ["relevant chunk"]

    with patch("verify._load", return_value=_fake_model()):
        results = verify.check_claims(answer, chunks)

    # Only the [Cite:1] sentence is scorable; [NOT FOUND] has no chunk index to check.
    assert len(results) == 1
    assert "Second sentence" in results[0]["claim"]
