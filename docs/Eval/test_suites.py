"""Schema validation for eval suites (P0-EVL-2 "schema validation of suites")."""

import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SUITES = sorted((HERE / "suites").glob("*.json"))
MANIFEST_IDS = {p["id"] for p in json.loads((HERE / "fixtures" / "manifest.json").read_text("utf-8"))["papers"]}
LANGS = {"en", "hi", "te", "ta", "bn", "mr", "gu", "kn", "ml", "pa", "or"}
TYPES = {"native", "terminology", "broad", "cross_lingual", "code_mixed", "transliterated"}


def test_there_is_a_suite():
    assert SUITES


@pytest.mark.parametrize("path", SUITES, ids=[p.name for p in SUITES])
def test_suite_schema(path):
    suite = json.loads(path.read_text("utf-8"))
    corpus = set(suite["corpus"])
    assert corpus <= MANIFEST_IDS, f"corpus ids not in the fixture manifest: {corpus - MANIFEST_IDS}"
    ids = [q["id"] for q in suite["queries"]]
    assert len(ids) == len(set(ids)), "duplicate query ids"
    for q in suite["queries"]:
        assert q["text"].strip(), q["id"]
        assert q["lang"] in LANGS and q["type"] in TYPES, q["id"]
        assert q["relevant_papers"] and set(q["relevant_papers"]) <= corpus, q["id"]
        grades = q["relevance_grades"]
        assert set(grades) == set(q["relevant_papers"]), q["id"]
        assert all(g in (1, 2, 3) for g in grades.values()), q["id"]
        assert 3 in grades.values(), f"{q['id']}: no paper answers it (grade 3)"
        assert isinstance(q.get("reviewed"), bool), q["id"]
        if "parallel_of" in q:
            assert q["parallel_of"] in ids, q["id"]


def test_every_fixture_paper_is_judged_somewhere():
    judged = set()
    for path in SUITES:
        for q in json.loads(path.read_text("utf-8"))["queries"]:
            judged.update(q["relevant_papers"])
    assert MANIFEST_IDS <= judged, f"papers no query targets: {MANIFEST_IDS - judged}"
