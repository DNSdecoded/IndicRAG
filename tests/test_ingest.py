"""Unit tests for ingest.py — per-section chunk sizing, content-hash dedup,
batch chunk building, and Indic section detection. No models/network required."""


class FakeCollection:
    """Minimal ChromaDB-collection stand-in supporting .get()/.count()."""
    name = "test"

    def __init__(self, docs=None):
        # docs: list of (id, metadata) tuples
        self._docs = list(docs or [])

    def get(self, where=None, limit=None, include=None):
        if where:
            matched = [(i, m) for i, m in self._docs
                       if all(m.get(k) == v for k, v in where.items())]
        else:
            matched = list(self._docs)
        if limit is not None:
            matched = matched[:limit]
        return {"ids": [i for i, _ in matched],
                "metadatas": [m for _, m in matched]}

    def count(self):
        return len(self._docs)


def test_per_section_chunk_size():
    """Dense sections (abstract) chunk smaller than narrative sections (results)."""
    import ingest, config

    abstract = "This is a short dense sentence. " * 200
    results = "This is a narrative result sentence. " * 200
    prepared = ingest._build_paper_chunks(
        "p1", "Title", [("abstract", abstract), ("results", results)], {}, FakeCollection()
    )
    assert prepared is not None

    ab = [c for c, m in zip(prepared["chunks"], prepared["metadatas"]) if m["section"] == "abstract"]
    res = [c for c, m in zip(prepared["chunks"], prepared["metadatas"]) if m["section"] == "results"]

    # Each section respects its configured ceiling (small tolerance for word boundaries).
    assert max(len(c) for c in ab) <= config.SECTION_CHUNK_SIZES["abstract"] + 10
    assert max(len(c) for c in res) <= config.SECTION_CHUNK_SIZES["results"] + 10
    # Narrative section produces larger chunks than the dense one.
    assert max(len(c) for c in res) > max(len(c) for c in ab)


def test_content_hash_dedup_skips_existing():
    """A paper whose content hash already exists under another paper_id is skipped."""
    import ingest

    coll = FakeCollection(docs=[("c1", {"paper_id": "orig", "content_hash": "abc123", "title": "X"})])
    out = ingest._build_paper_chunks(
        "newid", "Different Title", [("introduction", "word " * 100)],
        {"content_hash": "abc123"}, coll
    )
    assert out is None


def test_in_batch_content_dedup():
    """Two identical papers in one batch: only the first is prepared."""
    import ingest

    coll = FakeCollection()
    seen = set()
    first = ingest._build_paper_chunks(
        "p1", "T1", [("introduction", "alpha " * 100)], {"content_hash": "dup"}, coll, seen
    )
    second = ingest._build_paper_chunks(
        "p2", "T2", [("introduction", "beta " * 100)], {"content_hash": "dup"}, coll, seen
    )
    assert first is not None
    assert second is None


def test_indic_section_detection():
    """extract_sections detects Devanagari section headers, not just Latin."""
    import pdf_utils

    text = (
        "कुछ शीर्षक\n"
        "प्रस्तावना\n"
        "यह एक परिचय है जो पर्याप्त रूप से लंबा वाक्य है।\n"
        "निष्कर्ष\n"
        "यह अंतिम निष्कर्ष है।\n"
    )
    names = [n for n, _ in pdf_utils.extract_sections(text)]
    assert "प्रस्तावना" in names
    assert "निष्कर्ष" in names


if __name__ == "__main__":
    test_per_section_chunk_size()
    test_content_hash_dedup_skips_existing()
    test_in_batch_content_dedup()
    test_indic_section_detection()
    print("all ingest tests passed")
