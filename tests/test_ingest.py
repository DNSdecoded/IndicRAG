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


def test_resolve_urls_direct_url():
    from routes.ingest import _resolve_urls_to_ingest

    result = _resolve_urls_to_ingest(url="http://example.com/paper.pdf")
    assert result == [{"url": "http://example.com/paper.pdf", "id": "http://example.com/paper.pdf", "title": ""}]


def test_resolve_urls_arxiv_id(monkeypatch):
    import routes.ingest as ingest_routes

    monkeypatch.setattr(ingest_routes, "execute_arxiv_search", lambda q, max_results, **kw: {
        "passages": [{"pdf_url": "http://arxiv.org/pdf/2301.07041", "title": "A Paper"}]
    })

    result = ingest_routes._resolve_urls_to_ingest(arxiv_id="2301.07041")
    assert result == [{"url": "http://arxiv.org/pdf/2301.07041", "id": "2301.07041", "title": "A Paper"}]


def test_resolve_urls_doi(monkeypatch):
    import routes.ingest as ingest_routes

    monkeypatch.setattr(ingest_routes, "execute_open_access_search", lambda q, max_results: {
        "passages": [{"pdf_url": "http://oa.example.com/x.pdf", "title": "OA Paper"}]
    })

    result = ingest_routes._resolve_urls_to_ingest(doi="10.1234/abcd")
    assert result == [{"url": "http://oa.example.com/x.pdf", "id": "10.1234/abcd", "title": "OA Paper"}]


def test_resolve_urls_arxiv_no_pdf_found_skips(monkeypatch):
    import routes.ingest as ingest_routes

    monkeypatch.setattr(ingest_routes, "execute_arxiv_search", lambda q, max_results, **kw: {"passages": []})

    assert ingest_routes._resolve_urls_to_ingest(arxiv_id="nope") == []


def test_resolve_urls_reading_list_mixed(monkeypatch):
    import routes.ingest as ingest_routes

    def fake_arxiv(q, max_results, **kw):
        return {"passages": [{"pdf_url": f"http://arxiv.org/pdf/{q}", "title": f"Paper {q}"}]}

    def fake_oa(q, max_results):
        return {"passages": [{"pdf_url": f"http://oa.example.com/{q}.pdf", "title": f"DOI {q}"}]}

    monkeypatch.setattr(ingest_routes, "execute_arxiv_search", fake_arxiv)
    monkeypatch.setattr(ingest_routes, "execute_open_access_search", fake_oa)

    reading_list = "2301.07041\nhttps://direct.example.com/y.pdf\n10.1234/abcd\n\n  \n"
    result = ingest_routes._resolve_urls_to_ingest(reading_list=reading_list)

    assert result == [
        {"url": "http://arxiv.org/pdf/2301.07041", "id": "2301.07041", "title": "Paper 2301.07041"},
        {"url": "https://direct.example.com/y.pdf", "id": "https://direct.example.com/y.pdf", "title": ""},
        {"url": "http://oa.example.com/10.1234/abcd.pdf", "id": "10.1234/abcd", "title": "DOI 10.1234/abcd"},
    ]


def test_resolve_urls_nothing_provided_returns_empty():
    from routes.ingest import _resolve_urls_to_ingest

    assert _resolve_urls_to_ingest() == []


def test_batch_url_ingest_skips_existing_paper_id(monkeypatch):
    """Regression: an attacker (or an honest duplicate) whose sanitized id
    collides with an already-ingested paper_id must not silently overwrite
    that paper's chunks — /ingest/from-url has no confirmation step, unlike
    /upload (409s on collision) or /reindex (explicit, intentional overwrite)."""
    import routes.ingest as ingest_routes
    import vector_store

    calls = []

    class _FakeCollection:
        def get(self, include=None):
            return {"metadatas": [{"paper_id": "existing_paper"}]}

    monkeypatch.setattr(vector_store, "get_or_create_collection", lambda: _FakeCollection())
    monkeypatch.setattr(vector_store, "_chroma_call", lambda fn, **kw: fn(**kw))
    monkeypatch.setattr(ingest_routes, "_update_job", lambda *a, **k: None)
    monkeypatch.setattr("download_utils.download_pdf", lambda url: "/tmp/fake.pdf")

    class _FakeIngestModule:
        @staticmethod
        def ingest_pdf(path, paper_id, metadata):
            calls.append(paper_id)
            return (3, "Title")

    import sys
    monkeypatch.setitem(sys.modules, "ingest", _FakeIngestModule())
    monkeypatch.setattr("pathlib.Path.unlink", lambda self: None)
    monkeypatch.setattr(ingest_routes, "_post_ingest_refresh", lambda: None)

    urls = [
        {"url": "http://example.com/a.pdf", "id": "existing_paper", "title": "Colliding"},
        {"url": "http://example.com/b.pdf", "id": "new_paper", "title": "New"},
    ]
    ingest_routes._run_batch_url_ingest("job-1", urls)

    assert calls == ["new_paper"]  # existing_paper was skipped, never ingested


def test_per_section_chunk_size():
    """Dense sections (abstract) chunk smaller than narrative sections (results)."""
    import ingest
    import config

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
