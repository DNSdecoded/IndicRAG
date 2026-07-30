"""Phase 7 — literature-review report: plan sections → synthesize each → Markdown.

`run_report(topic)` is a plain blocking function (retrieval + several LLM calls)
so the route stays a thin background task and tests mock two seams:
`rag.llm_generate` (section planning) and `rag.answer_question` (per-section
synthesis, which already retrieves + cites + scores faithfulness).

v1 is corpus-only. External per-section search is deferred: keeping it corpus-only
sidesteps the v2.1 provenance rule (never silently merge corpus + external).
"""

import json
import logging
import re

import config
import lang_utils
import rag

logger = logging.getLogger(__name__)

# Fallback outline when the planner LLM returns nothing parseable — a generic but
# serviceable review skeleton.
_DEFAULT_SECTIONS = ["Background", "Methods", "Key Findings", "Gaps and Open Questions"]
_PLAN_MAX_TOKENS = 400


def plan_sections(topic: str, language: str = "en", max_sections: int = None) -> list[str]:
    """Decompose a topic into review section titles via one LLM call.

    Returns a de-duplicated, capped list; falls back to a default outline if the
    model returns nothing usable.
    """
    if max_sections is None:
        max_sections = config.REPORT_MAX_SECTIONS
    lang_name = lang_utils.get_language_name(language)
    lang_rule = f"The titles themselves MUST be written in {lang_name}."
    if language != "en":
        lang_rule += " Do not write them in English."
    prompt = (
        f"You are planning a literature-review report on: {topic!r}.\n"
        f"Propose at most {max_sections} section titles (e.g. background, methods "
        f"comparison, key findings, open gaps). Reply with ONLY a JSON array of "
        f"short title strings. This example shows the required FORMAT ONLY — its "
        f'shape, not its language: ["Background", "Methods", "Findings"].\n'
        + lang_rule
    )
    try:
        raw = rag.llm_generate(prompt, max_tokens=_PLAN_MAX_TOKENS)
        sections = _parse_sections(raw)
    except Exception as e:
        logger.warning(f"[Report] section planning failed, using default outline: {e}")
        sections = []
    if not sections:
        sections = list(_DEFAULT_SECTIONS)
    # de-dupe (case-insensitive) preserving order, then cap
    seen, out = set(), []
    for s in sections:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:max_sections]


def _parse_sections(raw: str) -> list[str]:
    """Pull a JSON array of strings out of an LLM reply, tolerating code fences/prose."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in arr if isinstance(x, (str, int, float)) and str(x).strip()]


# Inline citation markers: [3], [1, 3, 5], [2,4]. Excludes [NOT FOUND: ...]
# and other non-numeric brackets (the digit-only pattern won't match them).
# Leading spaces are captured separately so a fully-dangling marker is dropped
# together with the space in front of it (mirrors rag._CITE_MARKER_RE).
_MARKER_RE = re.compile(r"([ \t]*)\[(\d+(?:\s*,\s*\d+)*)\]")


def _remap_markers(body: str, cites: list[dict], registry: dict) -> str:
    """Rewrite a section's section-local [N] markers to document-global numbers.

    Each section is synthesized independently and numbers its citations from 1,
    so the same [N] means different papers across sections. We key each cited
    paper by title in a shared `registry` (assigning a stable global number on
    first sighting), then rewrite this section's inline markers through the
    local→global map. Markers with no matching citation (the synthesizer
    over-numbered, e.g. a stray [5]) are dropped rather than left dangling.
    """
    local_to_global: dict[str, int] = {}
    for c in cites:
        title = c.get("title", "Unknown")
        local = str(c.get("number"))
        if title not in registry:
            registry[title] = len(registry) + 1
        local_to_global[local] = registry[title]

    def _repl(m: re.Match) -> str:
        mapped: list[int] = []
        for part in m.group(2).split(","):
            g = local_to_global.get(part.strip())
            if g is not None and g not in mapped:
                mapped.append(g)
        if not mapped:  # every number in this marker was dangling
            return ""  # drops the preceding space too — no double/trailing space
        return m.group(1) + "[" + ", ".join(str(g) for g in mapped) + "]"

    return _MARKER_RE.sub(_repl, body)


def _render_global_refs(registry: dict) -> str:
    """One document-wide References list, ordered by global citation number."""
    if not registry:
        return ""
    lines = [f"- [{n}] {title}" for title, n in sorted(registry.items(), key=lambda kv: kv[1])]
    return "## References\n\n" + "\n".join(lines)


def run_report(topic: str, language: str = "en", progress_cb=None) -> dict:
    """Plan, synthesize, and assemble a cited Markdown literature review.

    Returns ``{topic, language, sections, markdown, citation_count}``.
    ``progress_cb(current, total, message)`` is called before each section.
    """
    sections = plan_sections(topic, language)
    total = len(sections)
    parts = [f"# Literature Review: {topic}\n"]
    # Document-global citation registry (title -> global number), shared across
    # sections so the same paper keeps one number throughout the report.
    registry: dict[str, int] = {}

    for i, sec in enumerate(sections):
        if progress_cb:
            progress_cb(i, total, f"Writing section: {sec}")
        # ponytail: section query = topic + aspect; answer_question detects the
        # language from this query (so a Hindi topic yields Hindi sections).
        query = f"{sec} — in the context of {topic}"
        try:
            res = rag.answer_question(query, strategy="A")
            body = res.get("answer", "").strip()
            cites = res.get("citations", [])
        except Exception as e:
            logger.error(f"[Report] section {sec!r} failed: {e}")
            body, cites = "_(section synthesis failed)_", []
        body = _remap_markers(body, cites, registry)
        parts.append(f"## {sec}\n\n{body}")

    if progress_cb:
        progress_cb(total, total, "Assembling report")
    refs = _render_global_refs(registry)
    if refs:
        parts.append(refs)
    markdown = "\n\n".join(parts) + "\n"
    citation_count = len(registry)  # unique sources across the whole report
    logger.info(f"[Report] {topic!r}: {total} sections, {citation_count} citations")
    return {
        "topic": topic,
        "language": language,
        "sections": sections,
        "markdown": markdown,
        "citation_count": citation_count,
    }


if __name__ == "__main__":  # ponytail: runnable self-check, no framework
    import unittest.mock as mock
    # Two sections cite the same paper "P" under *different* local numbers, plus
    # a second paper "Q" and a dangling [5]. Global numbering must: give P one
    # number everywhere, add Q once, and drop the dangling marker.
    _answers = iter([
        {"answer": "alpha [1]", "citations": [{"number": "1", "title": "P"}]},
        {"answer": "beta [1] gamma [2] delta [5]",
         "citations": [{"number": "1", "title": "Q"}, {"number": "2", "title": "P"}]},
    ])
    with mock.patch.object(rag, "llm_generate", return_value='["Background", "Methods", "Background"]'), \
         mock.patch.object(rag, "answer_question", side_effect=lambda *a, **k: next(_answers)):
        assert plan_sections("x") == ["Background", "Methods"], "dedupe/parse broken"
        r = run_report("graphene sensors")
        assert r["sections"] == ["Background", "Methods"]
        assert r["citation_count"] == 2, f"expected 2 unique sources, got {r['citation_count']}"
        md = r["markdown"]
        assert "# Literature Review: graphene sensors" in md
        # P=1 (first seen), Q=2. Section 2's local [1]->Q=2, [2]->P=1, [5] dropped.
        assert "alpha [1]" in md, "P should be global [1] in section 1"
        assert "beta [2] gamma [1] delta" in md, f"remap/drop wrong: {md!r}"
        assert "delta [" not in md, "dangling [5] not dropped"
        assert "delta " not in md, f"dropped marker left a stray space: {md!r}"
        assert "## References" in md
        assert "- [1] P" in md and "- [2] Q" in md
    # planner fallback when LLM returns junk
    with mock.patch.object(rag, "llm_generate", return_value="not json at all"):
        assert plan_sections("x") == _DEFAULT_SECTIONS
    print("report_runner self-check OK")
