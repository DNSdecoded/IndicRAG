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
    prompt = (
        f"You are planning a literature-review report on: {topic!r}.\n"
        f"Propose at most {max_sections} section titles (e.g. background, methods "
        f"comparison, key findings, open gaps). Reply with ONLY a JSON array of "
        f'short title strings, e.g. ["Background", "Methods", "Findings"].'
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


def _render_sources(citations: list[dict]) -> str:
    """A per-section Sources list. Citation numbers are section-local, so each
    section keeps its own list; renumbering across the report would break the
    inline [N] markers the synthesizer already wrote."""
    if not citations:
        return ""
    lines = [f"- [{c.get('number')}] {c.get('title', 'Unknown')}" for c in citations]
    return "\n\n**Sources:**\n\n" + "\n".join(lines)


def run_report(topic: str, language: str = "en", progress_cb=None) -> dict:
    """Plan, synthesize, and assemble a cited Markdown literature review.

    Returns ``{topic, language, sections, markdown, citation_count}``.
    ``progress_cb(current, total, message)`` is called before each section.
    """
    sections = plan_sections(topic, language)
    total = len(sections)
    parts = [f"# Literature Review: {topic}\n"]
    citation_count = 0

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
        citation_count += len(cites)
        parts.append(f"## {sec}\n\n{body}{_render_sources(cites)}")

    if progress_cb:
        progress_cb(total, total, "Assembling report")
    markdown = "\n\n".join(parts) + "\n"
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
    with mock.patch.object(rag, "llm_generate", return_value='["Background", "Methods", "Background"]'), \
         mock.patch.object(rag, "answer_question",
                           return_value={"answer": "text [1]", "citations": [{"number": "1", "title": "P"}]}):
        assert plan_sections("x") == ["Background", "Methods"], "dedupe/parse broken"
        r = run_report("graphene sensors")
        assert r["sections"] == ["Background", "Methods"]
        assert r["citation_count"] == 2
        assert "# Literature Review: graphene sensors" in r["markdown"]
        assert "**Sources:**" in r["markdown"]
    # planner fallback when LLM returns junk
    with mock.patch.object(rag, "llm_generate", return_value="not json at all"):
        assert plan_sections("x") == _DEFAULT_SECTIONS
    print("report_runner self-check OK")
