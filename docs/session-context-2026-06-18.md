# IndicRAG — Session Context (2026-06-17 → 2026-06-18)

A running record of what was done in this working session, for continuity in future sessions.

---

## 1. UI redesign (`static/index.html`)

Iterated on the color palette several times:
1. Teal/green (original) → indigo/amber → earthy terracotta → **final: slate/charcoal** (user-provided exact hex spec, light + dark mode).
2. Final palette: `--primary: #334155`, slate backgrounds (`#F5F7FA` light / `#0B1220` dark), charcoal user bubbles, neutral chips.
3. **Logo** changed from "IR" text to Devanagari **ज्ञ** (jnana = knowledge), in `.logo-wrap` using `'Noto Sans Devanagari'` font.
4. **Bot avatar** icon changed from network/share graph → lightbulb SVG (idea/knowledge motif).

User feedback noted: disliked "robo"/AI-generated-looking palettes; settled on the clean slate spec they supplied directly.

## 2. Git / GitHub work

- Repo: `https://github.com/DNSdecoded/IndicRAG`, branch `main`, user `sanjay sakhinala`.
- Committed UI changes and backend updates (api_server.py, config.py, rag.py, requirements.txt, vector_store.py).
- Added `graphify-out/` to `.gitignore`.
- Created PR #4 (`feat/ui-redesign-and-backend-update`) for CodeRabbit review; it was merged.
- `gh` CLI was not installed → installed GitHub CLI via MSI (winget was unresponsive); PR opened via browser.
- **Release v1.5.0**: `config.VERSION = "1.5.0"`, annotated tag `v1.5.0` pushed; GitHub release drafted via browser.
- **Removed `Co-Authored-By: Claude`** from commit history (rebase + force-push) and deleted the merged feature branch. GitHub contributor cache takes up to ~24h to drop Claude.

### Standing preference (saved to memory)
**Do not add `Co-Authored-By: Claude` trailers to commits.** User does not want Claude appearing as a GitHub contributor.

## 3. graphify run (interrupted)

- Ran `/graphify .` on the repo. `.venv` was auto-skipped → clean corpus of **37 files (~30K words): 21 code, 16 docs**.
- AST extraction: 233 nodes / 434 edges. One general-purpose subagent extracted docs: +116 semantic nodes.
- Merged total: **307 nodes, 548 edges**. Build/cluster step (Step 4) was interrupted before `graph.json` was finalized — graphify output is incomplete and lives in gitignored `graphify-out/`.
- Helper scripts left in `graphify-out/`: `_ast.py`, `_cache.py`, `_merge.py`, `_build.py` (all gitignored).

## 4. v2.0 Agentic Scientific RAG — planning (main deliverable)

Ran the brainstorming skill. Decisions captured from the user:

| Decision | Choice |
|----------|--------|
| Core capabilities | All four: query decomposition + iterative retrieval, tool use, self-verification, autonomous workflows |
| Orchestration | **Custom loop on Gemini function-calling** (no framework) |
| Multilingual | **Config toggle** keeping both Strategy A (native) and B (English-pivot) |
| Tool scope | **Corpus + external scholarly + code execution** (phased) |
| UX/latency | **Streaming agent steps** (SSE) |
| Structure | **Phased across v2.0 → v2.2** |
| Agent pattern | **Planner → Executor → Verifier → Synthesizer** (approved) |

### Output: `docs/feature-requests/v2.0-agentic-rag/`
- **`roadmap.md`** — vision, north-star architecture, v2.0/v2.1/v2.2 phases with scope + acceptance criteria, dependency order, risks, success metric.
- **`planning.md`** — current-state map, target `agent/` package layout, the loop, `Tool`/`ToolRegistry` contract, `AgentState`, A/B handling, API changes (`/agent`, `/agent/stream`), data flow, testing strategy.
- **`instruction.md`** — 11-step v2.0 build sequence (each independently testable) + v2.1/v2.2 follow-on + conventions.

### Phase summary
- **v2.0 Agentic Core:** decomposition, corpus tools, self-verification + re-retrieval, SSE streaming, both A/B modes, UI step panels. `/query` + `/chat` stay back-compatible; agent is additive.
- **v2.1 External Knowledge:** arXiv / Semantic Scholar / web search, corpus-vs-external provenance, on-the-fly ingestion.
- **v2.2 Execution + Autonomy:** sandboxed code/math execution, async multi-paper literature review + report generation, optional multi-agent.

### Next step
Invoke the **`writing-plans`** skill against `docs/feature-requests/v2.0-agentic-rag/` to turn the v2.0 build sequence into a concrete implementation plan. (Pending user review of the three docs.)

---

## Current state of the tree (uncommitted)
- `static/index.html`, backend modules, `.gitignore`, `config.py` (VERSION 1.5.0) — committed & pushed to `main`.
- New: `docs/feature-requests/v2.0-agentic-rag/{roadmap,planning,instruction}.md` and this `docs/session-context-2026-06-18.md` — not yet committed.
- `graphify-out/` — gitignored, partial.
