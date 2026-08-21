"""
Main RAG pipeline: retrieval and answer generation.
"""

from typing import Dict, List, Optional, Any
import logging
import re
import config
import embeddings
import vector_store
import lang_utils
import translation
import llm_client
import metrics
from google.genai import types

logger = logging.getLogger(__name__)


def citation_number_map(metadatas: List[Dict]) -> Dict[int, Dict]:
    """Map a per-paper citation number → representative metadata.

    Citations are numbered by unique paper (first-seen title order), NOT by
    chunk index. Several chunks of the same paper share one [Cite:N], so the
    marker in the answer resolves to exactly one source in the panel. This is
    the single source of truth used by format_context, extract_citations, and
    the agent sources builder — they MUST agree or citation numbers drift.
    """
    title_to_num: Dict[str, int] = {}
    num_to_meta: Dict[int, Dict] = {}
    for meta in metadatas:
        title = (meta.get('title') or 'Unknown').strip() or 'Unknown'
        if title not in title_to_num:
            num = len(title_to_num) + 1
            title_to_num[title] = num
            num_to_meta[num] = meta
    return num_to_meta


def _crop_url(crop_path: str) -> Optional[str]:
    """Map a stored crop path to its /figures URL, or None if outside FIGURES_DIR.

    The path segments are percent-encoded so a crafted filename can't break out
    of an HTML attribute when the URL is rendered into <img src>/<a href>.
    """
    from pathlib import Path
    from urllib.parse import quote
    try:
        rel = Path(crop_path).resolve().relative_to(config.FIGURES_DIR.resolve())
        return "/figures/" + quote(rel.as_posix())
    except (ValueError, OSError):
        return None


def extract_citations(answer: str, metadatas: List[Dict], chunks: List[str] = None,
                      visible_chunks: int = None) -> List[Dict]:
    """
    Extract [Cite:N] citations from answer text and resolve them to papers.

    Numbers are per unique paper (see citation_number_map), so [Cite:2] means
    "the 2nd distinct paper in the context", not "the 2nd chunk".

    Args:
        answer: Generated answer text containing citations
        metadatas: List of metadata dictionaries from retrieved chunks
        chunks: Unused; kept for backwards-compatible call sites
        visible_chunks: How many leading chunks actually reached the prompt
            (``format_context``'s ``chunks_used``). Callers hold the FULL
            retrieved metadata, but format_context truncates by chunk count and
            by total length — so without this, a number the model invented past
            the truncation point resolves to a real paper it was never shown,
            and the answer carries a citation that looks legitimate. Numbering
            the visible slice only makes such a marker dangle, and dangling
            markers are dropped. ``None`` means "truncation unknown", use all.

    Returns:
        List of citation dictionaries with number, title, and section
    """
    import re

    if visible_chunks is not None:
        metadatas = metadatas[:visible_chunks]
        if chunks:
            chunks = chunks[:visible_chunks]

    seen_nums = set()
    # Match [N] and comma-separated [N, N, ...] citation markers.
    # Still ignores ranges like [10-15] mg (no comma, so no match).
    for m in re.finditer(r'\[(\d+(?:\s*,\s*\d+)*)\]', answer):
        for part in m.group(1).split(","):
            try:
                seen_nums.add(int(part.strip()))
            except ValueError:
                pass

    num_to_meta = citation_number_map(metadatas)

    # All sections a paper contributed, in retrieval order — labeling with only
    # the first-seen chunk's section made an 11-section answer read as if it
    # came from the introduction alone.
    title_sections: Dict[str, list] = {}
    for m_ in metadatas:
        t = (m_.get('title') or 'Unknown').strip() or 'Unknown'
        s = m_.get('section', 'body')
        if s not in title_sections.setdefault(t, []):
            title_sections[t].append(s)

    # Phase 3: figure/table crops a paper contributed, so the UI can render them.
    figures_by_paper: Dict[str, list] = {}
    for m_ in metadatas:
        if m_.get('chunk_type') in ('figure', 'table') and m_.get('crop_path'):
            url = _crop_url(m_['crop_path'])
            if url:
                figures_by_paper.setdefault(m_.get('paper_id'), []).append({
                    'page': m_.get('page'),
                    'chunk_type': m_.get('chunk_type'),
                    'url': url,
                })

    citations = []
    for num in sorted(seen_nums):
        meta = num_to_meta.get(num)
        if meta:
            title = (meta.get('title') or 'Unknown').strip() or 'Unknown'
            sections = title_sections.get(title) or [meta.get('section', 'body')]
            citations.append({
                'number': str(num),
                'title': meta.get('title', 'Unknown'),
                'section': ', '.join(sections),
                'figures': figures_by_paper.get(meta.get('paper_id'), []),
            })
    return citations


# Inline citation markers: [3], [1, 3, 5], [2,4]. Digit-only, so [NOT FOUND: ...]
# and ranges like [10-15] never match. Leading whitespace is captured separately so
# a fully-dangling marker is dropped together with the space in front of it — and
# the optional newline is included because a marker alone on its own line otherwise
# left the newline behind, which markdown renders as a paragraph break.
_CITE_MARKER_RE = re.compile(r'([ \t]*\n?[ \t]*)\[(\d+(?:\s*,\s*\d+)*)\]')


def compact_citations(answer: str, metadatas: List[Dict], chunks: List[str] = None,
                      visible_chunks: int = None):
    """extract_citations, then renumber the survivors to a dense 1..M sequence.

    format_context numbers EVERY retrieved paper, but only the papers the answer
    actually cites reach the citation panel — so an answer drawing on papers 1
    and 4 of 4 read "[1] ... [4]" beside a two-entry panel. Renumber the cited
    papers in context order and rewrite the answer's markers to match. Markers
    that resolve to no paper (the model over-numbered) are dropped rather than
    left dangling, mirroring report_runner._remap_markers.

    Returns ``(rewritten answer, citations)`` — the citations carry the new
    dense numbers, so callers must use the returned answer, not the original.
    """
    citations = extract_citations(answer, metadatas, chunks, visible_chunks)
    old_to_new = {int(c['number']): i for i, c in enumerate(citations, 1)}

    def _repl(m: "re.Match") -> str:
        mapped: List[int] = []
        for part in m.group(2).split(','):
            try:
                new = old_to_new.get(int(part.strip()))
            except ValueError:
                new = None
            if new is not None and new not in mapped:
                mapped.append(new)
        if not mapped:  # every number in this marker was dangling
            # Drop the preceding space too — no double/trailing space. When the
            # marker sat alone on its line, the line's own trailing newline
            # survives, so the captured leading newline must go with the marker
            # or a blank line is left behind (markdown reads it as a paragraph
            # break). Otherwise put it back, or the lines splice together.
            if '\n' not in m.group(1):
                return ''
            rest = m.string[m.end():]
            return '' if (rest == '' or rest[0] == '\n') else '\n'
        return m.group(1) + '[' + ', '.join(str(n) for n in mapped) + ']'

    for i, c in enumerate(citations, 1):
        c['number'] = str(i)
    return _CITE_MARKER_RE.sub(_repl, answer), citations


def _hyde_embedding(user_query: str):
    """Draft a hypothetical answer and embed it, for HyDE retrieval.

    Bridges the lexical gap for complex/multi-hop queries: the hypothetical
    answer's vocabulary overlaps documents more than the bare question does.
    Falls back to embedding the raw query on any LLM failure.
    """
    try:
        hyde_config = types.GenerateContentConfig(
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=256,
            safety_settings=config.SAFETY_SETTINGS,
            # Throwaway hypothetical draft for embedding — thinking is wasted spend.
            thinking_config=llm_client.thinking_config_for("standard"),
        )
        response = llm_client.generate_with_failover(
            config.LLM_MODEL_NAME,
            f"Write a short, plausible-sounding answer to this question, "
            f"even if you are not sure it is correct:\n\n{user_query}",
            hyde_config,
        )
        hypothetical = safe_extract_text(response)
        if hypothetical:
            return embeddings.embed_query(hypothetical)
    except Exception as e:
        logger.debug(f"HyDE draft failed, falling back to direct query embedding: {e}")
    return embeddings.embed_query(user_query)


_TAGS_SENTINEL = "$tags_post_filter"


def _extract_tags_post_filter(filter_dict: Optional[Dict]) -> tuple:
    """Pull a tags post-filter out of an opaque filter_dict blob, if present.

    Tags can't be a ChromaDB `where` clause: PATCH /papers stores the tags
    string verbatim (unsplit), so a native $in match against split tag names
    never equals the stored comma-joined value — it would silently return
    zero results for any paper tagged with more than one tag. Instead, tags
    travel inside filter_dict as a reserved sentinel key and are applied as
    a Python-side post-filter in retrieve_context, while paper_id/year
    clauses still go to ChromaDB as before.

    Returns (chromadb_safe_filter_dict_or_None, tag_list_or_None).
    """
    if not filter_dict:
        return filter_dict, None
    if _TAGS_SENTINEL in filter_dict:
        return None, filter_dict[_TAGS_SENTINEL]
    if "$and" in filter_dict:
        tags = None
        remaining = []
        for clause in filter_dict["$and"]:
            if _TAGS_SENTINEL in clause:
                tags = clause[_TAGS_SENTINEL]
            else:
                remaining.append(clause)
        if tags is not None:
            if not remaining:
                return None, tags
            return (remaining[0] if len(remaining) == 1 else {"$and": remaining}), tags
    return filter_dict, None


_TAGS_POST_FILTER_KEYS = ("ids", "chunks", "documents", "metadatas", "distances")


def _copy_retrieval_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-copy a retrieval result and its parallel lists.

    The chunk strings and metadata dicts inside are still shared — callers read
    those — but the lists themselves are per-caller, so slicing or reordering a
    returned result can't mutate the cached entry.
    """
    copied = dict(result)
    for key in _TAGS_POST_FILTER_KEYS:
        value = copied.get(key)
        if isinstance(value, list):
            copied[key] = list(value)
    return copied


def _apply_tags_post_filter(results: Dict[str, list], tags: list) -> Dict[str, list]:
    """Keep only results whose stored `tags` metadata shares at least one tag.

    Only touches the parallel list-valued keys (chunks/metadatas/distances/...);
    any other key (formatted_context, chunks_used, ...) passes through unchanged.
    """
    wanted = {t.strip() for t in tags if t.strip()}
    if not wanted:
        return results
    keep = [
        i for i, meta in enumerate(results.get("metadatas", []))
        if wanted & {t.strip() for t in (meta or {}).get("tags", "").split(",") if t.strip()}
    ]
    filtered = dict(results)
    for key in _TAGS_POST_FILTER_KEYS:
        if key in filtered:
            filtered[key] = [filtered[key][i] for i in keep]
    return filtered


def retrieve_context(
    user_query: str,
    top_k: int = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    collection=None,
    use_hyde: bool = None,
) -> Dict[str, Any]:
    """
    Retrieve relevant context for a user query.
    
    Args:
        user_query: User's question
        top_k: Number of chunks to retrieve (default from config)
        filter_dict: Optional metadata filter
        collection: ChromaDB collection (uses default if None)
        
    Returns:
        Dictionary with:
            - 'chunks': List of retrieved text chunks (empty if no documents)
            - 'metadatas': List of metadata dicts
            - 'distances': List of similarity distances
            - 'formatted_context': Formatted context string for LLM
            - 'chunks_used': Number of chunks actually used in formatted context
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K
    if use_hyde is None:
        use_hyde = config.USE_HYDE

    from cache import retrieval_cache, make_key
    cache_scope = None if collection is None else getattr(collection, "name", id(collection))
    cache_key = make_key(user_query, top_k, filter_dict, cache_scope,
                         config.USE_RERANKER, config.MAX_CONTEXT_CHUNKS, use_hyde)
    # Decide cacheability BEFORE `collection` is materialized below — the store
    # step used to re-test `collection is None`, which is never true by then, so
    # nothing was ever cached and every repeat query re-embedded and re-searched.
    cacheable = collection is None and filter_dict is None
    if cacheable:
        cached = retrieval_cache.get(cache_key)
        if cached is not None:
            logger.debug("[Retrieval cache hit]")
            # Hand out a copy: the cached entry is shared by every subsequent
            # hit, so a caller that trims/sorts the returned lists in place
            # would corrupt the cache for everyone else.
            return _copy_retrieval_result(cached)

    if collection is None:
        collection = vector_store.get_or_create_collection()

    # Tags can't go to ChromaDB as a native where-clause (see _extract_tags_post_filter).
    # chroma_filter_dict is the ChromaDB-safe remainder; `filter_dict` itself is left
    # untouched since the cache-bypass checks below key off "was any filter requested".
    chroma_filter_dict, tags_post_filter = _extract_tags_post_filter(filter_dict)

    # Paper-scoped queries → exhaustive retrieval (all chunks of the selected
    # papers in document order), not top-k similarity. This is what makes
    # "reconstruct everything in this paper" work: the LLM sees the whole paper
    # instead of a semantic sample that can miss equations/hyperparameters.
    if chroma_filter_dict and 'paper_id' in chroma_filter_dict:
        scoped = _retrieve_scoped(chroma_filter_dict, collection)
        if tags_post_filter and scoped["chunks"]:
            filtered = _apply_tags_post_filter(scoped, tags_post_filter)
            formatted_context, chunks_used = format_context(
                chunks=filtered["chunks"], metadatas=filtered["metadatas"])
            filtered["formatted_context"] = formatted_context
            filtered["chunks_used"] = chunks_used
            return filtered
        return scoped

    # Embed the query — HyDE embeds a drafted hypothetical answer instead of
    # the bare question; lexical (BM25) search below always uses the real query.
    #
    # A failure here (model OOM, corrupt weights, driver fault) used to fail the
    # whole request, including queries the in-process BM25 index could have
    # answered on its own. Degrade to sparse-only instead; the result carries a
    # `degraded` marker (logged now, surfaced in the API response as follow-up work).
    try:
        query_embedding = _hyde_embedding(user_query) if use_hyde else embeddings.embed_query(user_query)
    except Exception as e:
        logger.error(f"Embedding failed, trying sparse-only retrieval: {e}", exc_info=True)
        sparse = _sparse_only_retrieval(user_query, top_k, collection, chroma_filter_dict, tags_post_filter)
        if sparse is None:
            raise  # no BM25 index either — nothing to degrade to
        return sparse

    # A tags post-filter runs AFTER retrieval, so fetching exactly top_k returns
    # nothing whenever the tagged papers rank below the cut (measured: 1 tagged
    # doc in a 13-doc corpus, top_k=5 → 0 results). Widen the fetch when tags are
    # active and narrow back to top_k once the filter has been applied.
    search_k = top_k
    if tags_post_filter:
        search_k = min(top_k * config.TAGS_OVERFETCH, config.TAGS_OVERFETCH_MAX)

    # Search vector store (dense)
    with metrics.stage("retrieval_dense"):
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=search_k,
            filter_dict=chroma_filter_dict,
            collection=collection
        )

    # Hybrid: fuse dense results with BM25 lexical search
    if config.USE_HYBRID_SEARCH and results['documents'] and not chroma_filter_dict:
        try:
            import bm25_search
            with metrics.stage("bm25_index_load"):
                bm25_idx = bm25_search.get_or_build_index(collection)
            if bm25_idx is not None:
                with metrics.stage("retrieval_bm25"):
                    sparse_ids, _ = bm25_idx.search(user_query, top_k=search_k)
                fused_ids = bm25_search.rrf(results['ids'], sparse_ids, k=config.RRF_K)
                id_to_doc = dict(zip(results['ids'], results['documents']))
                id_to_meta = dict(zip(results['ids'], results['metadatas']))
                id_to_dist = dict(zip(results['ids'], results['distances']))
                # Fetch any BM25-only hits that dense search missed
                missing_ids = [i for i in fused_ids if i not in id_to_doc]
                if missing_ids:
                    extra = collection.get(ids=missing_ids, include=["documents", "metadatas"])
                    for eid, edoc, emeta in zip(extra['ids'], extra['documents'], extra['metadatas']):
                        id_to_doc[eid] = edoc
                        id_to_meta[eid] = emeta
                        id_to_dist[eid] = 1.0
                results = {
                    'ids': [i for i in fused_ids if i in id_to_doc][:search_k],
                    'documents': [id_to_doc[i] for i in fused_ids if i in id_to_doc][:search_k],
                    'metadatas': [id_to_meta[i] for i in fused_ids if i in id_to_doc][:search_k],
                    'distances': [id_to_dist.get(i, 1.0) for i in fused_ids if i in id_to_doc][:search_k],
                }
        except Exception as e:
            logger.debug(f"Hybrid search skipped: {e}")

    if tags_post_filter:
        results = _apply_tags_post_filter(results, tags_post_filter)
        # Back down to the caller's budget now that the filter has run.
        for key in _TAGS_POST_FILTER_KEYS:
            if key in results:
                results[key] = results[key][:top_k]

    # Check if search returned results
    if not results['documents']:
        logger.warning(f"No results found for query: {user_query[:50]}")
        return {
            'chunks': [],
            'metadatas': [],
            'distances': [],
            'formatted_context': '',
            'chunks_used': 0
        }

    docs = results['documents']
    metas = results['metadatas']
    dists = results['distances']

    if config.USE_COLBERT_RERANK and docs:
        import colbert_rerank
        dense_sims = [1.0 - d for d in dists]  # cosine distance -> similarity
        colbert_top_k = min(len(docs), config.MAX_CONTEXT_CHUNKS * 3)
        with metrics.stage("rerank_colbert"):
            docs, metas, dists = colbert_rerank.rerank(
                user_query, docs, metas, dense_sims,
                top_k=colbert_top_k, weight=config.COLBERT_WEIGHT)

    if config.USE_RERANKER and docs:
        import rerank
        with metrics.stage("rerank_cross_encoder"):
            docs, metas, scores = rerank.rerank(
                user_query, docs, metas, top_k=config.MAX_CONTEXT_CHUNKS)
        dists = scores

    # Format context for LLM
    formatted_context, chunks_used = format_context(
        chunks=docs,
        metadatas=metas
    )

    result = {
        'chunks': docs,
        'metadatas': metas,
        'distances': dists,
        'formatted_context': formatted_context,
        'chunks_used': chunks_used
    }
    if cacheable:
        # Store a copy, not the object being returned — otherwise the very first
        # caller still holds a handle on the cached lists.
        retrieval_cache.put(cache_key, _copy_retrieval_result(result))
    return result


def format_context(chunks: List[str], metadatas: List[Dict],
                   max_chunks: int = None, max_length: int = None) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Args:
        chunks: List of text chunks
        metadatas: List of metadata dictionaries
        max_chunks: Override for max chunks kept (default config.MAX_CONTEXT_CHUNKS)
        max_length: Override for max total chars (default config.MAX_CONTEXT_LENGTH)

    Returns:
        (formatted context string with citations, number of chunks kept)
    """
    if max_chunks is None:
        max_chunks = config.MAX_CONTEXT_CHUNKS
    if max_length is None:
        max_length = config.MAX_CONTEXT_LENGTH

    context_parts = []
    total_length = 0
    chunks_used = 0
    title_to_num: Dict[str, int] = {}  # per-paper citation numbers; see citation_number_map

    for chunk, metadata in zip(chunks, metadatas):
        # Enforce maximum number of chunks
        if chunks_used >= max_chunks:
            break

        title = (metadata.get('title') or 'Unknown').strip() or 'Unknown'
        section = metadata.get('section', 'body')

        # One citation number per unique paper — chunks of the same paper reuse it.
        num = title_to_num.get(title, len(title_to_num) + 1)
        context_part = f"[{num}] {title} - {section}:\n{chunk}\n"

        # Check if adding this would exceed length limit
        if total_length + len(context_part) > max_length:
            break

        title_to_num.setdefault(title, num)
        context_parts.append(context_part)
        total_length += len(context_part)
        chunks_used += 1

    return "\n".join(context_parts), chunks_used


def _sparse_only_retrieval(user_query: str, top_k: int, collection,
                           filter_dict: Dict[str, Any] = None,
                           tags_post_filter: list = None) -> Dict[str, Any] | None:
    """BM25-only retrieval, used when the dense leg is unavailable.

    Returns None when there is nothing to degrade to (hybrid search disabled, or
    no BM25 index because the corpus is empty) so the caller can re-raise the
    original failure rather than serve a silently empty answer.

    The result carries `degraded='sparse_only'` and logs at WARNING. Note that no
    route reads that key yet — surfacing it in the API response is still to do, so
    for now degradation is visible in the logs but not to the end user.
    """
    if not config.USE_HYBRID_SEARCH:
        return None
    try:
        import bm25_search
        idx = bm25_search.get_or_build_index(collection)
        if idx is None:
            return None
        ids, _scores = idx.search(user_query, top_k=top_k)
        if not ids:
            return None
        got = vector_store._chroma_call(
            collection.get, ids=ids, include=['documents', 'metadatas'])
    except Exception as e:
        logger.error(f"Sparse-only fallback failed too: {e}", exc_info=True)
        return None

    # collection.get() does not preserve the order of the ids it was handed, and
    # BM25 rank is the only ranking signal left here — restore it explicitly.
    by_id = dict(zip(got.get('ids', []), zip(got.get('documents', []), got.get('metadatas', []))))
    ordered = [(i, *by_id[i]) for i in ids if i in by_id]
    if not ordered:
        return None

    results = {
        'ids': [r[0] for r in ordered],
        'chunks': [r[1] for r in ordered],
        'documents': [r[1] for r in ordered],
        'metadatas': [r[2] for r in ordered],
        'distances': [1.0] * len(ordered),
    }
    if tags_post_filter:
        results = _apply_tags_post_filter(results, tags_post_filter)
        for key in _TAGS_POST_FILTER_KEYS:
            if key in results:
                results[key] = results[key][:top_k]

    formatted_context, chunks_used = format_context(
        chunks=results['chunks'], metadatas=results['metadatas'])
    logger.warning("Serving DEGRADED sparse-only retrieval: %d chunks (dense leg unavailable)",
                   chunks_used)
    return {
        'chunks': results['chunks'],
        'metadatas': results['metadatas'],
        'distances': results['distances'],
        'formatted_context': formatted_context,
        'chunks_used': chunks_used,
        'degraded': 'sparse_only',
    }


def _retrieve_scoped(filter_dict: Dict[str, Any], collection) -> Dict[str, Any]:
    """Exhaustive retrieval for paper-scoped queries.

    Returns ALL chunks of the scoped paper(s) in document order
    (paper_id, chunk_index), capped by config.SCOPED_MAX_CHUNKS /
    SCOPED_MAX_CONTEXT_LENGTH — no similarity truncation, no reranker. Preserves
    the paper's natural structure so reconstruction-style queries see the whole
    document. Skips dense/BM25 entirely (no query embedding needed).
    """
    got = vector_store._chroma_call(
        collection.get, where=filter_dict, include=['documents', 'metadatas'])
    ids = got.get('ids', [])
    if not ids:
        logger.warning(f"No chunks for scoped filter {filter_dict}")
        return {'chunks': [], 'metadatas': [], 'distances': [],
                'formatted_context': '', 'chunks_used': 0}

    metas_all = got['metadatas']
    order = sorted(
        range(len(ids)),
        key=lambda i: (metas_all[i].get('paper_id', ''), metas_all[i].get('chunk_index', 0)),
    )[:config.SCOPED_MAX_CHUNKS]

    docs = [got['documents'][i] for i in order]
    metas = [metas_all[i] for i in order]

    formatted_context, chunks_used = format_context(
        docs, metas,
        max_chunks=config.SCOPED_MAX_CHUNKS,
        max_length=config.SCOPED_MAX_CONTEXT_LENGTH,
    )
    logger.info(f"Scoped retrieval: {len(ids)} chunks matched, {chunks_used} used "
                f"(filter={filter_dict})")
    return {
        'chunks': docs,
        'metadatas': metas,
        'distances': [0.0] * len(docs),
        'formatted_context': formatted_context,
        'chunks_used': chunks_used,
    }


def compare_papers(paper_ids: List[str], dimensions: List[str], model: str = None) -> Dict[str, Any]:
    """Build a papers x dimensions comparison matrix.

    Reuses the paper-scoped exhaustive retrieval (_retrieve_scoped) to see the
    whole paper per row, then asks one grounded extraction per dimension.
    Returns {"dimensions": [...], "matrix": {paper_id: {dimension: text}}}.
    """
    collection = vector_store.get_or_create_collection()

    matrix: Dict[str, Dict[str, str]] = {}
    for paper_id in paper_ids:
        scoped = _retrieve_scoped({"paper_id": {"$in": [paper_id]}}, collection)
        context = scoped["formatted_context"]
        matrix[paper_id] = {}
        if not context:
            for dim in dimensions:
                matrix[paper_id][dim] = "N/A — paper not found in corpus"
            continue
        for dim in dimensions:
            prompt = (
                f"Based ONLY on the following paper text, extract the information about: {dim}\n"
                f"If the paper does not address this dimension, say 'N/A'.\n"
                f"Be concise — 1-3 sentences. Cite with [1] if possible.\n\n{context}"
            )
            matrix[paper_id][dim] = llm_generate(prompt, max_tokens=300, model=model)

    return {"dimensions": dimensions, "matrix": matrix}


def build_prompt(
    user_query: str,
    context: str,
    target_lang: str,
    strategy: str = "A"
) -> str:
    """
    Build the prompt for the LLM.
    
    Args:
        user_query: User's question
        context: Formatted context from retrieval
        target_lang: Target language code (e.g., 'hi', 'ta')
        strategy: "A" for multilingual LLM, "B" for English + translation
        
    Returns:
        Complete prompt string
    """
    if strategy == "B":
        lang_name = "English"
    else:
        # Get language name
        lang_name = lang_utils.get_language_name(target_lang)
        # Guard against garbled names
        if not lang_name.isascii() and lang_name == target_lang:
            lang_name = f"{target_lang} (language code)"
            
    return config.QUERY_PROMPT_TEMPLATE.format(
        context=context,
        question=user_query,
        language=lang_name
    )


def llm_generate(prompt: str, max_tokens: int = None,
                 system_instruction: str = None,
                 model: str = None, provider: str = None) -> str:
    """
    Generate response from LLM with provider/model failover.

    Args:
        prompt: The complete prompt to send to the LLM
        max_tokens: Maximum tokens to generate
        system_instruction: Optional system prompt override
        model: Optional LLM model id (from the /models allowlist). Omit for default.
        provider: Optional provider override (gemini|openrouter). Usually inferred.

    Returns:
        Generated text response

    Raises:
        ValueError: If API key is not configured
        Exception: If API call fails
    """
    if max_tokens is None:
        max_tokens = config.LLM_MAX_TOKENS

    target_model = model or config.LLM_MODEL_NAME

    from cache import llm_cache, make_key
    # Include model and provider so different models/providers never share a
    # cached answer, and the system instruction so a caller that overrides it
    # doesn't collide with the default-prompt entry for the same user prompt.
    cache_key = make_key(prompt, max_tokens, config.LLM_TEMPERATURE, target_model, provider,
                         system_instruction or config.SYSTEM_PROMPT)
    cached = llm_cache.get(cache_key)
    if cached is not None:
        logger.debug("[LLM cache hit]")
        return cached

    generate_config = types.GenerateContentConfig(
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        safety_settings=config.SAFETY_SETTINGS,
        system_instruction=system_instruction or config.SYSTEM_PROMPT,
        # Minimise thinking so the token budget goes to the answer, not thoughts.
        # Sending nothing here would mean the model's own default (MEDIUM on
        # gemini-3.6-flash), whose thoughts come out of max_output_tokens.
        thinking_config=llm_client.thinking_config_for("standard"),
    )

    try:
        response = llm_client.generate_with_failover(target_model, prompt, generate_config, provider=provider)

        # Check if response has text
        if response.text:
            llm_cache.put(cache_key, response.text)
            return response.text

        # Handle blocked or empty responses
        if response.candidates:
            candidate = response.candidates[0]

            # Try to get partial text if available
            if candidate.content and candidate.content.parts:
                parts_text = [
                    part.text
                    for part in candidate.content.parts
                    if hasattr(part, 'text') and part.text
                ]
                if parts_text:
                    result_text = ''.join(parts_text)
                    llm_cache.put(cache_key, result_text)
                    return result_text

            finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
            raise Exception(
                f"Response blocked or incomplete. Finish reason: {finish_reason}. "
                f"This may be due to safety filters or token limits."
            )

        raise Exception("No response generated from Gemini API")

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise


llm_generate_stream = llm_client.llm_generate_stream


def prepare_query_for_stream(user_query: str, strategy: str = "A", top_k: int = None,
                             filter_dict: Optional[Dict] = None) -> dict:
    """Retrieve context and build prompt for /query/stream.

    Returns dict with keys:
      chunks_used, prompt, metadatas, detected_lang, lang_name
    If no docs: chunks_used=0, no_docs_msg set instead of prompt/metadatas.

    filter_dict scopes retrieval (e.g. {'paper_id': {'$in': [...]}}) so a query
    can be restricted to specific papers.
    """
    detected_lang = lang_utils.detect_language(user_query) or "en"
    lang_name = lang_utils.get_language_name(detected_lang)

    retrieval_query = user_query
    if strategy == "B" and detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
        try:
            retrieval_query = translation.translate_to_english(user_query, detected_lang)
        except Exception:
            pass

    context_data = retrieve_context(retrieval_query, top_k, filter_dict)

    if context_data["chunks_used"] == 0:
        no_docs_msg = config.NO_DOCUMENTS_RESPONSE
        if detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
            try:
                no_docs_msg = translation.translate_from_english(no_docs_msg, detected_lang)
            except Exception:
                pass
        return {"chunks_used": 0, "no_docs_msg": no_docs_msg, "detected_lang": detected_lang, "lang_name": lang_name}

    prompt_query = retrieval_query if strategy == "B" else user_query
    prompt = build_prompt(user_query=prompt_query, context=context_data["formatted_context"],
                          target_lang=detected_lang, strategy=strategy)
    return {"chunks_used": context_data["chunks_used"], "prompt": prompt,
            "metadatas": context_data["metadatas"], "detected_lang": detected_lang,
            "lang_name": lang_name, "degraded": context_data.get("degraded")}


def prepare_chat_for_stream(messages: List[Dict[str, str]], strategy: str = "A", top_k: int = None,
                            filter_dict: Optional[Dict] = None) -> dict:
    """Retrieve context and build prompt for /chat/stream (mirrors answer_with_history).

    Returns same shape as prepare_query_for_stream. filter_dict scopes retrieval
    to specific papers (e.g. {'paper_id': {'$in': [...]}}).
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Last message must be from the user")

    user_query = messages[-1]["content"]
    prior = messages[:-1]
    detected_lang = lang_utils.detect_language(user_query) or "en"
    lang_name = lang_utils.get_language_name(detected_lang)

    retrieval_query = user_query
    if strategy == "B" and detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
        try:
            retrieval_query = translation.translate_to_english(user_query, detected_lang)
        except Exception:
            pass

    context_data = retrieve_context(retrieval_query, top_k, filter_dict)

    if context_data["chunks_used"] == 0:
        no_docs_msg = config.NO_DOCUMENTS_RESPONSE
        if detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
            try:
                no_docs_msg = translation.translate_from_english(no_docs_msg, detected_lang)
            except Exception:
                pass
        return {"chunks_used": 0, "no_docs_msg": no_docs_msg, "detected_lang": detected_lang, "lang_name": lang_name}

    # Build history string (same as answer_with_history)
    max_msgs = config.CHAT_HISTORY_MAX_TURNS * 2
    trimmed = prior[-max_msgs:] if len(prior) > max_msgs else prior
    history_lines = []
    turn, i = 1, 0
    while i < len(trimmed):
        if trimmed[i]["role"] == "user":
            user_line = f"[Turn {turn}] User: {trimmed[i]['content']}"
            if i + 1 < len(trimmed) and trimmed[i + 1]["role"] == "assistant":
                history_lines.append(f"{user_line}\n[Turn {turn}] Assistant: {trimmed[i + 1]['content']}")
                i += 2
            else:
                history_lines.append(user_line)
                i += 1
            turn += 1
        else:
            i += 1

    prompt_query = retrieval_query if strategy == "B" else user_query
    prompt = build_prompt(user_query=prompt_query, context=context_data["formatted_context"],
                          target_lang=detected_lang, strategy=strategy)
    history_str = "\n\n".join(history_lines)
    if history_str:
        prompt = f"## Conversation History\n{history_str}\n\n---\n\n{prompt}"

    return {"chunks_used": context_data["chunks_used"], "prompt": prompt,
            "metadatas": context_data["metadatas"], "detected_lang": detected_lang,
            "lang_name": lang_name, "degraded": context_data.get("degraded")}


generate_with_failover = llm_client.generate_with_failover


def safe_extract_text(response) -> str:
    """Safely extract text from a google-genai response, handling empty/blocked responses."""
    try:
        if response.text:
            return response.text
    except (ValueError, AttributeError):
        pass
    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            parts = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
            if parts:
                return "".join(parts)
    return ""


def _run_faithfulness(answer: str, chunks: List[str], metadatas: List[Dict] = None) -> dict:
    """Run faithfulness verification if configured; log warnings for ungrounded claims.

    metadatas (aligned with chunks) lets each [N] resolve to the right paper's
    chunk(s), since citations are numbered per-paper, not per-chunk.

    Returns {"claims": [...], "confidence": float} — confidence is the mean
    per-claim support score, surfaced to callers as answer_confidence.
    """
    try:
        import verify
        with metrics.stage("nli_verify"):
            results = verify.check_claims(answer, chunks, metadatas)
        for r in results:
            if not r["grounded"]:
                logger.warning(f"Ungrounded claim (score={r['support']:.2f}): {r['claim'][:120]}")
        confidence = sum(r["support"] for r in results) / len(results) if results else 0.0
        return {"claims": results, "confidence": round(confidence, 4)}
    except Exception as e:
        logger.warning(f"Faithfulness check failed: {e}", exc_info=True)
        return {"claims": [], "confidence": 0.0}


def answer_question_strategy_a(
    user_query: str,
    top_k: int = None,
    filter_dict: Optional[Dict] = None,
    model: str = None,
    provider: str = None,
) -> Dict[str, Any]:
    """
    Answer question using Strategy A: Direct multilingual LLM.
    
    Args:
        user_query: User's question in any language
        top_k: Number of chunks to retrieve
        filter_dict: Optional metadata filter
        
    Returns:
        Dictionary with:
            - 'answer': Generated answer in user's language
            - 'language': Detected language code
            - 'language_name': Native language name
            - 'chunks_used': Number of context chunks used
            - 'citations': List of cited papers
    """
    # Detect language
    detected_lang = lang_utils.detect_language(user_query)
    if not detected_lang:
        detected_lang = "en"  # Default to English
    
    lang_name = lang_utils.get_language_name(detected_lang)
    
    logger.info(f"Detected language: {lang_name} ({detected_lang})")
    
    # Retrieve context
    logger.info("Retrieving relevant context...")
    context_data = retrieve_context(user_query, top_k, filter_dict)
    
    # Handle empty collection
    if context_data['chunks_used'] == 0:
        logger.warning("No documents available for answering question")
        
        # Translate the no documents response if it's an indicative language
        no_docs_msg = config.NO_DOCUMENTS_RESPONSE
        if detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
             try:
                 no_docs_msg = translation.translate_from_english(no_docs_msg, detected_lang)
             except Exception:
                 pass # Fallback to English if translation fails
        
        return {
            'answer': no_docs_msg,
            'language': detected_lang,
            'language_name': lang_name,
            'chunks_used': 0,
            'citations': []
        }
    
    logger.info(f"Retrieved {len(context_data['chunks'])} chunks, using {context_data['chunks_used']}")
    
    # Build prompt
    prompt = build_prompt(
        user_query=user_query,
        context=context_data['formatted_context'],
        target_lang=detected_lang,
        strategy="A"
    )
    
    # Generate answer
    logger.info("Generating answer...")
    answer = llm_generate(prompt, model=model, provider=provider)
    
    # Extract citations using robust parser, compacting [1],[4] → [1],[2] so the
    # answer's markers match the cited-only panel.
    answer, citations = compact_citations(
        answer, context_data['metadatas'], context_data.get('chunks'),
        visible_chunks=context_data.get('chunks_used'))

    result = {
        'answer': answer,
        'language': detected_lang,
        'language_name': lang_name,
        'chunks_used': context_data['chunks_used'],
        'citations': citations,
        # None on the normal path; 'sparse_only' when the dense leg was
        # unavailable and this answer came from BM25 alone.
        'degraded': context_data.get('degraded'),
    }
    faith_result = _run_faithfulness(
        answer, context_data.get('chunks', []), context_data.get('metadatas', []))
    result['faithfulness'] = faith_result["claims"]
    result['answer_confidence'] = faith_result["confidence"]
    return result


def answer_question_strategy_b(
    user_query: str,
    top_k: int = None,
    filter_dict: Optional[Dict] = None,
    model: str = None,
    provider: str = None,
) -> Dict[str, Any]:
    """
    Answer question using Strategy B: English reasoning + translation.
    
    Args:
        user_query: User's question in any language
        top_k: Number of chunks to retrieve
        filter_dict: Optional metadata filter
        
    Returns:
        Dictionary with same structure as strategy_a
    """
    # Detect language
    detected_lang = lang_utils.detect_language(user_query)
    if not detected_lang:
        detected_lang = "en"
    
    lang_name = lang_utils.get_language_name(detected_lang)
    
    logger.info(f"Detected language: {lang_name} ({detected_lang})")
    
    # Translate query to English if needed
    if detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
        logger.info("Translating query to English...")
        english_query = translation.translate_to_english(user_query, detected_lang)
        logger.info(f"English query: {english_query}")
    else:
        english_query = user_query
    
    # Retrieve context using English query
    logger.info("Retrieving relevant context...")
    context_data = retrieve_context(english_query, top_k, filter_dict)
    
    # Handle empty collection
    if context_data['chunks_used'] == 0:
        logger.warning("No documents available for answering question")
        
        no_docs_msg = config.NO_DOCUMENTS_RESPONSE
        if detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
             try:
                 # Translate the no documents response
                 no_docs_msg = translation.translate_from_english(no_docs_msg, detected_lang)
             except Exception:
                 pass # Fallback to English
             
        return {
            'answer': no_docs_msg,
            'language': detected_lang,
            'language_name': lang_name,
            'chunks_used': 0,
            'citations': []
        }
    
    logger.info(f"Retrieved {len(context_data['chunks'])} chunks, using {context_data['chunks_used']}")
    
    # Build prompt for English answer
    prompt = build_prompt(
        user_query=english_query,
        context=context_data['formatted_context'],
        target_lang="en",
        strategy="B"
    )
    
    # Generate answer in English
    logger.info("Generating answer in English...")
    english_answer = llm_generate(prompt, model=model, provider=provider)
    
    # Extract citations from ENGLISH answer (before translation) using robust parser.
    # Compacting here means the translated answer inherits the dense numbering.
    english_answer, citations = compact_citations(
        english_answer, context_data['metadatas'], context_data.get('chunks'),
        visible_chunks=context_data.get('chunks_used'))
    
    # Translate answer to target language if needed
    if detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
        logger.info(f"Translating answer to {lang_name}...")
        answer = translation.translate_from_english(english_answer, detected_lang)
    else:
        answer = english_answer
    
    result = {
        'answer': answer,
        'language': detected_lang,
        'language_name': lang_name,
        'chunks_used': context_data['chunks_used'],
        'citations': citations,
        'english_answer': english_answer,
        'degraded': context_data.get('degraded'),
    }
    faith_result = _run_faithfulness(
        english_answer, context_data.get('chunks', []), context_data.get('metadatas', []))
    result['faithfulness'] = faith_result["claims"]
    result['answer_confidence'] = faith_result["confidence"]
    return result


def answer_question(
    user_query: str,
    strategy: str = "A",
    top_k: int = None,
    filter_dict: Optional[Dict] = None,
    model: str = None,
    provider: str = None,
) -> Dict[str, Any]:
    """
    Main entry point: Answer a user's question in their language.
    
    Args:
        user_query: User's question in any supported language
        strategy: "A" for multilingual LLM, "B" for English + translation
        top_k: Number of chunks to retrieve
        filter_dict: Optional metadata filter (e.g., {"year": 2023})
        
    Returns:
        Dictionary with answer, language info, and citations
    """
    if strategy == "A":
        return answer_question_strategy_a(user_query, top_k, filter_dict, model=model, provider=provider)
    elif strategy == "B":
        return answer_question_strategy_b(user_query, top_k, filter_dict, model=model, provider=provider)
    else:
        raise ValueError(f"Invalid strategy: {strategy}. Must be 'A' or 'B'")


def answer_with_history(
    messages: List[Dict[str, str]],
    strategy: str = "A",
    top_k: int = None,
    filter_dict: Optional[Dict] = None,
    model: str = None,
    provider: str = None,
) -> Dict[str, Any]:
    """
    Answer the latest user message while incorporating conversation history.

    Args:
        messages: Full conversation so far as a list of
                  ``{"role": "user"|"assistant", "content": str}`` dicts.
                  The last element must have ``role == "user"``.
        strategy: "A" for multilingual LLM, "B" for English + translation.
        top_k: Number of chunks to retrieve.
        filter_dict: Optional ChromaDB metadata filter.

    Returns:
        Same shape as ``answer_question()``.
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Last message must be from the user")

    user_query = messages[-1]["content"]
    prior = messages[:-1]

    detected_lang = lang_utils.detect_language(user_query) or "en"
    lang_name = lang_utils.get_language_name(detected_lang)

    # For strategy B, retrieve using an English translation of the query
    if strategy == "B" and detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
        try:
            retrieval_query = translation.translate_to_english(user_query, detected_lang)
        except Exception:
            retrieval_query = user_query
    else:
        retrieval_query = user_query

    context_data = retrieve_context(retrieval_query, top_k, filter_dict)

    if context_data["chunks_used"] == 0:
        no_docs_msg = config.NO_DOCUMENTS_RESPONSE
        if detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
            try:
                no_docs_msg = translation.translate_from_english(no_docs_msg, detected_lang)
            except Exception:
                pass
        return {
            "answer": no_docs_msg,
            "language": detected_lang,
            "language_name": lang_name,
            "chunks_used": 0,
            "citations": [],
        }

    # Trim history to the last CHAT_HISTORY_MAX_TURNS exchanges (user + assistant each)
    max_msgs = config.CHAT_HISTORY_MAX_TURNS * 2
    trimmed = prior[-max_msgs:] if len(prior) > max_msgs else prior

    # Build numbered history so the model can track conversational structure
    history_lines = []
    turn = 1
    i = 0
    while i < len(trimmed):
        if trimmed[i]["role"] == "user":
            user_line = f"[Turn {turn}] User: {trimmed[i]['content']}"
            if i + 1 < len(trimmed) and trimmed[i + 1]["role"] == "assistant":
                asst_line = f"[Turn {turn}] Assistant: {trimmed[i + 1]['content']}"
                history_lines.append(f"{user_line}\n{asst_line}")
                i += 2
            else:
                history_lines.append(user_line)
                i += 1
            turn += 1
        else:
            i += 1
    history_str = "\n\n".join(history_lines)

    # Strategy B answers in English, so the question must reach the model in
    # English too — prepare_chat_for_stream already does this, and /chat vs
    # /chat/stream must not build different prompts for the same request.
    prompt_query = retrieval_query if strategy == "B" else user_query
    prompt = build_prompt(
        user_query=prompt_query,
        context=context_data["formatted_context"],
        target_lang=detected_lang,
        strategy=strategy,
    )
    if history_str:
        prompt = f"## Conversation History\n{history_str}\n\n---\n\n{prompt}"

    english_answer = llm_generate(prompt, model=model, provider=provider)
    # Compact before any translation so the translated answer carries the same numbers.
    english_answer, citations = compact_citations(
        english_answer, context_data["metadatas"], context_data.get("chunks"),
        visible_chunks=context_data.get("chunks_used"))

    if strategy == "B" and detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
        try:
            answer = translation.translate_from_english(english_answer, detected_lang)
        except Exception:
            answer = english_answer
    else:
        answer = english_answer

    result: Dict[str, Any] = {
        "answer": answer,
        "language": detected_lang,
        "language_name": lang_name,
        "chunks_used": context_data["chunks_used"],
        "citations": citations,
        "degraded": context_data.get("degraded"),
    }
    if strategy == "B" and answer != english_answer:
        result["english_answer"] = english_answer
    faith_result = _run_faithfulness(
        english_answer, context_data.get("chunks", []), context_data.get("metadatas", []))
    result["faithfulness"] = faith_result["claims"]
    result["answer_confidence"] = faith_result["confidence"]
    return result


if __name__ == "__main__":
    # Test retrieval (without LLM)
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    logger.info("Testing RAG Pipeline (Retrieval Only)")
    logger.info("=" * 60)
    
    test_query = "What is the treatment for diabetes?"
    
    logger.info(f"\nQuery: {test_query}")
    logger.info("\nRetrieving context...")
    
    try:
        context_data = retrieve_context(test_query, top_k=3)
        
        if context_data['chunks_used'] == 0:
            logger.warning("No documents found. Please ingest PDFs first.")
        else:
            logger.info(f"\nRetrieved {len(context_data['chunks'])} chunks, using {context_data['chunks_used']}:")
            logger.info("-" * 60)
            logger.info(context_data['formatted_context'])
        
        logger.info("\n" + "=" * 60)
        logger.info("Retrieval test successful!")
        logger.info("\nTo test full answer generation:")
        logger.info("1. Ensure Gemini API key is configured in .env")
        logger.info("2. Run: python examples/example_query.py")
        logger.info("3. Or start the API server: python start_server.py")
        
    except Exception as e:
        logger.error(f"\nError: {e}")
        logger.info("\nMake sure you have:")
        logger.info("1. Ingested some PDFs (run: python ingest.py)")
        logger.info("2. Installed all dependencies (pip install -r requirements.txt)")
