"""
Main RAG pipeline: retrieval and answer generation.
"""

from typing import Dict, List, Optional, Any
import logging
import config
import embeddings
import vector_store
import lang_utils
import translation
import llm_client
from google.genai import types

logger = logging.getLogger(__name__)


def extract_citations(answer: str, metadatas: List[Dict], chunks: List[str] = None) -> List[Dict]:
    """
    Extract and parse citations from answer text, handling multiple formats.
    
    Supports:
        - Single citations: [1]
        - Comma-separated: [1, 2] or [1,2,3]
        - Ranges: [1-3]
        - Consecutive: [1][2]
    
    Args:
        answer: Generated answer text containing citations
        metadatas: List of metadata dictionaries from retrieved chunks
        chunks: Optional list of text chunks for logging cited paragraphs
        
    Returns:
        List of citation dictionaries with number, title, and section
    """
    import re
    
    seen_nums = set()

    # Only match the explicit [Cite:N] prefix to avoid over-matching ranges like [10-15] mg
    for m in re.finditer(r'\[Cite:\s*(\d+)\]', answer):
        try:
            seen_nums.add(int(m.group(1)))
        except ValueError:
            pass

    citations = []
    for num in sorted(seen_nums):
        idx = num - 1
        if 0 <= idx < len(metadatas):
            citations.append({
                'number': str(num),
                'title': metadatas[idx].get('title', 'Unknown'),
                'section': metadatas[idx].get('section', 'body')
            })
            if chunks and idx < len(chunks):
                logger.debug(f"Citation [Cite:{num}] refers to paragraph: {chunks[idx][:200]}...")

    return citations


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
    if collection is None and filter_dict is None:
        cached = retrieval_cache.get(cache_key)
        if cached is not None:
            logger.debug("[Retrieval cache hit]")
            return cached

    if collection is None:
        collection = vector_store.get_or_create_collection()

    # Embed the query — HyDE embeds a drafted hypothetical answer instead of
    # the bare question; lexical (BM25) search below always uses the real query.
    query_embedding = _hyde_embedding(user_query) if use_hyde else embeddings.embed_query(user_query)
    
    # Search vector store (dense)
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        filter_dict=filter_dict,
        collection=collection
    )

    # Hybrid: fuse dense results with BM25 lexical search
    if config.USE_HYBRID_SEARCH and results['documents'] and not filter_dict:
        try:
            import bm25_search
            bm25_idx = bm25_search.get_or_build_index(collection)
            if bm25_idx is not None:
                sparse_ids, _ = bm25_idx.search(user_query, top_k=top_k)
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
                    'ids': [i for i in fused_ids if i in id_to_doc][:top_k],
                    'documents': [id_to_doc[i] for i in fused_ids if i in id_to_doc][:top_k],
                    'metadatas': [id_to_meta[i] for i in fused_ids if i in id_to_doc][:top_k],
                    'distances': [id_to_dist.get(i, 1.0) for i in fused_ids if i in id_to_doc][:top_k],
                }
        except Exception as e:
            logger.debug(f"Hybrid search skipped: {e}")

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
        docs, metas, dists = colbert_rerank.rerank(
            user_query, docs, metas, dense_sims,
            top_k=colbert_top_k, weight=config.COLBERT_WEIGHT)

    if config.USE_RERANKER and docs:
        import rerank
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
    if collection is None and filter_dict is None:
        retrieval_cache.put(cache_key, result)
    return result


def format_context(chunks: List[str], metadatas: List[Dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    
    Args:
        chunks: List of text chunks
        metadatas: List of metadata dictionaries
        
    Returns:
        Formatted context string with citations
    """
    context_parts = []
    total_length = 0
    chunks_used = 0
    
    for i, (chunk, metadata) in enumerate(zip(chunks, metadatas), 1):
        # Enforce maximum number of chunks
        if chunks_used >= config.MAX_CONTEXT_CHUNKS:
            break
        
        # Format: [i] Title - Section: chunk text
        title = metadata.get('title', 'Unknown')
        section = metadata.get('section', 'body')
        
        # Build context part FIRST to get accurate length
        context_part = f"[Cite:{i}] {title} - {section}:\n{chunk}\n"
        
        # Check if adding this would exceed length limit
        if total_length + len(context_part) > config.MAX_CONTEXT_LENGTH:
            break
        
        context_parts.append(context_part)
        total_length += len(context_part)
        chunks_used += 1
    
    return "\n".join(context_parts), chunks_used


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
                 system_instruction: str = None) -> str:
    """
    Generate response from LLM using Google Gemini API.
    
    Args:
        prompt: The complete prompt to send to the LLM
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text response
        
    Raises:
        ValueError: If API key is not configured
        Exception: If API call fails
    """
    if max_tokens is None:
        max_tokens = config.LLM_MAX_TOKENS

    from cache import llm_cache, make_key
    cache_key = make_key(prompt, max_tokens, config.LLM_TEMPERATURE)
    cached = llm_cache.get(cache_key)
    if cached is not None:
        logger.debug("[LLM cache hit]")
        return cached

    generate_config = types.GenerateContentConfig(
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        safety_settings=config.SAFETY_SETTINGS,
        system_instruction=system_instruction or config.SYSTEM_PROMPT,
    )

    try:
        response = llm_client.generate_with_failover(config.LLM_MODEL_NAME, prompt, generate_config)

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


def prepare_query_for_stream(user_query: str, strategy: str = "A", top_k: int = None) -> dict:
    """Retrieve context and build prompt for /query/stream.

    Returns dict with keys:
      chunks_used, prompt, metadatas, detected_lang, lang_name
    If no docs: chunks_used=0, no_docs_msg set instead of prompt/metadatas.
    """
    detected_lang = lang_utils.detect_language(user_query) or "en"
    lang_name = lang_utils.get_language_name(detected_lang)

    retrieval_query = user_query
    if strategy == "B" and detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
        try:
            retrieval_query = translation.translate_to_english(user_query, detected_lang)
        except Exception:
            pass

    context_data = retrieve_context(retrieval_query, top_k)

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
            "metadatas": context_data["metadatas"], "detected_lang": detected_lang, "lang_name": lang_name}


def prepare_chat_for_stream(messages: List[Dict[str, str]], strategy: str = "A", top_k: int = None) -> dict:
    """Retrieve context and build prompt for /chat/stream (mirrors answer_with_history).

    Returns same shape as prepare_query_for_stream.
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

    context_data = retrieve_context(retrieval_query, top_k)

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
            "metadatas": context_data["metadatas"], "detected_lang": detected_lang, "lang_name": lang_name}


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


def _run_faithfulness(answer: str, chunks: List[str]) -> List[dict]:
    """Run faithfulness verification if configured; log warnings for ungrounded claims."""
    try:
        import verify
        results = verify.check_claims(answer, chunks)
        for r in results:
            if not r["grounded"]:
                logger.warning(f"Ungrounded claim (score={r['support']:.2f}): {r['claim'][:120]}")
        return results
    except Exception as e:
        logger.warning(f"Faithfulness check failed: {e}", exc_info=True)
        return []


def answer_question_strategy_a(
    user_query: str,
    top_k: int = None,
    filter_dict: Optional[Dict] = None
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
    answer = llm_generate(prompt)
    
    # Extract citations using robust parser
    citations = extract_citations(answer, context_data['metadatas'], context_data.get('chunks'))

    result = {
        'answer': answer,
        'language': detected_lang,
        'language_name': lang_name,
        'chunks_used': context_data['chunks_used'],
        'citations': citations
    }
    result['faithfulness'] = _run_faithfulness(answer, context_data.get('chunks', []))
    return result


def answer_question_strategy_b(
    user_query: str,
    top_k: int = None,
    filter_dict: Optional[Dict] = None
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
    english_answer = llm_generate(prompt)
    
    # Extract citations from ENGLISH answer (before translation) using robust parser
    citations = extract_citations(english_answer, context_data['metadatas'], context_data.get('chunks'))
    
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
        'english_answer': english_answer
    }
    result['faithfulness'] = _run_faithfulness(english_answer, context_data.get('chunks', []))
    return result


def answer_question(
    user_query: str,
    strategy: str = "A",
    top_k: int = None,
    filter_dict: Optional[Dict] = None
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
        return answer_question_strategy_a(user_query, top_k, filter_dict)
    elif strategy == "B":
        return answer_question_strategy_b(user_query, top_k, filter_dict)
    else:
        raise ValueError(f"Invalid strategy: {strategy}. Must be 'A' or 'B'")


def answer_with_history(
    messages: List[Dict[str, str]],
    strategy: str = "A",
    top_k: int = None,
    filter_dict: Optional[Dict] = None,
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

    prompt = build_prompt(
        user_query=user_query,
        context=context_data["formatted_context"],
        target_lang=detected_lang,
        strategy=strategy,
    )
    if history_str:
        prompt = f"## Conversation History\n{history_str}\n\n---\n\n{prompt}"

    english_answer = llm_generate(prompt)
    citations = extract_citations(english_answer, context_data["metadatas"], context_data.get("chunks"))

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
    }
    if strategy == "B" and answer != english_answer:
        result["english_answer"] = english_answer
    result["faithfulness"] = _run_faithfulness(english_answer, context_data.get("chunks", []))
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
