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
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Client is initialised lazily inside llm_generate() so we can still import
# this module even when the API key is absent (e.g. retrieval-only mode).
_genai_client: genai.Client | None = None
_client_lock = __import__('threading').Lock()


def _get_client() -> genai.Client:
    """Return (and lazily create) the shared Google GenAI client (thread-safe)."""
    global _genai_client
    if _genai_client is None:
        with _client_lock:
            if _genai_client is None:
                if not config.LLM_API_KEY:
                    raise ValueError(
                        "Google Gemini API key not configured. "
                        "Please set LLM_API_KEY environment variable or in .env file."
                    )
                _genai_client = genai.Client(api_key=config.LLM_API_KEY)
    return _genai_client


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
    
    citations = []
    seen_nums = set()
    
    # Match [1], [1, 2], [1-3], [1,2,3], etc.
    raw_citations = re.findall(r'\[([\d\s,\-]+)\]', answer)
    
    for raw in raw_citations:
        # Parse comma-separated and ranges
        parts = raw.replace(' ', '').split(',')
        for part in parts:
            if '-' in part:
                try:
                    start, end = part.split('-')
                    s, e = int(start), int(end)
                    if e - s > 50:
                        continue
                    for num in range(s, e + 1):
                        seen_nums.add(num)
                except ValueError:
                    pass
            else:
                # Single number
                try:
                    seen_nums.add(int(part))
                except ValueError:
                    pass
    
    # Build citations list (sorted for consistent ordering)
    for num in sorted(seen_nums):
        idx = num - 1
        if 0 <= idx < len(metadatas):
            citations.append({
                'number': str(num),
                'title': metadatas[idx].get('title', 'Unknown'),
                'section': metadatas[idx].get('section', 'body')
            })
            if chunks and idx < len(chunks):
                logger.debug(f"Citation [{num}] refers to paragraph: {chunks[idx][:200]}...")
    
    return citations


def retrieve_context(
    user_query: str,
    top_k: int = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    collection=None
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
    
    if collection is None:
        collection = vector_store.get_or_create_collection()
    
    # Embed the query
    query_embedding = embeddings.embed_query(user_query)
    
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

    return {
        'chunks': docs,
        'metadatas': metas,
        'distances': dists,
        'formatted_context': formatted_context,
        'chunks_used': chunks_used
    }


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
        context_part = f"[{i}] {title} - {section}:\n{chunk}\n"
        
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


def llm_generate(prompt: str, max_tokens: int = None) -> str:
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

    client = _get_client()

    # Safety settings - set to BLOCK_NONE for scientific content
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE",
        ),
    ]

    generate_config = types.GenerateContentConfig(
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        safety_settings=safety_settings,
        system_instruction=config.SYSTEM_PROMPT,
    )

    try:
        response = _call_gemini(client, config.LLM_MODEL_NAME, prompt, generate_config)

        # Check if response has text
        if response.text:
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
                    return ''.join(parts_text)

            finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
            raise Exception(
                f"Response blocked or incomplete. Finish reason: {finish_reason}. "
                f"This may be due to safety filters or token limits."
            )

        raise Exception("No response generated from Gemini API")

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(Exception), reraise=True)
def _call_gemini(client, model, contents, gen_config):
    return client.models.generate_content(model=model, contents=contents,
                                          config=gen_config)


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
        logger.debug(f"Faithfulness check skipped: {e}")
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


def _build_chat_prompt(
    user_query: str,
    context: str,
    history: str,
    lang_name: str,
) -> str:
    """Build a prompt that includes conversation history before the retrieved context."""
    history_section = f"## Conversation History\n{history}\n\n---\n\n" if history else ""
    return (
        f"{history_section}"
        f"## Context\n{context}\n\n---\n\n"
        f"## Current Question\n{user_query}\n\n"
        f"## Instructions\n"
        f"- Answer in: {lang_name}.\n"
        f"- Use only the context above; cite each claim inline as [n].\n"
        f"- This is a conversation — refer to prior turns only when directly relevant.\n"
        f"- Lead with a direct answer, then add technical depth only as the question requires.\n"
        f"- If the context is insufficient, state exactly what is missing instead of inferring.\n"
    )


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

    # Always show the user's original query in the chat prompt so the model
    # sees what the user actually typed (retrieval_query is only used for
    # vector search in Strategy B — the answer language instruction handles output lang).
    prompt_lang = "English" if strategy == "B" else lang_name
    prompt = _build_chat_prompt(
        user_query=user_query,
        context=context_data["formatted_context"],
        history=history_str,
        lang_name=prompt_lang,
    )

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
