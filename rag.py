"""
Main RAG pipeline: retrieval and answer generation.
"""

from typing import Dict, List, Optional, Any
import config
import embeddings
import vector_store
import lang_utils
import translation


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
            - 'chunks': List of retrieved text chunks
            - 'metadatas': List of metadata dicts
            - 'distances': List of similarity distances
            - 'formatted_context': Formatted context string for LLM
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K
    
    if collection is None:
        collection = vector_store.get_or_create_collection()
    
    # Embed the query
    query_embedding = embeddings.embed_query(user_query)
    
    # Search vector store
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        filter_dict=filter_dict,
        collection=collection
    )
    
    # Format context for LLM
    formatted_context = format_context(
        chunks=results['documents'],
        metadatas=results['metadatas']
    )
    
    return {
        'chunks': results['documents'],
        'metadatas': results['metadatas'],
        'distances': results['distances'],
        'formatted_context': formatted_context
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
    # Limit total context length
    context_parts = []
    total_length = 0
    
    for i, (chunk, metadata) in enumerate(zip(chunks, metadatas), 1):
        # Stop if we exceed max context length
        if total_length + len(chunk) > config.MAX_CONTEXT_LENGTH:
            break
        
        # Format: [1] Title - Section: chunk text
        title = metadata.get('title', 'Unknown')
        section = metadata.get('section', 'body')
        
        context_part = f"[{i}] {title} - {section}:\n{chunk}\n"
        context_parts.append(context_part)
        total_length += len(context_part)
    
    return "\n".join(context_parts)


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
    # Get language name
    lang_name = lang_utils.get_language_name(target_lang)
    
    if strategy == "A":
        # Strategy A: Ask LLM to answer directly in target language
        prompt = f"""{config.SYSTEM_PROMPT}

{config.QUERY_PROMPT_TEMPLATE.format(
    context=context,
    question=user_query,
    language=lang_name
)}

Remember to:
1. Answer ONLY based on the provided context
2. Use {lang_name} language for your response
3. Use simple, clear language suitable for a general audience
4. Include citations [1], [2], etc. when referencing specific papers
5. If the context is insufficient, clearly state this in {lang_name}
"""
    
    else:  # Strategy B
        # Strategy B: Ask LLM to answer in English (will translate later)
        prompt = f"""{config.SYSTEM_PROMPT}

Context from scientific papers:
{context}

Question: {user_query}

Please answer the question in English using only the information from the context above.
Use clear, simple language suitable for a general audience.
Include citations [1], [2], etc. when referencing specific papers.
"""
    
    return prompt


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
    import google.generativeai as genai
    
    if max_tokens is None:
        max_tokens = config.LLM_MAX_TOKENS
    
    # Check if API key is configured
    if not config.LLM_API_KEY:
        raise ValueError(
            "Google Gemini API key not configured. "
            "Please set LLM_API_KEY environment variable or in .env file."
        )
    
    # Configure Gemini
    genai.configure(api_key=config.LLM_API_KEY)
    
    # Safety settings - set to BLOCK_NONE for scientific content
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]
    
    # Create model with generation config
    generation_config = {
        "temperature": config.LLM_TEMPERATURE,
        "max_output_tokens": max_tokens,
    }
    
    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    try:
        # Generate response
        response = model.generate_content(prompt)
        
        # Check if response has text
        if hasattr(response, 'text') and response.text:
            return response.text
        
        # Handle blocked or empty responses
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # Check finish reason
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                
                # Try to get partial text if available
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts_text = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            parts_text.append(part.text)
                    
                    if parts_text:
                        return ''.join(parts_text)
                
                # If no text, raise error with finish reason
                raise Exception(
                    f"Response blocked or incomplete. Finish reason: {finish_reason}. "
                    f"This may be due to safety filters or token limits."
                )
        
        # If we get here, no valid response
        raise Exception("No response generated from Gemini API")
    
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise


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
    
    print(f"Detected language: {lang_name} ({detected_lang})")
    
    # Retrieve context
    print("Retrieving relevant context...")
    context_data = retrieve_context(user_query, top_k, filter_dict)
    
    print(f"Retrieved {len(context_data['chunks'])} chunks")
    
    # Build prompt
    prompt = build_prompt(
        user_query=user_query,
        context=context_data['formatted_context'],
        target_lang=detected_lang,
        strategy="A"
    )
    
    # Generate answer
    print("Generating answer...")
    answer = llm_generate(prompt)
    
    # Extract citations (simple approach: find [1], [2], etc.)
    import re
    citations = []
    citation_nums = re.findall(r'\[(\d+)\]', answer)
    for num in set(citation_nums):
        idx = int(num) - 1
        if idx < len(context_data['metadatas']):
            citations.append({
                'number': num,
                'title': context_data['metadatas'][idx].get('title', 'Unknown'),
                'section': context_data['metadatas'][idx].get('section', 'body')
            })
    
    return {
        'answer': answer,
        'language': detected_lang,
        'language_name': lang_name,
        'chunks_used': len(context_data['chunks']),
        'citations': citations
    }


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
    
    print(f"Detected language: {lang_name} ({detected_lang})")
    
    # Translate query to English if needed
    if detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
        print("Translating query to English...")
        english_query = translation.translate_to_english(user_query, detected_lang)
        print(f"English query: {english_query}")
    else:
        english_query = user_query
    
    # Retrieve context using English query
    print("Retrieving relevant context...")
    context_data = retrieve_context(english_query, top_k, filter_dict)
    
    print(f"Retrieved {len(context_data['chunks'])} chunks")
    
    # Build prompt for English answer
    prompt = build_prompt(
        user_query=english_query,
        context=context_data['formatted_context'],
        target_lang="en",
        strategy="B"
    )
    
    # Generate answer in English
    print("Generating answer in English...")
    english_answer = llm_generate(prompt)
    
    # Translate answer to target language if needed
    if detected_lang != "en" and lang_utils.is_indic_language(detected_lang):
        print(f"Translating answer to {lang_name}...")
        answer = translation.translate_from_english(english_answer, detected_lang)
    else:
        answer = english_answer
    
    # Extract citations
    import re
    citations = []
    citation_nums = re.findall(r'\[(\d+)\]', answer)
    for num in set(citation_nums):
        idx = int(num) - 1
        if idx < len(context_data['metadatas']):
            citations.append({
                'number': num,
                'title': context_data['metadatas'][idx].get('title', 'Unknown'),
                'section': context_data['metadatas'][idx].get('section', 'body')
            })
    
    return {
        'answer': answer,
        'language': detected_lang,
        'language_name': lang_name,
        'chunks_used': len(context_data['chunks']),
        'citations': citations,
        'english_answer': english_answer  # Include for debugging
    }


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


if __name__ == "__main__":
    # Test retrieval (without LLM)
    print("Testing RAG Pipeline (Retrieval Only)")
    print("=" * 60)
    
    test_query = "What is the treatment for diabetes?"
    
    print(f"\nQuery: {test_query}")
    print("\nRetrieving context...")
    
    try:
        context_data = retrieve_context(test_query, top_k=3)
        
        print(f"\nRetrieved {len(context_data['chunks'])} chunks:")
        print("-" * 60)
        print(context_data['formatted_context'])
        
        print("\n" + "=" * 60)
        print("Retrieval test successful!")
        print("\nTo test full answer generation:")
        print("1. Implement llm_generate() function")
        print("2. Run: python example_query.py")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Ingested some PDFs (run: python ingest.py)")
        print("2. Installed all dependencies (pip install -r requirements.txt)")
