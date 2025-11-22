"""
Example: Query the RAG system in multiple languages.

Before running this, you must:
1. Ingest some PDFs (run example_ingest.py)
2. Implement the llm_generate() function in rag.py
"""

import rag


# Example queries in different Indian languages
EXAMPLE_QUERIES = {
    "Hindi": "मधुमेह का इलाज क्या है?",  # What is the treatment for diabetes?
    "Tamil": "நீரிழிவு நோய்க்கான சிகிச்சை என்ன?",  # What is the treatment for diabetes?
    "Telugu": "డయాబెటిస్ చికిత్స ఏమిటి?",  # What is the treatment for diabetes?
    "Marathi": "मधुमेहाचा उपचार काय आहे?",  # What is the treatment for diabetes?
    "Bengali": "ডায়াবেটিসের চিকিৎসা কী?",  # What is the treatment for diabetes?
    "English": "What is the treatment for diabetes?",
}


def print_result(result: dict):
    """Pretty print the result."""
    print("\n" + "=" * 60)
    print(f"Language: {result['language_name']} ({result['language']})")
    print(f"Chunks used: {result['chunks_used']}")
    print("-" * 60)
    print("Answer:")
    print(result['answer'])
    
    if result.get('citations'):
        print("\n" + "-" * 60)
        print("Citations:")
        for citation in result['citations']:
            print(f"  [{citation['number']}] {citation['title']} - {citation['section']}")
    
    if result.get('english_answer'):
        print("\n" + "-" * 60)
        print("English version (for debugging):")
        print(result['english_answer'])
    
    print("=" * 60)


def main():
    """
    Example query script.
    """
    print("=" * 60)
    print("Multilingual RAG System - Query Examples")
    print("=" * 60)
    
    # Check if llm_generate is implemented
    try:
        # This will raise NotImplementedError if not implemented
        test_result = rag.llm_generate("test", max_tokens=1)
    except NotImplementedError:
        print("\n⚠️  LLM not configured!")
        print("\nBefore running queries, you must implement llm_generate() in rag.py")
        print("\nOptions:")
        print("1. Use Ollama (local): Install Ollama and uncomment the Ollama code")
        print("2. Use OpenAI API: Set LLM_API_KEY and uncomment OpenAI code")
        print("3. Use Google Gemini: Set LLM_API_KEY and uncomment Gemini code")
        print("\nSee rag.py:llm_generate() for implementation examples.")
        return
    except Exception as e:
        # Other errors are fine, we're just checking if it's implemented
        pass
    
    print("\nAvailable example queries:")
    for i, (lang, query) in enumerate(EXAMPLE_QUERIES.items(), 1):
        print(f"{i}. {lang}: {query}")
    
    print("\nOptions:")
    print("A. Run all example queries (Strategy A: Multilingual LLM)")
    print("B. Run all example queries (Strategy B: English + Translation)")
    print("C. Enter custom query")
    print("Q. Quit")
    
    choice = input("\nEnter choice: ").strip().upper()
    
    if choice == "A":
        print("\n🚀 Running all queries with Strategy A...")
        for lang, query in EXAMPLE_QUERIES.items():
            print(f"\n{'='*60}")
            print(f"Query ({lang}): {query}")
            try:
                result = rag.answer_question(query, strategy="A")
                print_result(result)
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif choice == "B":
        print("\n🚀 Running all queries with Strategy B...")
        for lang, query in EXAMPLE_QUERIES.items():
            print(f"\n{'='*60}")
            print(f"Query ({lang}): {query}")
            try:
                result = rag.answer_question(query, strategy="B")
                print_result(result)
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif choice == "C":
        print("\nEnter your question in any language:")
        custom_query = input("> ").strip()
        
        if not custom_query:
            print("No query entered.")
            return
        
        print("\nChoose strategy:")
        print("A. Multilingual LLM (direct)")
        print("B. English reasoning + Translation")
        strategy_choice = input("Strategy (A/B): ").strip().upper()
        
        if strategy_choice not in ["A", "B"]:
            print("Invalid strategy.")
            return
        
        print(f"\n🚀 Processing query with Strategy {strategy_choice}...")
        try:
            result = rag.answer_question(custom_query, strategy=strategy_choice)
            print_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == "Q":
        print("Goodbye!")
    
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
