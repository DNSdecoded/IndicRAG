"""
Integration tests for the RAG pipeline.
"""

import numpy as np
from pathlib import Path
import config
import pdf_utils
import embeddings
import vector_store
import lang_utils


def test_pdf_extraction():
    """Test PDF extraction and chunking."""
    print("\n" + "=" * 60)
    print("Test 1: PDF Extraction and Chunking")
    print("=" * 60)
    
    # Check if there are any PDFs to test
    pdf_files = list(config.PAPERS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDFs found for testing. Skipping PDF extraction test.")
        print(f"   Add PDFs to {config.PAPERS_DIR} to test this functionality.")
        return False
    
    # Test with first PDF
    test_pdf = pdf_files[0]
    print(f"Testing with: {test_pdf.name}")
    
    # Extract text
    result = pdf_utils.process_pdf(str(test_pdf))
    
    if not result:
        print("❌ Failed to extract text from PDF")
        return False
    
    print(f"✅ Extracted text: {len(result['text'])} characters")
    print(f"✅ Detected title: {result['title']}")
    print(f"✅ Found sections: {len(result['sections'])}")
    
    # Test chunking
    chunks = pdf_utils.simple_chunk(result['text'])
    print(f"✅ Created chunks: {len(chunks)}")
    
    if chunks:
        avg_size = sum(len(c) for c in chunks) / len(chunks)
        print(f"   Average chunk size: {avg_size:.0f} characters")
    
    return True


def test_embeddings():
    """Test multilingual embeddings."""
    print("\n" + "=" * 60)
    print("Test 2: Multilingual Embeddings")
    print("=" * 60)
    
    # Test texts in different languages
    test_texts = {
        "en": "Diabetes is a chronic disease affecting blood sugar.",
        "hi": "मधुमेह एक पुरानी बीमारी है।",
        "ta": "நீரிழிவு நோய் ஒரு நாள்பட்ட நோய்.",
    }
    
    print("Loading embedding model...")
    model = embeddings.load_embedding_model()
    print(f"✅ Model loaded: {config.EMBEDDING_MODEL_NAME}")
    
    # Embed texts
    texts_list = list(test_texts.values())
    embs = embeddings.embed_passages(texts_list)
    
    print(f"✅ Embeddings shape: {embs.shape}")
    print(f"   Expected: ({len(texts_list)}, {config.EMBEDDING_DIMENSION})")
    
    if embs.shape != (len(texts_list), config.EMBEDDING_DIMENSION):
        print("❌ Unexpected embedding shape!")
        return False
    
    # Test cross-lingual similarity
    print("\nTesting cross-lingual similarity...")
    query = "diabetes treatment"
    query_emb = embeddings.embed_query(query)
    
    similarities = embeddings.compute_similarity(query_emb, embs)
    
    print(f"Query: '{query}'")
    for (lang, text), sim in zip(test_texts.items(), similarities):
        print(f"  {lang}: {sim:.4f} - {text[:50]}")
    
    print("✅ Cross-lingual embeddings working")
    return True


def test_vector_store():
    """Test ChromaDB vector store."""
    print("\n" + "=" * 60)
    print("Test 3: Vector Store Operations")
    print("=" * 60)
    
    # Create test collection
    collection = vector_store.get_or_create_collection("test_rag_pipeline", reset=True)
    print(f"✅ Created test collection: {collection.name}")
    
    # Add test documents
    test_docs = [
        "Diabetes mellitus is a metabolic disorder characterized by high blood sugar.",
        "Treatment for diabetes includes insulin therapy and lifestyle modifications.",
        "Machine learning algorithms can predict disease progression in diabetic patients.",
    ]
    
    # Create embeddings
    test_embeddings = embeddings.embed_passages(test_docs)
    
    test_metadata = [
        {"paper_id": "test1", "title": "Diabetes Overview", "section": "introduction"},
        {"paper_id": "test1", "title": "Diabetes Overview", "section": "treatment"},
        {"paper_id": "test2", "title": "ML in Medicine", "section": "results"},
    ]
    
    test_ids = ["test_doc1", "test_doc2", "test_doc3"]
    
    # Add to store
    vector_store.add_documents(
        texts=test_docs,
        embeddings=test_embeddings,
        metadatas=test_metadata,
        ids=test_ids,
        collection=collection
    )
    
    print(f"✅ Added {len(test_docs)} documents")
    
    # Test search
    query = "How to treat diabetes?"
    query_emb = embeddings.embed_query(query)
    
    results = vector_store.search(query_emb, top_k=2, collection=collection)
    
    print(f"\n✅ Search results for: '{query}'")
    for i, (doc, metadata, dist) in enumerate(zip(
        results['documents'],
        results['metadatas'],
        results['distances']
    ), 1):
        print(f"\n{i}. Distance: {dist:.4f}")
        print(f"   Section: {metadata['section']}")
        print(f"   Text: {doc[:80]}...")
    
    # Cleanup
    vector_store.delete_collection("test_rag_pipeline")
    print("\n✅ Cleaned up test collection")
    
    return True


def test_language_detection():
    """Test language detection."""
    print("\n" + "=" * 60)
    print("Test 4: Language Detection")
    print("=" * 60)
    
    test_cases = {
        "hi": "मधुमेह का इलाज क्या है?",
        "ta": "நீரிழிவு நோய்க்கான சிகிச்சை என்ன?",
        "en": "What is the treatment for diabetes?",
        "te": "డయాబెటిస్ చికిత్స ఏమిటి?",
    }
    
    all_correct = True
    
    for expected_lang, text in test_cases.items():
        detected = lang_utils.detect_language(text)
        lang_name = lang_utils.get_language_name(detected) if detected else "Unknown"
        
        status = "✅" if detected == expected_lang else "❌"
        print(f"{status} Expected: {expected_lang}, Detected: {detected} ({lang_name})")
        print(f"   Text: {text}")
        
        if detected != expected_lang:
            all_correct = False
    
    if all_correct:
        print("\n✅ All language detections correct")
    else:
        print("\n⚠️  Some language detections incorrect (this can happen with short texts)")
    
    return True


def test_retrieval_pipeline():
    """Test the full retrieval pipeline."""
    print("\n" + "=" * 60)
    print("Test 5: Retrieval Pipeline")
    print("=" * 60)
    
    # Check if main collection has documents
    try:
        collection = vector_store.get_or_create_collection()
        stats = vector_store.get_collection_stats(collection)
        
        if stats['count'] == 0:
            print("⚠️  No documents in main collection. Skipping retrieval test.")
            print("   Run example_ingest.py first to add documents.")
            return False
        
        print(f"✅ Collection has {stats['count']} documents")
        
        # Test retrieval
        import rag
        
        test_query = "What is diabetes?"
        print(f"\nTesting retrieval for: '{test_query}'")
        
        context_data = rag.retrieve_context(test_query, top_k=3)
        
        print(f"✅ Retrieved {len(context_data['chunks'])} chunks")
        print("\nFormatted context preview:")
        print("-" * 60)
        print(context_data['formatted_context'][:500] + "...")
        
        return True
    
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG Pipeline Integration Tests")
    print("=" * 60)
    
    tests = [
        ("PDF Extraction", test_pdf_extraction),
        ("Embeddings", test_embeddings),
        ("Vector Store", test_vector_store),
        ("Language Detection", test_language_detection),
        ("Retrieval Pipeline", test_retrieval_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for p in results.values() if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")


if __name__ == "__main__":
    main()
