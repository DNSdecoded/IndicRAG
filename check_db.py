import chromadb
import config
import vector_store

def dump_collection():
    collection = vector_store.get_or_create_collection()
    res = collection.get()
    
    ids = res['ids']
    print(f"Total IDs: {len(ids)}")
    print(f"Unique IDs: {len(set(ids))}")
    
    from collections import Counter
    counts = Counter(ids)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    for k, v in duplicates.items():
        print(f"Duplicate {k}: {v} times")
        
dump_collection()
