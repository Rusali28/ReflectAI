import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"
DATA_PATH = "/Users/rusalisaha/Documents/Fall 2025/Programming with LLMs/ReflectAI/RAG/knowledge_base"

emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

def initialize_knowledge_base():
    """
    Reads .txt files, chunks them, and builds the Vector Index.
    """
    print("[RAG] Checking Knowledge Base...")
    
    collection = client.get_or_create_collection(
        name="cbt_library", 
        embedding_function=emb_fn
    )
    
    if collection.count() > 0:
        print(f"[RAG] Library loaded ({collection.count()} items ready).")
        return

    print("[RAG] Indexing new data...")
    documents = []
    ids = []
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"[ERROR] '{DATA_PATH}' folder missing.")
        return

    # Loop through all files in the knowledge_base folder
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # CHUNKING: Split by double newlines to keep strategies separate
            chunks = text.split("\n\n")
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append(chunk)
                    ids.append(f"{filename}_{i}")

    if documents:
        collection.add(documents=documents, ids=ids)
        print(f"[RAG] Successfully indexed {len(documents)} strategies.")
    else:
        print("[RAG] No documents found to index.")
def retrieve_context(query, n_results=1):
    """
    The Search Function.
    Input: "I feel anxious"
    Output: "The 5-4-3-2-1 Grounding Technique..."
    """
    collection = client.get_collection(name="cbt_library", embedding_function=emb_fn)
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results['documents'][0]

if __name__ == "__main__":
    initialize_knowledge_base()
    print("\n--- TEST SEARCH: 'I can't sleep' ---")
    print(retrieve_context("I can't sleep"))