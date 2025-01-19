import os
import faiss
from sentence_transformers import SentenceTransformer
import json

def create_faiss_index(data, index_path="models/document_index"):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    texts = [item["content"] for item in data]
    embeddings = model.encode(texts)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, index_path)
    print(f"Index saved at {index_path}")

def query_faiss_index(query, data, index_path="models/document_index", top_k=5):
    index = faiss.read_index(index_path)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    query_embedding = model.encode([query])
    
    distances, indices = index.search(query_embedding, top_k)
    return [data[idx] for idx in indices[0]]

if __name__ == "__main__":
    
    with open("data/wiki_chunks.json", "r", encoding="utf-8") as f:
        preprocessed_data = json.load(f)
    
    create_faiss_index(preprocessed_data)

    results = query_faiss_index("What is Machine Learning?", preprocessed_data)
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['content']}")
