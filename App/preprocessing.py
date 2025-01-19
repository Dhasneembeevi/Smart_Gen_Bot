import json
import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

def fetch_wikipedia_pages(topics, language="en"):
    user_agent = "SmartGenChatbot/1.0 (https://github.com/Dhasneembeevi/Smart_Gen_Bot; dhasmohamed1020@gmail.com)"
    base_url = f"https://{language}.wikipedia.org/w/api.php"
    documents = []

    for topic in topics:
        params = {
            "action": "query",
            "format": "json",
            "titles": topic,
            "prop": "extracts|info",
            "exintro": True,
            "inprop": "url"
        }
        
        response = requests.get(base_url, params=params, headers={'User-Agent': user_agent})
        data = response.json()

        pages = data.get('query', {}).get('pages', {})
        page_data = next(iter(pages.values()), {})

        if page_data:
            print(f"Fetching page: {topic}")
            documents.append({
                "title": topic,
                "summary": page_data.get('extract', ''),
                "content": page_data.get('extract', '')
            })
        else:
            print(f"Page '{topic}' does not exist.")
    
    return documents

def preprocess_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        for chunk in text_splitter.split_text(doc["content"]):
            chunks.append({
                "title": doc["title"],
                "content": chunk
            })
    return chunks

if __name__ == "__main__":
    topics = ["Machine Learning", "Python (programming language)", "Docker"]
    wiki_documents = fetch_wikipedia_pages(topics)
    
    os.makedirs("data", exist_ok=True)

    with open("data/wiki_documents.json", "w", encoding="utf-8") as f:
        json.dump(wiki_documents, f, ensure_ascii=False, indent=4)

    preprocessed_data = preprocess_documents(wiki_documents)
    
    with open("data/wiki_chunks.json", "w", encoding="utf-8") as f:
        json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)

    print("Preprocessing complete. Data saved to data/wiki_chunks.json")
