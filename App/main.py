from fastapi import FastAPI
from generation import generate_response
import json
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

with open("data/wiki_chunks.json", "r", encoding="utf-8") as f:
    preprocessed_data = json.load(f)

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Chatbot!"}

@app.get("/ask/")
def ask_question(query: str):
    response = generate_response(query, preprocessed_data)
    return {"query": query, "response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
