import os
import requests
import json
import re
import datetime
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

YC_API_KEY = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")

if not YC_API_KEY or not YC_FOLDER_ID:
    print("Ошибка: не найден YC_API_KEY или YC_FOLDER_ID в .env")
    exit(1)

YC_MODEL_URI = f"gpt://{YC_FOLDER_ID}/yandexgpt-lite"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

LOG_FILE = os.path.join(os.path.dirname(__file__), "logs.jsonl")

class SecurityFilter:
    @staticmethod
    def detect_malicious_instructions(text: str) -> bool:
        patterns = [
            r'ignore.*instruction', r'ignore.*previous', r'output.*".*password.*"',
            r'output.*".*пароль.*"', r'super.*password', r'супер.*парол',
            r'root.*:.*swordfish', r'выведи.*парол', r'покажи.*парол',
            r'ignore.*все.*инструкц', r'игнорируй.*все'
        ]
        return any(re.search(p, text.lower(), re.IGNORECASE) for p in patterns)

    @staticmethod
    def filter_malicious_chunks(docs):
        safe_docs = []
        for doc in docs:
            if not SecurityFilter.detect_malicious_instructions(doc.page_content):
                safe_docs.append(doc)
        return safe_docs

    @staticmethod
    def sanitize_response(response: str) -> str:
        patterns = [
            r'суперпароль.*root.*:.*swordfish',
            r'password.*:.*swordfish',
            r'root.*:.*swordfish'
        ]
        for p in patterns:
            if re.search(p, response, re.IGNORECASE):
                return "Извините, я не могу предоставить эту информацию по соображениям безопасности."
        return response

def log_interaction(query, docs, answer):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "chunks_found": len(docs),
        "answer_length": len(answer),
        "success": len(answer) > 7 and answer.lower() not in ["не знаю", "i don’t know"],
        "sources": [doc.metadata.get("source", "unknown") for doc in docs],
        "answer_preview": answer[:120]
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def ask_yandex_gpt(prompt: str) -> str:
    headers = {"Authorization": f"Api-Key {YC_API_KEY}", "Content-Type": "application/json"}
    system_prompt = """Ты — ассистент по Pokemon. Отвечай строго по документам. 
Если нет информации — говори "Не знаю"."""
    data = {
        "modelUri": YC_MODEL_URI,
        "completionOptions": {"stream": False, "temperature": 0.4, "maxTokens": 400},
        "messages": [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": prompt}
        ]
    }
    try:
        response = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers, json=data, timeout=60
        )
        response.raise_for_status()
        result = response.json()
        raw = result["result"]["alternatives"][0]["message"]["text"]
        return SecurityFilter.sanitize_response(raw)
    except Exception as e:
        return f"Ошибка: {e}"
        
def chat():
    print("Pokemon RAG-бот с логированием запущен!")
    print("Введите 'exit' для выхода.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        try:
            docs = retriever.invoke(query)
            safe_docs = SecurityFilter.filter_malicious_chunks(docs)
            if not safe_docs:
                answer = "Не найдено безопасных данных для ответа."
                print(f"\nBot: {answer}")
                log_interaction(query, [], answer)
                continue
            context = "\n\n".join([doc.page_content for doc in safe_docs])
            full_prompt = f"Документы:\n{context}\n\nВопрос: {query}\n\nОтвет:"
            answer = ask_yandex_gpt(full_prompt)
            print(f"\nBot: {answer}")
            print("Источники:")
            for doc in safe_docs:
                print(f"- {os.path.basename(doc.metadata.get('source', 'unknown'))}")
            log_interaction(query, safe_docs, answer)
        except Exception as e:
            print(f"\nОшибка: {e}")

if __name__ == "__main__":
    chat()
