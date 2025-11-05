import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import requests
import re

load_dotenv()

YC_API_KEY = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")

YC_MODEL_URI = f"gpt://{YC_FOLDER_ID}/yandexgpt-lite"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

def ask_yandex_gpt(prompt: str) -> str:
    headers = {
        "Authorization": f"Api-Key {YC_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "modelUri": YC_MODEL_URI,
        "completionOptions": {"stream": False, "temperature": 0.3, "maxTokens": 400},
        "messages": [{"role": "user", "text": prompt}],
    }

    try:
        response = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["result"]["alternatives"][0]["message"]["text"].strip()
    except Exception as e:
        return f"Ошибка запроса: {e}"

def evaluate_answer(answer: str, expected: str) -> bool:
    """Проверяет, содержит ли ответ ожидаемую строку (грубая оценка)."""
    if expected.lower() == "не знаю":
        return "не знаю" in answer.lower()
    return expected.lower() in answer.lower()

def run_tests(golden_file="golden_questions.json", log_file="logs.jsonl"):
    # Загружаем golden set
    with open(golden_file, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    with open(log_file, "a", encoding="utf-8") as log:
        for item in golden_data:
            question = item["question"]
            expected = item["expected"]

            print(f"\n🔹 Testing: {question}")
            start_time = time.time()

            docs = retriever.invoke(question)
            chunks_found = len(docs) > 0
            sources = [os.path.basename(d.metadata.get("source", "unknown")) for d in docs]

            context = "\n\n".join([d.page_content for d in docs])
            full_prompt = f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            answer = ask_yandex_gpt(full_prompt)

            duration = round(time.time() - start_time, 2)
            is_success = evaluate_answer(answer, expected)
            answer_len = len(answer)

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "expected": expected,
                "answer": answer,
                "answer_length": answer_len,
                "success": is_success,
                "chunks_found": chunks_found,
                "sources": sources,
                "duration_sec": duration,
            }

            log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            log.flush()

            print(f"Answer: {answer[:80]}...")
            print(f"Chunks found: {chunks_found} | Success: {is_success}")

    print("\nTesting completed. Logs saved to", log_file)


if __name__ == "__main__":
    run_tests()
