import os
import json
import hashlib
import logging
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid


# === Пути ===
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR.parent / "knowledge_base"
CHROMA_DIR = BASE_DIR.parent / "chroma_db"
LOG_DIR = BASE_DIR.parent / "logs"
STATE_FILE = BASE_DIR.parent / "update_state.json"

# === Создание директорий ===
LOG_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# === Настройка логов ===
logging.basicConfig(
    filename=LOG_DIR / "update.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# === Инициализация эмбеддингов и text splitter ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "],
)


def calc_hash(path: Path) -> str:
    """Вычислить SHA256 для файла"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_state():
    """Загрузка состояния (хэшей файлов)"""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    """Сохранение состояния"""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main():
    start_time = time.time()
    logging.info("=== Start index update ===")

    # Загружаем состояние
    prev_state = load_state()
    new_state = {}
    new_files = []
    changed_files = []

    # Находим новые и изменённые файлы
    for file in KB_DIR.glob("*.txt"):
        file_hash = calc_hash(file)
        new_state[file.name] = file_hash
        if file.name not in prev_state:
            new_files.append(file)
        elif prev_state[file.name] != file_hash:
            changed_files.append(file)

    # Если изменений нет
    if not new_files and not changed_files:
        logging.info("No new or modified files. Index is up to date.")
        print("No new or modified files. Index is up to date.")
        return

    docs = []
    ids = []

    # Обработка новых и изменённых файлов
    for file in tqdm(new_files + changed_files, desc="Updating index"):
        text = file.read_text(encoding="utf-8")
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": file.name, "chunk_id": i}))
            ids.append(str(uuid.uuid4()))

    # Обновление векторного индекса
    try:
        db = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )

        # Удаляем старые версии изменённых файлов
        for f in changed_files:
            db._collection.delete(where={"source": f.name})

        db.add_documents(documents=docs, ids=ids)
        db.persist()

        # Сохраняем новое состояние
        save_state(new_state)

        duration = round(time.time() - start_time, 2)
        log_msg = (
            f"Index updated successfully. Added {len(new_files)} new files, "
            f"updated {len(changed_files)} changed files, total {len(docs)} chunks. "
            f"Duration: {duration}s"
        )
        logging.info(log_msg)
        print(log_msg)

    except Exception as e:
        logging.error(f"Error updating index: {e}", exc_info=True)
        print(f"Error updating index: {e}")


if __name__ == "__main__":
    main()
