from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import uuid


KB_DIR = Path("../knowledge_base")
CHROMA_DIR = Path("../chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "],
)

docs = []
ids = []

for file in tqdm(KB_DIR.glob("*.txt")):
    text = file.read_text(encoding="utf-8")
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        docs.append(Document(page_content=chunk, metadata={"source": file.name, "chunk_id": i}))
        ids.append(str(uuid.uuid4()))

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    ids=ids, 
    persist_directory=str(CHROMA_DIR)
)

query = "What moves have Luycdd?"
results = db.similarity_search(query, k=3)
for r in results:
    print(f"{r.metadata['source']} → {r.page_content[:120]}...")
