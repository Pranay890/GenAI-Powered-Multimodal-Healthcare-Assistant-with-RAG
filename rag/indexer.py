import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import pickle


def build_index(pdf_path="data/medical_guidelines.pdf"):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages])
    chunks = text.split("\n\n")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open("rag/vectorstore.pkl", "wb") as f:
        pickle.dump({"index": index, "texts": chunks}, f)

    print("Vectorstore built successfully!")


build_index()
