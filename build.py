import pandas as pd
import numpy as np
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
CSV_PATH = "/kaggle/input/vimedical-disease/ViMedical_Disease.csv"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

EMBEDDINGS_OUT = "/kaggle/working/embeddings.npy"
DOCS_OUT = "/kaggle/working/documents.pkl"
FAISS_OUT = "/kaggle/working/faiss.index"


# =========================
# Load CSV
# =========================
def load_csv(path):
    return pd.read_csv(path).dropna()


# =========================
# Build documents (1 disease = 1 doc)
# =========================
def build_documents(df):
    documents = []
    grouped = df.groupby("Disease")["Question"].apply(list)

    for disease, questions in grouped.items():
        symptom_text = "\n".join(f"- {q.strip()}" for q in questions)

        documents.append({
            "text": f"Triệu chứng:\n{symptom_text}",
            "metadata": {
                "disease": disease,
                "num_samples": len(questions)
            }
        })

    return documents


# =========================
# Load embedding model (GPU)
# =========================
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Embedding device: {device}")
    return SentenceTransformer(EMBED_MODEL, device=device)


# =========================
# Embed
# =========================
def embed_documents(model, documents):
    texts = [d["text"] for d in documents]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings.astype("float32")


# =========================
# Build FAISS
# =========================
def build_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("🔹 Loading CSV...")
    df = load_csv(CSV_PATH)

    print("🔹 Building documents...")
    documents = build_documents(df)
    print(f"Diseases: {len(documents)}")

    print("🔹 Loading model...")
    model = load_model()

    print("🔹 Embedding...")
    embeddings = embed_documents(model, documents)

    print("🔹 Building FAISS...")
    index = build_faiss(embeddings)

    print("🔹 Saving artifacts...")
    np.save(EMBEDDINGS_OUT, embeddings)

    with open(DOCS_OUT, "wb") as f:
        pickle.dump(documents, f)

    faiss.write_index(index, FAISS_OUT)

    print("✅ BUILD DONE")
