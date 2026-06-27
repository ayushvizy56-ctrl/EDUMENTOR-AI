import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config.settings import EMBED_DIR

# Using a small but accurate embedding model
# Downloads automatically on first use (~90MB)
_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading embedding model...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model ready")
    return _embedder


def build_vector_store(chunks: list[dict], store_name: str) -> dict:
    """
    Converts text chunks into embeddings and stores them in a FAISS index.

    FAISS lets us do lightning-fast similarity search — given a question,
    find the most relevant chunks in milliseconds even with thousands of docs.

    Args:
        chunks: list of dicts from pdf_ingestion.py
        store_name: name to save the index under (e.g. "physics_notes")

    Returns:
        dict with index and metadata
    """
    embedder = _get_embedder()
    texts    = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build flat index — exact search, best for < 10k docs
    dimension = embeddings.shape[1]
    index     = faiss.IndexFlatIP(dimension)   # Inner Product = cosine after normalisation
    index.add(embeddings)

    # Save index and metadata to disk
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    index_path = EMBED_DIR / f"{store_name}.index"
    meta_path  = EMBED_DIR / f"{store_name}.meta"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Vector store saved — {index.ntotal} vectors at {index_path}")
    return {"index": index, "chunks": chunks, "name": store_name}


def load_vector_store(store_name: str) -> dict:
    """Load a previously built vector store from disk."""
    index_path = EMBED_DIR / f"{store_name}.index"
    meta_path  = EMBED_DIR / f"{store_name}.meta"

    if not index_path.exists():
        raise FileNotFoundError(f"No vector store found for '{store_name}'")

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded vector store '{store_name}' — {index.ntotal} vectors")
    return {"index": index, "chunks": chunks, "name": store_name}


def search(query: str, store: dict, top_k: int = 4) -> list[dict]:
    """
    Find the top_k most relevant chunks for a given query.

    This is the retrieval part of RAG — before we ask the LLM anything,
    we fetch only the relevant context from the student's documents.
    """
    embedder   = _get_embedder()
    query_vec  = embedder.encode([query], batch_size=1)
    query_vec  = np.array(query_vec, dtype="float32")
    faiss.normalize_L2(query_vec)

    scores, indices = store["index"].search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = store["chunks"][idx].copy()
        chunk["relevance_score"] = round(float(score), 3)
        results.append(chunk)

    return results