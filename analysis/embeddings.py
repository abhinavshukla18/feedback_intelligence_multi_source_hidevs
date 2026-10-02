import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "data/review_chunks.csv"
DB_PATH = "data/chroma_db"
COLLECTION_NAME = "reviews"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model():
    return SentenceTransformer(MODEL_NAME)


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client, client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # compare by cosine similarity
    )


def build_knowledge_base(batch_size=500):
    """Embed every chunk and store it in the Chroma database."""
    chunks = pd.read_csv(CHUNKS_FILE).dropna(subset=["chunk_text"])
    print(f"Loaded {len(chunks)} chunks")

    model = load_model()
    print("Creating embeddings (the first run downloads the model)...")
    embeddings = model.encode(
        chunks["chunk_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Start fresh so re-running never creates duplicates
    client, _ = get_collection()
    client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    for start in range(0, len(chunks), batch_size):
        part = chunks.iloc[start:start + batch_size]
        collection.add(
            ids=part["chunk_id"].astype(str).tolist(),
            embeddings=embeddings[start:start + batch_size].tolist(),
            documents=part["chunk_text"].tolist(),
            metadatas=[
                {
                    "review_id": str(r.review_id),
                    "source": str(r.source),
                    "rating": float(r.rating),
                    "date": str(r.date),
                    "week": str(r.week),
                }
                for r in part.itertuples()
            ],
        )
    print(f"Stored {collection.count()} chunks in {DB_PATH}")


def search_reviews(query, model=None, n_results=5, source=None):
    """Find the chunks whose meaning is closest to the query."""
    model = model or load_model()
    _, collection = get_collection()
    query_vector = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results,
        where={"source": source} if source else None,
    )

    hits = []
    for text, meta, dist in zip(results["documents"][0],
                                results["metadatas"][0],
                                results["distances"][0]):
        hits.append({
            "text": text,
            "source": meta["source"],
            "rating": meta["rating"],
            "similarity": round(1 - dist, 3),  # cosine distance -> similarity
        })
    return hits


if __name__ == "__main__":
    build_knowledge_base()

    model = load_model()
    for query in ["app keeps crashing", "cannot log in to my account",
                  "too many ads"]:
        print(f"\nSearch: '{query}'")
        for hit in search_reviews(query, model=model, n_results=3):
            print(f"  ({hit['similarity']}) [{hit['source']}, "
                  f"{hit['rating']:.0f} stars] {hit['text'][:110]}")