import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from analysis.preprocess import clean_text

INPUT_FILE = "data/clean_reviews.csv"
OUTPUT_FILE = "data/review_chunks.csv"

CHUNK_SIZE = 300     # max characters per chunk
CHUNK_OVERLAP = 50   # characters repeated between neighbouring chunks


def build_splitter(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splitter that prefers paragraph breaks, then sentences, then words."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )


def chunk_reviews(df, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Turn a reviews table into a chunks table (one row per chunk)."""
    splitter = build_splitter(chunk_size, overlap)
    rows = []

    for _, row in df.iterrows():
        pieces = splitter.split_text(str(row["text"]))
        for i, piece in enumerate(pieces):
            piece = piece.lstrip(".!? ").strip()  # tidy leftover punctuation
            cleaned = clean_text(piece)
            if not cleaned:
                continue  # skip empty fragments
            rows.append({
                "chunk_id": f"{row['source']}-{row['review_id']}-{i}",
                "review_id": row["review_id"],
                "chunk_index": i,
                "source": row["source"],
                "date": row["date"],
                "week": row["week"],
                "rating": row["rating"],
                "chunk_text": piece,
                "chunk_clean_text": cleaned,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    reviews = pd.read_csv(INPUT_FILE)

    lengths = reviews["text"].astype(str).str.len()
    print(f"Reviews: {len(reviews)}")
    print(f"Average length: {lengths.mean():.0f} characters")
    print(f"Longest review: {lengths.max()} characters")
    print(f"Reviews longer than {CHUNK_SIZE} characters: {(lengths > CHUNK_SIZE).sum()}")

    chunks = chunk_reviews(reviews)
    chunks_per_review = chunks.groupby("review_id").size()
    print()
    print(f"Chunks created: {len(chunks)}")
    print(f"Reviews that got split into 2+ chunks: {(chunks_per_review > 1).sum()}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    chunks.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved to {OUTPUT_FILE}")

    # Show the longest review and how it was split
    longest_id = reviews.loc[lengths.idxmax(), "review_id"]
    print()
    print("Longest review, split into chunks:")
    for _, c in chunks[chunks["review_id"] == longest_id].iterrows():
        print(f"  [{c['chunk_index']}] {c['chunk_text']}")