import os
import pandas as pd
from transformers import pipeline

CHUNKS_FILE = "data/review_chunks.csv"
REVIEWS_FILE = "data/clean_reviews.csv"
CHUNK_OUTPUT = "data/chunk_sentiment.csv"
REVIEW_OUTPUT = "data/review_sentiment.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def load_classifier():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,          # return the probability of all 3 labels
        truncation=True,
        max_length=256,
    )


def score_texts(texts, classifier, batch_size=32):
    """Return sentiment label, confidence and the 3 probabilities per text."""
    results = classifier(list(texts), batch_size=batch_size)
    rows = []
    for scores in results:
        probs = {s["label"].lower(): s["score"] for s in scores}
        label = max(probs, key=probs.get)
        rows.append({
            "sentiment": label,
            "confidence": round(probs[label], 4),
            "p_negative": round(probs.get("negative", 0), 4),
            "p_neutral": round(probs.get("neutral", 0), 4),
            "p_positive": round(probs.get("positive", 0), 4),
        })
    return pd.DataFrame(rows)


def label_from_probs(df):
    """Pick the label and confidence from averaged probabilities."""
    probs = df[["p_negative", "p_neutral", "p_positive"]]
    df["sentiment"] = probs.idxmax(axis=1).str.replace("p_", "", regex=False)
    df["confidence"] = probs.max(axis=1).round(4)
    return df


def analyze_sentiment(chunks, reviews, classifier=None):
    """Score every chunk, then combine into one score per review."""
    classifier = classifier or load_classifier()

    print(f"Scoring {len(chunks)} chunks (this can take a few minutes)...")
    scores = score_texts(chunks["chunk_text"].astype(str), classifier)
    chunk_df = pd.concat([chunks.reset_index(drop=True), scores], axis=1)

    # Average the chunk probabilities back to one row per review
    prob_cols = ["p_negative", "p_neutral", "p_positive"]
    combined = (chunk_df.groupby(["source", "review_id"])[prob_cols]
                .mean().reset_index())
    combined = label_from_probs(combined)

    review_df = reviews.merge(combined, on=["source", "review_id"], how="inner")
    return chunk_df, review_df


if __name__ == "__main__":
    chunks = pd.read_csv(CHUNKS_FILE, dtype={"review_id": str})
    reviews = pd.read_csv(REVIEWS_FILE, dtype={"review_id": str})

    chunk_df, review_df = analyze_sentiment(chunks, reviews)

    os.makedirs("data", exist_ok=True)
    chunk_df.to_csv(CHUNK_OUTPUT, index=False, encoding="utf-8")
    review_df.to_csv(REVIEW_OUTPUT, index=False, encoding="utf-8")
    print(f"Saved {len(chunk_df)} chunk scores and {len(review_df)} review scores")

    print()
    print("Sentiment per source:")
    print(pd.crosstab(review_df["source"], review_df["sentiment"]))
    print()
    print("Star rating vs predicted sentiment (sanity check):")
    print(pd.crosstab(review_df["rating"], review_df["sentiment"]))
    print()
    print(f"Average confidence: {review_df['confidence'].mean():.2f}")
    print(f"Low-confidence reviews (< 0.6): {(review_df['confidence'] < 0.6).sum()}")