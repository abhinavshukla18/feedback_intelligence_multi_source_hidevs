import re
import time
import pandas as pd

from analysis.ai_engine import ask_reviews
from analysis.embeddings import load_model, search_reviews
from analysis.issues import CATEGORIES

REVIEWS_FILE = "data/review_sentiment.csv"

# Test question -> the issue category its top results should mention
RETRIEVAL_TESTS = {
    "app keeps crashing": "Crashes and bugs",
    "cannot log in to my account": "Login and account",
    "too many ads": "Ads",
    "downloads and offline mode not working": "Playback and downloads",
    "subscription is too expensive": "Pricing and subscription",
    "shuffle keeps playing the same songs": "Shuffle and queue",
    "search never finds the artist I want": "Search and recommendations",
    "battery drains and the app is slow": "Slow performance and battery",
}


def star_label(rating):
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def evaluate_sentiment():
    df = pd.read_csv(REVIEWS_FILE)
    df = df[df["source"] != "Survey CSV"].copy()  # survey rows are only sample data
    df["star_label"] = df["rating"].apply(star_label)
    clear = df[df["rating"] != 3]  # 1-2 and 4-5 star reviews have a clear "right" answer

    print("=== Sentiment vs star ratings ===")
    print(f"Reviews checked: {len(df)}")
    print(f"Agreement, all reviews: {(df['sentiment'] == df['star_label']).mean() * 100:.1f}%")
    print(f"Agreement, clear 1-2 / 4-5 star reviews: "
          f"{(clear['sentiment'] == clear['star_label']).mean() * 100:.1f}%")
    opposite = (((clear["star_label"] == "negative") & (clear["sentiment"] == "positive")) |
                ((clear["star_label"] == "positive") & (clear["sentiment"] == "negative")))
    print(f"Opposite-direction disagreements: {opposite.mean() * 100:.1f}%")
    print("Agreement by model confidence (clear reviews only):")
    for name, low, high in [("high (0.8+)", 0.8, 1.01),
                            ("medium (0.6-0.8)", 0.6, 0.8),
                            ("low (below 0.6)", 0.0, 0.6)]:
        part = clear[(clear["confidence"] >= low) & (clear["confidence"] < high)]
        if len(part):
            rate = (part["sentiment"] == part["star_label"]).mean() * 100
            print(f"  {name}: {rate:.1f}% ({len(part)} reviews)")


def evaluate_retrieval(model, k=5):
    print("\n=== Retrieval relevance (top-5 keyword hit rate) ===")
    results = []
    for query, issue in RETRIEVAL_TESTS.items():
        pattern = re.compile(CATEGORIES[issue])
        hits = search_reviews(query, model=model, n_results=k)
        relevant = sum(1 for h in hits if pattern.search(h["text"].lower()))
        results.append(relevant / k)
        print(f"  {relevant}/{k} relevant: {query}")
    print(f"Average precision@{k}: {sum(results) / len(results) * 100:.0f}%")


def evaluate_speed(model):
    print("\n=== Speed ===")
    queries = list(RETRIEVAL_TESTS)[:5]
    start = time.perf_counter()
    for q in queries:
        search_reviews(q, model=model, n_results=5)
    print(f"Average search time: {(time.perf_counter() - start) / len(queries) * 1000:.0f} ms")
    start = time.perf_counter()
    ask_reviews("What do people say about ads?", embed_model=model)
    print(f"One full question (search + AI answer): {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    evaluate_sentiment()
    embed_model = load_model()
    evaluate_retrieval(embed_model)
    evaluate_speed(embed_model)