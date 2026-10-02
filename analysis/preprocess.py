import os
import re
import pandas as pd

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
SPACE_PATTERN = re.compile(r"\s+")
SYMBOL_PATTERN = re.compile(r"[^a-z0-9\s']")

INPUT_FILE = "data/all_reviews.csv"
OUTPUT_FILE = "data/clean_reviews.csv"


def clean_text(text):
    """Make text simple for topic analysis: lowercase, no links, no symbols."""
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = SYMBOL_PATTERN.sub(" ", text)  # removes emojis and punctuation
    return SPACE_PATTERN.sub(" ", text).strip()


def is_mostly_english(text, threshold=0.8):
    """Rough check: are most of the letters plain English letters?"""
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(c.isascii() for c in letters)
    return ascii_letters / len(letters) >= threshold


def preprocess_reviews(df, min_words=2):
    """Clean a raw reviews table and return the cleaned version."""
    df = df.copy()
    print(f"Starting with {len(df)} reviews")

    # 1. Fix types and drop broken rows
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["text", "date", "rating"])
    df = df[df["rating"].between(1, 5)]
    print(f"Removed {before - len(df)} rows with missing or invalid values")

    # 2. Clean the text
    df["text"] = df["text"].astype(str).str.strip()
    df["clean_text"] = df["text"].apply(clean_text)
    df["word_count"] = df["clean_text"].str.split().str.len()
    before = len(df)
    df = df[df["word_count"] >= min_words]
    print(f"Removed {before - len(df)} reviews that were too short")

    # 3. Keep English reviews only
    before = len(df)
    df = df[df["text"].apply(is_mostly_english)]
    print(f"Removed {before - len(df)} non-English reviews")

    # 4. Remove copy-pasted duplicates within each source
    before = len(df)
    df = df.drop_duplicates(subset=["source", "clean_text"])
    print(f"Removed {before - len(df)} duplicate reviews")

    # 5. Add helper columns for trends later
    df["day"] = df["date"].dt.normalize()
    df["week"] = df["date"].dt.to_period("W").dt.start_time

    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    print(f"Finished with {len(df)} clean reviews")
    return df


if __name__ == "__main__":
    raw = pd.read_csv(INPUT_FILE)
    clean = preprocess_reviews(raw)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved to {OUTPUT_FILE}")

    print()
    print("Reviews per source after cleaning:")
    print(clean["source"].value_counts())
    print()
    print("Before and after for 3 sample reviews:")
    for _, row in clean.head(3).iterrows():
        print("ORIGINAL:", row["text"][:100])
        print("CLEANED: ", row["clean_text"][:100])
        print()