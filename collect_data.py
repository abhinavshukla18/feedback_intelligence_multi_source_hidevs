import os
import pandas as pd

from fetchers.playstore import fetch_playstore_history
from fetchers.appstore import fetch_appstore_reviews
from fetchers.csv_loader import load_csv_reviews

OUTPUT_FILE = "data/all_reviews.csv"


def collect_all_reviews(playstore_days=42, playstore_per_day=60, appstore_pages=10,
                        csv_path="data/sample_survey.csv"):
    """Fetch from all sources, merge into one clean table."""
    sources = {
                "Google Play": lambda: fetch_playstore_history(days=playstore_days, per_day=playstore_per_day),
        "App Store": lambda: fetch_appstore_reviews(max_pages=appstore_pages),
        "Survey CSV": lambda: load_csv_reviews(csv_path),
    }

    frames = []
    for name, fetch in sources.items():
        try:
            df = fetch()
            print(f"{name}: {len(df)} reviews")
            if not df.empty:
                frames.append(df)
        except Exception as error:
            # One broken source should never stop the others
            print(f"{name} failed: {error}")

    if not frames:
        print("No data collected from any source.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Clean up
    combined = combined.drop_duplicates(subset=["source", "review_id"])
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"] != ""]
    combined = combined.sort_values("date", ascending=False).reset_index(drop=True)
    return combined


def save_reviews(df, path=OUTPUT_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} reviews to {path}")


if __name__ == "__main__":
    all_reviews = collect_all_reviews()
    if not all_reviews.empty:
        save_reviews(all_reviews)
        print()
        print("Reviews per source:")
        print(all_reviews["source"].value_counts())
        print()
        print("Average rating per source:")
        print(all_reviews.groupby("source")["rating"].mean().round(2))