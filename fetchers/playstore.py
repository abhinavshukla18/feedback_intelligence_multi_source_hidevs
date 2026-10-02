import time
from datetime import datetime, timedelta
import pandas as pd
from google_play_scraper import reviews, Sort

SPOTIFY_APP_ID = "com.spotify.music"
COLUMNS = ["source", "review_id", "date", "rating",
           "text", "author", "app_version"]


def _to_dataframe(raw_reviews):
    """Convert raw Play Store reviews to our common format (the schema)."""
    rows = []
    for r in raw_reviews:
        rows.append({
            "source": "Google Play",
            "review_id": r["reviewId"],
            "date": r["at"],
            "rating": r["score"],
            "text": r["content"],
            "author": r["userName"],
            "app_version": r["appVersion"],
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_playstore_reviews(app_id=SPOTIFY_APP_ID, count=500,
                            lang="en", country="us", retries=3):
    """Fetch the newest reviews (a quick snapshot)."""
    raw_reviews = []
    for attempt in range(1, retries + 1):
        try:
            raw_reviews, _ = reviews(app_id, lang=lang, country=country,
                                     sort=Sort.NEWEST, count=count)
            break
        except Exception as error:
            print(f"Attempt {attempt} failed: {error}")
            if attempt < retries:
                time.sleep(2 * attempt)
            else:
                print("Play Store fetch failed. Returning empty result.")
    return _to_dataframe(raw_reviews)


def fetch_playstore_history(app_id=SPOTIFY_APP_ID, days=42, per_day=60,
                            lang="en", country="us", retries=3,
                            max_requests=400):
    """Page backwards through old reviews, keeping up to `per_day` per day."""
    cutoff = datetime.now() - timedelta(days=days)
    kept = {}      # date -> list of reviews kept for that day
    token = None   # tells Google Play where to continue from

    for request_number in range(max_requests):
        batch = None
        for attempt in range(1, retries + 1):
            try:
                batch, token = reviews(app_id, lang=lang, country=country,
                                       sort=Sort.NEWEST, count=200,
                                       continuation_token=token)
                break
            except Exception as error:
                print(f"Request {request_number + 1}, attempt {attempt} failed: {error}")
                time.sleep(2 * attempt)

        if not batch:
            print("Stopping: no more data (or repeated failures).")
            break

        for r in batch:
            day_list = kept.setdefault(r["at"].date(), [])
            if len(day_list) < per_day:
                day_list.append(r)

        oldest = batch[-1]["at"]
        if request_number % 10 == 0:
            print(f"  ...reached {oldest.date()}")
        if oldest < cutoff or token is None:
            break
        time.sleep(0.5)  # be polite to Google's servers

    raw = [r for day_list in kept.values() for r in day_list if r["at"] >= cutoff]
    return _to_dataframe(raw)


if __name__ == "__main__":
    df = fetch_playstore_history(days=14, per_day=20)
    print(f"Fetched {len(df)} reviews")
    print(df["date"].min(), "to", df["date"].max())