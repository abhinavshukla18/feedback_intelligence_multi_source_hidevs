import time
import requests
import pandas as pd

SPOTIFY_APP_ID = "324684580"


def fetch_appstore_reviews(app_id=SPOTIFY_APP_ID, country="us",
                           max_pages=10, retries=3):
    """Fetch recent App Store reviews from Apple's RSS feed."""
    entries = []

    for page in range(1, max_pages + 1):
        url = (f"https://itunes.apple.com/{country}/rss/customerreviews/"
               f"page={page}/id={app_id}/sortby=mostrecent/json")

        page_entries = None
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # raises an error for 404, 429, 500...
                page_entries = response.json().get("feed", {}).get("entry", [])
                break
            except Exception as error:
                print(f"Page {page}, attempt {attempt} failed: {error}")
                if attempt < retries:
                    time.sleep(2 * attempt)

        if not page_entries:
            break  # no more pages, or the fetch kept failing
        entries.extend(page_entries)

    # Convert to our common format (the "schema")
    rows = []
    for e in entries:
        if "im:rating" not in e:
            continue  # the first entry on page 1 describes the app itself, skip it
        title = e["title"]["label"]
        content = e["content"]["label"]
        rows.append({
            "source": "App Store",
            "review_id": e["id"]["label"],
            "date": pd.to_datetime(e["updated"]["label"], utc=True).tz_localize(None),
            "rating": int(e["im:rating"]["label"]),
            "text": f"{title}. {content}",
            "author": e["author"]["name"]["label"],
            "app_version": e.get("im:version", {}).get("label"),
        })

    columns = ["source", "review_id", "date", "rating",
               "text", "author", "app_version"]
    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    df = fetch_appstore_reviews(max_pages=2)
    print(f"Fetched {len(df)} reviews")
    print(df.head())
    