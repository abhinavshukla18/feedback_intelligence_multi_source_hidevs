import pandas as pd

from analysis.issues import OTHER, priority_level, tag_issues
from analysis.preprocess import clean_text, is_mostly_english, preprocess_reviews
from analysis.trends import detect_direction, trend_table
from fetchers.csv_loader import COLUMNS, load_csv_reviews


# ---------- preprocessing ----------
def test_clean_text_removes_links_emojis_and_case():
    assert clean_text("LOVE it!!! 😍 visit https://spotify.com now") == "love it visit now"


def test_clean_text_keeps_apostrophes():
    assert clean_text("It's great") == "it's great"


def test_is_mostly_english():
    assert is_mostly_english("this app keeps crashing")
    assert not is_mostly_english("यह ऐप बहुत अच्छा है")
    assert not is_mostly_english("12345 !!!")


def test_preprocess_drops_bad_rows_short_and_duplicates():
    raw = pd.DataFrame({
        "source": ["App Store"] * 5,
        "review_id": ["1", "2", "3", "4", "5"],
        "date": ["2026-09-01", "2026-09-02", "not a date", "2026-09-04", "2026-09-05"],
        "rating": [5, 1, 3, 9, 1],
        "text": ["Great app love it", "Crashes every time", "Fine app okay",
                 "Rating is invalid here", "Crashes every time"],
    })
    result = preprocess_reviews(raw)
    # row 3 has a broken date, row 4 an invalid rating, row 5 is a duplicate
    assert set(result["review_id"]) == {"1", "2"}
    assert "clean_text" in result.columns
    assert "week" in result.columns


# ---------- CSV loader error handling ----------
def test_csv_loader_missing_file_returns_empty():
    df = load_csv_reviews("data/does_not_exist.csv")
    assert df.empty
    assert list(df.columns) == COLUMNS


def test_csv_loader_missing_columns_returns_empty(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n")
    assert load_csv_reviews(str(path)).empty


def test_csv_loader_reads_valid_file(tmp_path):
    path = tmp_path / "ok.csv"
    path.write_text("date,rating,comment,name\n"
                    "2026-09-20,2,App crashes,Asha\n"
                    "2026-09-21,5,Love it,Ravi\n")
    df = load_csv_reviews(str(path))
    assert len(df) == 2
    assert (df["source"] == "Survey CSV").all()


# ---------- issue tagging and priority ----------
def test_tag_issues_matches_expected_categories():
    texts = ["the app keeps crashing", "cannot log in to my account",
             "too many ads", "nothing specific here"]
    chunks = pd.DataFrame({
        "source": "Google Play",
        "review_id": [str(i) for i in range(len(texts))],
        "date": pd.Timestamp("2026-09-01"),
        "p_negative": 0.9,
        "chunk_text": texts,
        "chunk_clean_text": texts,
    })
    tagged = tag_issues(chunks)
    found = dict(zip(tagged["key"], tagged["issue"]))
    assert found["Google Play-0"] == "Crashes and bugs"
    assert found["Google Play-1"] == "Login and account"
    assert found["Google Play-2"] == "Ads"
    assert found["Google Play-3"] == OTHER


def test_priority_levels():
    assert priority_level(30, 0.9) == "CRITICAL"
    assert priority_level(20, 0.9) == "HIGH"
    assert priority_level(8, 0.9) == "MEDIUM"
    assert priority_level(1, 0.9) == "LOW"


# ---------- trend detection ----------
def scores(values):
    return pd.DataFrame({"avg_sentiment": values})


def test_detect_direction_improving_and_declining():
    direction, change = detect_direction(scores([0.0, 0.1, 0.2, 0.3]))
    assert direction == "improving" and change > 0
    direction, change = detect_direction(scores([0.3, 0.2, 0.1, 0.0]))
    assert direction == "declining" and change < 0


def test_detect_direction_stable_and_not_enough_data():
    assert detect_direction(scores([0.1, 0.1, 0.1]))[0] == "stable"
    assert detect_direction(scores([0.1, 0.2]))[0] == "not enough data"


def test_recent_window_catches_a_late_decline():
    table = scores([0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0])
    assert detect_direction(table)[0] == "stable"
    assert detect_direction(table, last=4)[0] == "declining"


def test_trend_table_skips_periods_with_too_few_reviews():
    dates = list(pd.date_range("2026-09-01", periods=12, freq="h"))
    dates.append(pd.Timestamp("2026-09-02 10:00"))
    df = pd.DataFrame({
        "date": dates,
        "review_id": range(13),
        "sentiment_score": 0.5,
        "sentiment": "positive",
    })
    table = trend_table(df, "D", min_reviews=10)
    assert len(table) == 1
    assert table.iloc[0]["reviews"] == 12