import os
import numpy as np
import pandas as pd

REVIEWS_FILE = "data/review_sentiment.csv"
TREND_OUTPUT = "data/sentiment_trend.csv"


def load_scored_reviews(path=REVIEWS_FILE):
    df = pd.read_csv(path, dtype={"review_id": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # -1 = very negative, +1 = very positive
    df["sentiment_score"] = df["p_positive"] - df["p_negative"]
    return df


def choose_period(df):
    """Use weeks if the data spans 4+ weeks, otherwise days."""
    span_days = (df["date"].max() - df["date"].min()).days
    return "W" if span_days >= 28 else "D"


def trend_table(df, period, min_reviews=10):
    """Average sentiment per day/week. Skips periods with too few reviews."""
    df = df.copy()
    if period == "W":
        df["period"] = df["date"].dt.to_period("W").dt.start_time
    else:
        df["period"] = df["date"].dt.normalize()

    table = df.groupby("period").agg(
        reviews=("review_id", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        pct_negative=("sentiment", lambda s: (s == "negative").mean() * 100),
    ).reset_index()

    table = table[table["reviews"] >= min_reviews].reset_index(drop=True)
    table["avg_sentiment"] = table["avg_sentiment"].round(3)
    table["pct_negative"] = table["pct_negative"].round(1)
    return table


def detect_direction(table, threshold=0.05, last=None):
    """Fit a line through the averages and report the overall change."""
    if last:
        table = table.tail(last)
    if len(table) < 3:
        return "not enough data", 0.0
    x = np.arange(len(table))
    slope = np.polyfit(x, table["avg_sentiment"], 1)[0]
    change = float(slope * (len(table) - 1))
    if change > threshold:
        direction = "improving"
    elif change < -threshold:
        direction = "declining"
    else:
        direction = "stable"
    return direction, round(change, 3)


if __name__ == "__main__":
    df = load_scored_reviews()
    period = choose_period(df)
    unit = "week" if period == "W" else "day"

    print(f"Reviews: {len(df)}")
    print("Date range per source:")
    print(df.groupby("source")["date"].agg(["min", "max"]))
    print(f"\nGrouping by {unit}")

    groups = []
    for name, g in df.groupby("source"):
        span = (g["date"].max() - g["date"].min()).days
        if len(g) >= 30 and span >= 14:   # only sources with real history
            groups.append((name, g))

    all_tables = []
    for name, group in groups:
        table = trend_table(group, period)
        direction, change = detect_direction(table)
        recent_dir, recent_change = detect_direction(table, last=4)
        print(f"\n{name}: overall, sentiment is {direction} "
              f"(change of {change:+} over {len(table)} {unit}s)")
        print(f"{name}: over the last 4 {unit}s it is {recent_dir} "
              f"(change of {recent_change:+})")
        print(table.to_string(index=False))
        table.insert(0, "source", name)
        all_tables.append(table)

    os.makedirs("data", exist_ok=True)
    pd.concat(all_tables).to_csv(TREND_OUTPUT, index=False, encoding="utf-8")
    print(f"\nSaved to {TREND_OUTPUT}")