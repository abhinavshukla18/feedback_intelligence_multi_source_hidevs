import pandas as pd
import streamlit as st
import os

from analysis.trends import choose_period, trend_table, detect_direction
from analysis.issues import prioritize_issues
from analysis.ai_engine import ask_reviews
from analysis.embeddings import load_model
from reports.report_generator import generate_weekly_report

REVIEWS_FILE = "data/review_sentiment.csv"
CHUNKS_FILE = "data/chunk_sentiment.csv"

st.set_page_config(page_title="Feedback Intelligence", page_icon="📊", layout="wide")

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # running locally without a secrets file: the .env file is used


@st.cache_data
def load_data():
    reviews = pd.read_csv(REVIEWS_FILE, dtype={"review_id": str})
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews = reviews.dropna(subset=["date"])
    reviews["sentiment_score"] = reviews["p_positive"] - reviews["p_negative"]

    chunks = pd.read_csv(CHUNKS_FILE, dtype={"review_id": str})
    chunks["date"] = pd.to_datetime(chunks["date"], errors="coerce")
    return reviews, chunks


@st.cache_resource
def get_embedding_model():
    return load_model()


try:
    reviews, chunks = load_data()
except FileNotFoundError:
    st.error("Data files not found. Run the pipeline first (collect_data.py, "
             "then the analysis steps).")
    st.stop()

st.title(" Multi-Source Feedback Intelligence")
st.caption(f"Latest review in the data: {reviews['date'].max():%d %b %Y, %H:%M}")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
min_date = reviews["date"].min().date()
max_date = reviews["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date)
if len(date_range) != 2:
    st.info("Pick an end date to continue.")
    st.stop()
start, end = date_range

all_sources = sorted(reviews["source"].unique())
sources = st.sidebar.multiselect("Source", all_sources, default=all_sources)
sentiments = st.sidebar.multiselect(
    "Sentiment", ["negative", "neutral", "positive"],
    default=["negative", "neutral", "positive"])
min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)

st.sidebar.divider()
if st.sidebar.button("Generate weekly PDF report"):
    with st.spinner("Building report..."):
        pdf_path = generate_weekly_report()
    if pdf_path:
        with open(pdf_path, "rb") as f:
            st.sidebar.download_button(
                "Download report", f.read(),
                file_name=os.path.basename(pdf_path), mime="application/pdf")
    else:
        st.sidebar.warning("Not enough data to build a report.")

# ---------- Apply filters ----------
mask = (
    (reviews["date"].dt.date >= start) & (reviews["date"].dt.date <= end)
    & reviews["source"].isin(sources)
    & reviews["sentiment"].isin(sentiments)
    & (reviews["confidence"] >= min_conf)
)
view = reviews[mask]

chunk_mask = (
    (chunks["date"].dt.date >= start) & (chunks["date"].dt.date <= end)
    & chunks["source"].isin(sources)
)
chunk_view = chunks[chunk_mask]

if view.empty:
    st.warning("No reviews match these filters. Try widening them.")
    st.stop()

# ---------- Metric cards ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(view):,}")
c2.metric("Average rating", f"{view['rating'].mean():.2f} ★")
c3.metric("Negative reviews", f"{(view['sentiment'] == 'negative').mean() * 100:.1f}%")
c4.metric("Avg sentiment score", f"{view['sentiment_score'].mean():+.2f}")

tab_trends, tab_issues, tab_reviews, tab_ask = st.tabs(
    [" Trends", " Issues", " Reviews", " Ask the reviews"])

# ---------- Trends ----------
with tab_trends:
    st.subheader("Sentiment over time")
    # Only sources with real history (2+ weeks) are used for trends
    long_sources = [s for s, g in view.groupby("source")
                    if (g["date"].max() - g["date"].min()).days >= 14]
    trend_view = view[view["source"].isin(long_sources)]
    if trend_view.empty:
        period, unit, table = "D", "day", pd.DataFrame()
    else:
        period = choose_period(trend_view)
        unit = "week" if period == "W" else "day"
        table = trend_table(trend_view, period)

    if len(table) < 3:
        st.info("Not enough data in this selection to show a trend. "
                "Widen the date range or include more sources.")
    else:
        direction, change = detect_direction(table)
        recent_dir, recent_change = detect_direction(table, last=4)
        a, b = st.columns(2)
        a.metric(f"Overall trend ({len(table)} {unit}s)", direction, f"{change:+.3f}")
        b.metric(f"Last 4 {unit}s", recent_dir, f"{recent_change:+.3f}")

        st.write("Average sentiment score (-1 very negative, +1 very positive)")
        st.line_chart(table.set_index("period")["avg_sentiment"])
        st.write("Share of negative reviews (%)")
        st.bar_chart(table.set_index("period")["pct_negative"])

    st.caption("Trends use only sources with at least 2 weeks of history "
               "(currently Google Play).")

# ---------- Issues ----------
with tab_issues:
    st.subheader("Prioritized issues")
    st.caption("Based on negative reviews in the selected dates and sources. "
               "The sentiment filter does not apply here.")
    if (chunk_view["sentiment"] == "negative").sum() < 10:
        st.info("Too few negative reviews in this selection to prioritize issues.")
    else:
        summary, total, coverage = prioritize_issues(chunk_view)
        st.write(f"{total} negative reviews analysed, "
                 f"{coverage:.0f}% matched to a known issue.")
        show = summary[["issue", "level", "reviews", "share_pct",
                        "priority_score", "emerging", "example"]]
        st.dataframe(show, hide_index=True)
        st.write("Priority score by issue")
        st.bar_chart(summary.set_index("issue")["priority_score"])

# ---------- Reviews browser ----------
with tab_reviews:
    st.subheader(f"{len(view):,} reviews")
    cols = ["date", "source", "rating", "sentiment", "confidence", "text"]
    st.dataframe(view.sort_values("date", ascending=False)[cols], hide_index=True)

# ---------- Ask the reviews ----------
with tab_ask:
    st.subheader("Ask the reviews")
    st.caption("Searches all collected reviews and summarises the answer with an AI model.")
    question = st.text_input("Your question",
                             placeholder="What do people say about login problems?")
    if st.button("Ask") and question.strip():
        with st.spinner("Searching reviews and thinking..."):
            result = ask_reviews(question, embed_model=get_embedding_model())
        st.markdown(result["answer"])
        with st.expander("Reviews used for this answer"):
            for i, hit in enumerate(result["reviews"], start=1):
                st.write(f"[{i}] ({hit['source']}, {hit['rating']:.0f}★, "
                         f"similarity {hit['similarity']}) {hit['text']}")