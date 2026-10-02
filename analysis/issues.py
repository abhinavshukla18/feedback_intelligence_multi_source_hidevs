import os
import re
import pandas as pd

CHUNKS_FILE = "data/chunk_sentiment.csv"
OUTPUT_FILE = "data/prioritized_issues.csv"
OTHER = "Other / uncategorized"

# Each pattern matches the START of a word, so "crash" also catches "crashes".
CATEGORIES = {
    "Crashes and bugs": r"\b(crash|bug|glitch|freez|froze|stuck|error|broken|not working|doesn['\s]?t work|won['\s]?t (open|load|work)|force clos|shut(s|ting)? ?down|shuts? off|keeps? closing|closes? (by itself|itself))",
    "Login and account": r"\b(log ?in|log ?out|logged|sign ?in|sign ?up|password|account|verif|authenticat)",
    "Playback and downloads": r"\b(download|offline|playback|buffer|stutter|skip|pause|stops? playing|cuts? out|won['\s]?t play|doesn['\s]?t play)",
    "Ads": r"\b(ads?|adverts?|advertis\w*|commercials?)\b",
    "Pricing and subscription": r"\b(price|pricing|expensive|subscription|subscribe|cancel|charg|refund|billing|paying|pay for|too much money)",
    "Slow performance and battery": r"\b(slow|lag|battery|drain|loading|takes forever|overheat|storage|memory)",
    "Update and UI changes": r"\b(update|redesign|layout|interface|new version|ui|design|used to)",
    "Search and recommendations": r"\b(search|recommend|algorithm|discover|suggest)",
    "Shuffle and queue": r"\b(shuffle|queue|repeat)",
    "Audio quality": r"\b(sound|audio|volume|equali[sz]er|bass|lossless)",
    "Privacy": r"\b(privacy|tracking|permission|collect|health data|personal data)",
}
PATTERNS = {name: re.compile(p) for name, p in CATEGORIES.items()}


def tag_issues(negative):
    """One row per (negative chunk, issue it mentions)."""
    rows = []
    for r in negative.itertuples():
        text = str(r.chunk_clean_text)
        matched = [n for n, p in PATTERNS.items() if p.search(text)] or [OTHER]
        for issue in matched:
            rows.append({
                "issue": issue,
                "key": f"{r.source}-{r.review_id}",
                "date": r.date,
                "p_negative": r.p_negative,
                "text": r.chunk_text,
            })
    return pd.DataFrame(rows)


def priority_level(share_pct, avg_negativity):
    if share_pct >= 25:
        return "CRITICAL"
    if share_pct >= 15:
        return "HIGH"
    if share_pct >= 5:
        return "MEDIUM"
    return "LOW"


def prioritize_issues(chunks):
    chunks = chunks.copy()
    chunks["date"] = pd.to_datetime(chunks["date"], errors="coerce")
    chunks = chunks.dropna(subset=["date"])
    chunks["review_id"] = chunks["review_id"].astype(str)

    negative = chunks[chunks["sentiment"] == "negative"]
    tagged = tag_issues(negative)
    total = tagged["key"].nunique()

    # Split the date range in half to spot problems that are growing
    midpoint = chunks["date"].min() + (chunks["date"].max() - chunks["date"].min()) / 2
    tagged["is_recent"] = tagged["date"] > midpoint

    summary = tagged.groupby("issue").agg(
        reviews=("key", "nunique"),
        avg_negativity=("p_negative", "mean"),
    )
    recent = tagged[tagged["is_recent"]].groupby("issue")["key"].nunique()
    earlier = tagged[~tagged["is_recent"]].groupby("issue")["key"].nunique()
    summary["recent"] = recent.reindex(summary.index).fillna(0).astype(int)
    summary["earlier"] = earlier.reindex(summary.index).fillna(0).astype(int)

    summary["share_pct"] = (summary["reviews"] / total * 100).round(1)
    summary["avg_negativity"] = summary["avg_negativity"].round(2)
    summary["priority_score"] = (summary["reviews"] * summary["avg_negativity"]).round(1)
    summary["level"] = [priority_level(s, a) for s, a in
                        zip(summary["share_pct"], summary["avg_negativity"])]
    recent_total = tagged[tagged["is_recent"]]["key"].nunique()
    earlier_total = tagged[~tagged["is_recent"]]["key"].nunique()
    summary["recent_pct"] = summary["recent"] / max(recent_total, 1) * 100
    summary["earlier_pct"] = summary["earlier"] / max(earlier_total, 1) * 100
    summary["emerging"] = ((summary["recent"] >= 5) &
                           (summary["recent_pct"] >= 1.3 * summary["earlier_pct"]) &
                           (summary["recent_pct"] - summary["earlier_pct"] >= 2))

    # The most negative example for each issue, avoiding repeats across issues
    used = set()
    examples = {}
    for issue in summary.sort_values("priority_score", ascending=False).index:
        candidates = tagged[tagged["issue"] == issue].sort_values(
            "p_negative", ascending=False)
        pick = candidates.iloc[0]
        for _, row in candidates.iterrows():
            if row["key"] not in used:
                pick = row
                break
        used.add(pick["key"])
        examples[issue] = pick["text"]
    summary["example"] = (pd.Series(examples)
                          .str.replace(r"\s+", " ", regex=True).str.slice(0, 120))

    if OTHER in summary.index:
        summary.loc[OTHER, "level"] = "n/a"
        summary.loc[OTHER, "emerging"] = False

    summary = summary.sort_values("priority_score", ascending=False).reset_index()
    covered = tagged[tagged["issue"] != OTHER]["key"].nunique()
    return summary, total, covered / total * 100


if __name__ == "__main__":
    chunks = pd.read_csv(CHUNKS_FILE, dtype={"review_id": str})
    summary, total, coverage = prioritize_issues(chunks)

    print(f"Negative reviews analysed: {total}")
    print(f"Reviews matched to a known issue: {coverage:.0f}%\n")

    pd.set_option("display.width", 200)
    cols = ["issue", "reviews", "share_pct", "avg_negativity",
            "priority_score", "level", "emerging"]
    print(summary[cols].to_string(index=False))

    print("\nExample for each CRITICAL or HIGH issue:")
    for r in summary[summary["level"].isin(["CRITICAL", "HIGH"])].itertuples():
        print(f"  [{r.level}] {r.issue}: {r.example}")

    os.makedirs("data", exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved to {OUTPUT_FILE}")