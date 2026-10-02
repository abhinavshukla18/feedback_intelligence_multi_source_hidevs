import os
from datetime import datetime, timedelta
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")  # draw charts without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from analysis.issues import OTHER, prioritize_issues
from analysis.trends import (choose_period, detect_direction,
                             load_scored_reviews, trend_table)

CHUNKS_FILE = "data/chunk_sentiment.csv"
OUTPUT_DIR = "reports"
CHART_DIR = "reports/charts"
GREEN = "#1DB954"
LEVEL_COLORS = {"CRITICAL": "#c0392b", "HIGH": "#e67e22",
                "MEDIUM": "#f1c40f", "LOW": "#95a5a6"}


# ---------- helpers ----------
def safe(text):
    """Make review text safe for the PDF (no emojis, no markup characters)."""
    text = str(text).replace("\n", " ")
    text = text.encode("cp1252", "ignore").decode("cp1252")
    return escape(text)


def window(frame, start, stop):
    return frame[(frame["date"] > start) & (frame["date"] <= stop)]


def week_stats(df):
    return {
        "reviews": len(df),
        "rating": df["rating"].mean(),
        "pct_negative": (df["sentiment"] == "negative").mean() * 100,
        "score": df["sentiment_score"].mean(),
    }


def safe_prioritize(chunks):
    if (chunks["sentiment"] == "negative").sum() < 10:
        return None, 0, 0
    return prioritize_issues(chunks)


def issue_rows(summary, prev_summary):
    prev_share = {}
    if prev_summary is not None:
        prev_share = prev_summary.set_index("issue")["share_pct"].to_dict()
    rows = []
    for r in summary[summary["issue"] != OTHER].head(8).itertuples():
        rows.append({
            "issue": r.issue, "level": r.level, "reviews": r.reviews,
            "share": r.share_pct,
            "change": r.share_pct - prev_share.get(r.issue, 0.0),
            "example": r.example,
        })
    return rows


# ---------- charts ----------
def make_trend_chart(df, path):
    """Weekly sentiment line, using only sources with real history."""
    long_sources = [s for s, g in df.groupby("source")
                    if len(g) >= 30 and (g["date"].max() - g["date"].min()).days >= 14]
    data = df[df["source"].isin(long_sources)]
    if data.empty:
        return None
    period = choose_period(data)
    table = trend_table(data, period)
    if len(table) < 3:
        return None
    unit = "week" if period == "W" else "day"

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(table["period"], table["avg_sentiment"], marker="o", color=GREEN)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Avg sentiment score")
    ax.set_title(f"Average sentiment per {unit} ({', '.join(long_sources)})", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"path": path, "n": len(table), "unit": unit,
            "overall": detect_direction(table),
            "recent": detect_direction(table, last=4)}


def make_mix_chart(df, path):
    counts = (df["sentiment"].value_counts()
              .reindex(["negative", "neutral", "positive"]).fillna(0))
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(counts.index, counts.values, color=["#e74c3c", "#95a5a6", GREEN])
    ax.set_title("Sentiment mix this week", fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_issue_chart(summary, path):
    top = summary[summary["issue"] != OTHER].head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.barh(top["issue"], top["priority_score"],
            color=[LEVEL_COLORS.get(level, "#95a5a6") for level in top["level"]])
    ax.set_title("Top issues (priority score)", fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- text ----------
def build_takeaways(this_m, prev_m, rows, trend):
    points = []
    diff = this_m["score"] - prev_m["score"]
    direction = "up" if diff > 0 else "down"
    points.append(f"Average sentiment is {direction} {abs(diff):.2f} versus last week "
                  f"({prev_m['score']:+.2f} to {this_m['score']:+.2f}).")
    points.append(f"{this_m['pct_negative']:.0f}% of reviews were negative, compared "
                  f"with {prev_m['pct_negative']:.0f}% the week before.")
    if rows:
        top = rows[0]
        points.append(f"Top issue: {top['issue']} ({top['level']}), present in "
                      f"{top['share']:.0f}% of negative reviews.")
        rising = [r for r in rows if r["reviews"] >= 5 and r["change"] >= 2]
        if rising:
            fastest = max(rising, key=lambda r: r["change"])
            points.append(f"Fastest-growing complaint: {fastest['issue']} "
                          f"(+{fastest['change']:.1f} percentage points of negative reviews).")
    if trend:
        points.append(f"Longer-term trend over {trend['n']} {trend['unit']}s: "
                      f"{trend['overall'][0]}. Over the last 4 {trend['unit']}s: "
                      f"{trend['recent'][0]}.")
    return points


def draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm,
                      f"Feedback Intelligence Report | Generated {datetime.now():%d %b %Y %H:%M}")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ---------- main ----------
def generate_weekly_report():
    df = load_scored_reviews()
    chunks = pd.read_csv(CHUNKS_FILE, dtype={"review_id": str})
    chunks["date"] = pd.to_datetime(chunks["date"], errors="coerce")
    chunks = chunks.dropna(subset=["date"])

    end = df["date"].max()
    this_start = end - timedelta(days=7)
    prev_start = end - timedelta(days=14)

    this_all = window(df, this_start, end)
    prev_all = window(df, prev_start, this_start)

    # Only compare sources that have enough reviews in BOTH weeks
    comparable = [s for s in df["source"].unique()
                  if (this_all["source"] == s).sum() >= 30
                  and (prev_all["source"] == s).sum() >= 30]
    if not comparable:
        print("Not enough data in both weeks to build a report.")
        return None

    this_df = this_all[this_all["source"].isin(comparable)]
    prev_df = prev_all[prev_all["source"].isin(comparable)]
    this_chunks = window(chunks, this_start, end)
    this_chunks = this_chunks[this_chunks["source"].isin(comparable)]
    prev_chunks = window(chunks, prev_start, this_start)
    prev_chunks = prev_chunks[prev_chunks["source"].isin(comparable)]

    this_m, prev_m = week_stats(this_df), week_stats(prev_df)
    this_summary, total_neg, coverage = safe_prioritize(this_chunks)
    prev_summary, _, _ = safe_prioritize(prev_chunks)
    rows = issue_rows(this_summary, prev_summary) if this_summary is not None else []

    # Charts
    os.makedirs(CHART_DIR, exist_ok=True)
    trend = make_trend_chart(df, f"{CHART_DIR}/trend.png")
    make_mix_chart(this_df, f"{CHART_DIR}/mix.png")
    if rows:
        make_issue_chart(this_summary, f"{CHART_DIR}/issues.png")

    # ----- Build the PDF -----
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=10)
    grey = ParagraphStyle("Grey", parent=small, textColor=colors.grey)

    story = [
        Paragraph("Weekly Feedback Report: Spotify", styles["Title"]),
        Paragraph(f"{this_start:%d %b} to {end:%d %b %Y} | Compared with the previous 7 days "
                  f"| Sources: {safe(', '.join(comparable))}", grey),
        Spacer(1, 0.2 * cm),
        Paragraph("Sources without enough history in both weeks are left out of the "
                  "week-over-week comparison.", grey),
        Spacer(1, 0.4 * cm),
        Paragraph("Key takeaways", styles["Heading2"]),
    ]
    for point in build_takeaways(this_m, prev_m, rows, trend):
        story.append(Paragraph("&bull; " + safe(point), body))

    story += [Spacer(1, 0.4 * cm), Paragraph("This week at a glance", styles["Heading2"])]
    metrics = [
        ["Metric", "This week", "Last week", "Change"],
        ["Reviews", f"{this_m['reviews']:,}", f"{prev_m['reviews']:,}",
         f"{this_m['reviews'] - prev_m['reviews']:+,}"],
        ["Average rating", f"{this_m['rating']:.2f}", f"{prev_m['rating']:.2f}",
         f"{this_m['rating'] - prev_m['rating']:+.2f}"],
        ["Negative reviews", f"{this_m['pct_negative']:.1f}%", f"{prev_m['pct_negative']:.1f}%",
         f"{this_m['pct_negative'] - prev_m['pct_negative']:+.1f} pts"],
        ["Avg sentiment score", f"{this_m['score']:+.2f}", f"{prev_m['score']:+.2f}",
         f"{this_m['score'] - prev_m['score']:+.2f}"],
    ]
    mt = Table(metrics, colWidths=[5 * cm, 4 * cm, 4 * cm, 4 * cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(GREEN)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story += [mt, Spacer(1, 0.4 * cm)]

    if trend:
        story.append(Image(trend["path"], width=17 * cm, height=17 * cm * 3 / 7))
    pair = [Image(f"{CHART_DIR}/mix.png", width=6 * cm, height=6 * cm)]
    if rows:
        pair.append(Image(f"{CHART_DIR}/issues.png", width=9.2 * cm, height=6 * cm))
    story.append(Table([pair]))

    if rows:
        story += [Spacer(1, 0.4 * cm), Paragraph("Top issues", styles["Heading2"]),
                  Paragraph(f"Based on {total_neg} negative reviews; {coverage:.0f}% were "
                            "matched to a known issue. One review can mention several issues. "
                            "Share = percentage of negative reviews mentioning the issue; "
                            "Change = difference from last week.",
                            grey), Spacer(1, 0.2 * cm)]
        data = [["Issue", "Level", "Reviews", "Share", "Change", "Example"]]
        for r in rows:
            data.append([Paragraph(safe(r["issue"]), small), r["level"], str(r["reviews"]),
                         f"{r['share']:.1f}%", f"{r['change']:+.1f} pts",
                         Paragraph(safe(r["example"]), small)])
        it = Table(data, colWidths=[3.2 * cm, 1.8 * cm, 1.5 * cm, 2.0 * cm, 1.8 * cm, 6.7 * cm],
                   repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(GREEN)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        for i, r in enumerate(rows, start=1):
            style.append(("BACKGROUND", (1, i), (1, i), colors.HexColor(LEVEL_COLORS[r["level"]])))
            if r["level"] in ("CRITICAL", "HIGH"):
                style.append(("TEXTCOLOR", (1, i), (1, i), colors.white))
        it.setStyle(TableStyle(style))
        story.append(it)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = f"{OUTPUT_DIR}/weekly_report_{end:%Y-%m-%d}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title="Weekly Feedback Report")
    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return pdf_path


if __name__ == "__main__":
    path = generate_weekly_report()
    if path:
        print(f"Report saved to {path}")