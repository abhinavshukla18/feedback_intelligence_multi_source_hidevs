import pandas as pd

COLUMNS = ["source", "review_id", "date", "rating",
           "text", "author", "app_version"]

# Left side: our name. Right side: the column name in the CSV file.
DEFAULT_COLUMN_MAP = {
    "date": "date",
    "rating": "rating",
    "text": "comment",
    "author": "name",
}


def load_csv_reviews(file_path, column_map=None):
    """Load survey feedback from a CSV file into our common format."""
    column_map = column_map or DEFAULT_COLUMN_MAP

    try:
        raw = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"CSV file not found: {file_path}")
        return pd.DataFrame(columns=COLUMNS)
    except Exception as error:
        print(f"Could not read CSV: {error}")
        return pd.DataFrame(columns=COLUMNS)

    # Check the CSV actually has the columns we need
    needed = [column_map["date"], column_map["rating"], column_map["text"]]
    missing = [col for col in needed if col not in raw.columns]
    if missing:
        print(f"CSV is missing required columns: {missing}")
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame()
    df["source"] = ["Survey CSV"] * len(raw)
    df["review_id"] = [f"csv-{i}" for i in range(len(raw))]
    df["date"] = pd.to_datetime(raw[column_map["date"]], errors="coerce")
    df["rating"] = pd.to_numeric(raw[column_map["rating"]], errors="coerce")
    df["text"] = raw[column_map["text"]]
    df["author"] = raw[column_map["author"]] if column_map.get("author") in raw.columns else None
    df["app_version"] = None

    # Drop rows with no comment text or an unreadable date
    df = df.dropna(subset=["text", "date"])
    return df[COLUMNS]


if __name__ == "__main__":
    df = load_csv_reviews("data/sample_survey.csv")
    print(f"Loaded {len(df)} survey responses")
    print(df.head())