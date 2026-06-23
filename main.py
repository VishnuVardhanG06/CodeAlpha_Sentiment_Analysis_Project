"""
main.py
-------
CodeAlpha Internship – Task 3: Sentiment Analysis
Entry point – orchestrates all pipeline steps.

Run:
    python main.py
"""

import os
import sys
import textwrap

import pandas as pd

# ── Make sure sub-packages are importable ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.sentiment_analysis import (
    run_sentiment_analysis,
    sentiment_summary,
    print_sentiment_report,
)
from visualization.create_visualizations import generate_all_charts


# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
CLEANED_CSV   = os.path.join(PROCESSED_DIR, "cleaned_reviews.csv")
SAMPLE_SIZE   = 5_000   # number of reviews to use (keeps runtime short)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 – Load dataset
# ════════════════════════════════════════════════════════════════════════════

def step1_load_dataset() -> pd.DataFrame:
    """
    Download (via kagglehub) and load the Amazon Product Reviews dataset.

    Returns
    -------
    pd.DataFrame
        Raw reviews DataFrame (limited to SAMPLE_SIZE rows).
    """
    print("\n" + "=" * 60)
    print("[Step 1] Loading dataset...")
    print("=" * 60)

    try:
        import kagglehub                                    # optional dependency
        dataset_path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
    except Exception as exc:
        print(f"  kagglehub download failed: {exc}")
        print("  Falling back to local raw/ directory...")
        dataset_path = RAW_DIR

    # Find the first CSV file inside the downloaded folder
    csv_file = None
    for root, _, files in os.walk(dataset_path):
        for fname in files:
            if fname.endswith(".csv"):
                csv_file = os.path.join(root, fname)
                break
        if csv_file:
            break

    if csv_file is None:
        raise FileNotFoundError(
            f"No CSV file found in: {dataset_path}\n"
            "Please place a raw CSV file inside data/raw/ and re-run."
        )

    print(f"  Reading: {csv_file}")
    df = pd.read_csv(csv_file, nrows=SAMPLE_SIZE)

    print(f"\n  Dataset shape  : {df.shape}")
    print(f"  Column names   : {list(df.columns)}")
    print(f"  Data types:\n{df.dtypes.to_string()}")

    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 – Clean the text data
# ════════════════════════════════════════════════════════════════════════════

def _resolve_review_column(df: pd.DataFrame) -> str:
    """
    Detect which column contains the review text.

    Checks common column names used in Amazon review datasets.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
        Name of the text column.
    """
    candidates = [
        "Text", "reviewText", "review_text", "review", "text",
        "Review Text", "Summary", "summary",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # Last resort: pick the column with the longest average string length
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if not str_cols:
        raise ValueError("No text column found in dataset.")
    return max(str_cols, key=lambda c: df[c].dropna().str.len().mean())


def clean_text(text: str) -> str:
    """
    Clean a single review string.

    Operations:
        1. Convert to lowercase
        2. Remove extra whitespace
        3. Remove non-alphanumeric symbols (keep letters, digits, spaces)

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned text string.
    """
    import re
    text = str(text).lower()                        # lowercase
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # remove symbols
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


def step2_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame and save the result to CSV.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a 'cleaned_review' column.
    """
    print("\n" + "=" * 60)
    print("[Step 2] Cleaning data...")
    print("=" * 60)

    text_col = _resolve_review_column(df)
    print(f"  Review column detected: '{text_col}'")

    original_rows = len(df)

    # 1. Drop rows where the review text is null
    df = df.dropna(subset=[text_col])
    print(f"  Removed {original_rows - len(df)} null rows  -> {len(df):,} remain")

    # 2. Drop exact duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before_dedup - len(df)} duplicate rows -> {len(df):,} remain")

    # 3. Apply text cleaning
    df = df.copy()
    df["cleaned_review"] = df[text_col].apply(clean_text)

    # 4. Remove any rows that became empty after cleaning
    df = df[df["cleaned_review"].str.len() > 0]

    print(f"  Final cleaned dataset: {df.shape[0]:,} rows")

    # Save to processed/
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False)
    print(f"  Saved -> {CLEANED_CSV}")

    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 & 4 – Sentiment Analysis + Patterns
# ════════════════════════════════════════════════════════════════════════════

def step3_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run VADER sentiment analysis and print the report.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with sentiment columns.
    """
    print("\n" + "=" * 60)
    print("[Step 3] Performing Sentiment Analysis (NLTK VADER)...")
    print("=" * 60)

    df = run_sentiment_analysis(df, text_column="cleaned_review")

    print("\n[Step 4] Analyzing sentiment patterns...")
    print("=" * 60)
    summary = sentiment_summary(df)
    print_sentiment_report(summary)

    return df, summary


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 – Visualizations (delegated to visualization module)
# ════════════════════════════════════════════════════════════════════════════

def step5_visualizations(summary: dict) -> None:
    """Generate and save all charts."""
    generate_all_charts(summary["counts"])


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 – Business Insights
# ════════════════════════════════════════════════════════════════════════════

def step6_business_insights(summary: dict) -> None:
    """
    Print actionable business insights derived from sentiment results.

    Parameters
    ----------
    summary : dict
        Output from sentiment_summary().
    """
    print("\n" + "=" * 60)
    print("[Step 6] Business Insights")
    print("=" * 60)

    counts  = summary["counts"]
    pct     = summary["percentages"]
    overall = summary["overall"]

    pos_pct = pct.get("Positive", 0)
    neg_pct = pct.get("Negative", 0)
    neu_pct = pct.get("Neutral",  0)

    insights = []

    # Insight 1 - Overall customer satisfaction
    if overall == "Positive":
        insights.append(
            "[+] Overall, customers are satisfied with the product. "
            f"({pos_pct:.1f}% positive reviews)"
        )
    elif overall == "Negative":
        insights.append(
            "[-] Overall customer sentiment is negative. "
            "Immediate product or service improvements are recommended."
        )
    else:
        insights.append(
            "[~] Customer sentiment is mostly neutral, indicating mixed or "
            "indifferent experiences."
        )

    # Insight 2 - Marketing opportunity
    if pos_pct >= 50:
        insights.append(
            "[Marketing] The high volume of positive reviews "
            "can be leveraged in campaigns and social proof strategies."
        )

    # Insight 3 - Product improvement flag
    if neg_pct >= 20:
        insights.append(
            "[Product Dev] A notable proportion of negative reviews "
            f"({neg_pct:.1f}%) suggests recurring pain points. "
            "Review negative feedback for improvement areas."
        )

    # Insight 4 - Neutral reviews
    if neu_pct >= 20:
        insights.append(
            "[CX] A significant share of neutral reviews "
            f"({neu_pct:.1f}%) indicates room to convert satisfied-but-not-delighted "
            "customers into loyal advocates through better engagement."
        )

    # Insight 5 - Volume note
    total = counts.sum()
    insights.append(
        f"[Data] Analysis is based on {total:,} reviews -- "
        "a representative sample for trend identification."
    )

    for i, insight in enumerate(insights, 1):
        wrapped = textwrap.fill(f"  {i}. {insight}", width=70, subsequent_indent="     ")
        print(wrapped)

    print("=" * 60)
    print("\n  Project complete. Check visualizations/ for charts.")
    print("  Cleaned data saved in data/processed/cleaned_reviews.csv")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Run the full sentiment analysis pipeline."""
    print("\n" + "=" * 60)
    print("  CodeAlpha Internship -- Task 3: Sentiment Analysis")
    print("=" * 60)

    # Step 1: Load
    raw_df = step1_load_dataset()

    # Step 2: Clean
    cleaned_df = step2_clean_data(raw_df)

    # Step 3 & 4: Analyse + Report
    analysed_df, summary = step3_sentiment_analysis(cleaned_df)

    # Step 5: Visualise
    step5_visualizations(summary)

    # Step 6: Insights
    step6_business_insights(summary)


if __name__ == "__main__":
    main()
