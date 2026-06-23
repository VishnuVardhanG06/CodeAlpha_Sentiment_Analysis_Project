"""
sentiment_analysis.py
---------------------
CodeAlpha Internship – Task 3: Sentiment Analysis
Module: Core Sentiment Analysis using NLTK VADER

This module loads the cleaned dataset, applies VADER sentiment analysis,
classifies each review as Positive, Neutral, or Negative, and returns
the enriched DataFrame along with a summary report.
"""

import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def download_vader_lexicon():
    """Download VADER lexicon if not already present."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        print("  Downloading VADER lexicon...")
        nltk.download("vader_lexicon", quiet=True)
    print("  VADER lexicon is ready.")


def load_cleaned_data(filepath: str) -> pd.DataFrame:
    """
    Load the cleaned reviews CSV file.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the cleaned reviews.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {filepath}\n"
            "Please run the data-cleaning step first."
        )
    df = pd.read_csv(filepath)
    print(f"  Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def classify_sentiment(compound_score: float) -> str:
    """
    Classify a VADER compound score into a sentiment label.

    VADER thresholds (standard):
        compound >= 0.05  → Positive
        compound <= -0.05 → Negative
        else              → Neutral

    Parameters
    ----------
    compound_score : float
        VADER compound sentiment score in range [-1, 1].

    Returns
    -------
    str
        One of 'Positive', 'Neutral', or 'Negative'.
    """
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def run_sentiment_analysis(df: pd.DataFrame, text_column: str = "cleaned_review") -> pd.DataFrame:
    """
    Apply VADER sentiment analysis to every review in the DataFrame.

    Adds four new columns:
        - vader_neg     : negative sentiment score
        - vader_neu     : neutral sentiment score
        - vader_pos     : positive sentiment score
        - vader_compound: overall compound score
        - sentiment     : final label (Positive / Neutral / Negative)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the cleaned text column.
    text_column : str
        Name of the column that holds the cleaned review text.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new sentiment columns appended.
    """
    download_vader_lexicon()
    sid = SentimentIntensityAnalyzer()

    print(f"  Running VADER on {len(df):,} reviews...")

    # Compute polarity scores for each review
    scores = df[text_column].apply(sid.polarity_scores)

    df["vader_neg"]      = scores.apply(lambda x: x["neg"])
    df["vader_neu"]      = scores.apply(lambda x: x["neu"])
    df["vader_pos"]      = scores.apply(lambda x: x["pos"])
    df["vader_compound"] = scores.apply(lambda x: x["compound"])

    # Apply classification rule
    df["sentiment"] = df["vader_compound"].apply(classify_sentiment)

    print("  Sentiment classification complete.")
    return df


def sentiment_summary(df: pd.DataFrame) -> dict:
    """
    Compute count and percentage for each sentiment class.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'sentiment' column.

    Returns
    -------
    dict
        Dictionary with keys 'counts' (pd.Series) and
        'percentages' (pd.Series) and 'overall' (str).
    """
    counts      = df["sentiment"].value_counts()
    percentages = (counts / len(df) * 100).round(2)

    # Overall opinion is the majority class
    overall = counts.idxmax()

    return {
        "counts":      counts,
        "percentages": percentages,
        "overall":     overall,
    }


def print_sentiment_report(summary: dict) -> None:
    """
    Pretty-print the sentiment analysis summary to the console.

    Parameters
    ----------
    summary : dict
        Output from sentiment_summary().
    """
    counts      = summary["counts"]
    percentages = summary["percentages"]
    overall     = summary["overall"]

    print("\n" + "=" * 50)
    print("       SENTIMENT ANALYSIS REPORT")
    print("=" * 50)

    for label in ["Positive", "Neutral", "Negative"]:
        count = counts.get(label, 0)
        pct   = percentages.get(label, 0.0)
        print(f"  {label:<10}: {count:>6,}  ({pct:.2f}%)")

    print("-" * 50)
    print(f"  Total Reviews : {counts.sum():,}")
    print(f"  Overall Opinion: {overall}")
    print("=" * 50)
