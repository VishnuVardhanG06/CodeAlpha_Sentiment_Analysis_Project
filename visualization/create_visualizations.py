"""
create_visualizations.py
------------------------
CodeAlpha Internship  Task 3: Sentiment Analysis
Module: Visualization

Generates:
    1. Sentiment Distribution Bar Chart  → visualizations/sentiment_bar_chart.png
    2. Sentiment Distribution Pie Chart  → visualizations/sentiment_pie_chart.png

All charts are saved as high-resolution PNG files.
"""

import os
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# ── Colour palette (consistent across both charts) ─────────────────────────
COLORS = {
    "Positive": "#2ecc71",   # green
    "Neutral":  "#f39c12",   # amber
    "Negative": "#e74c3c",   # red
}

OUTPUT_DIR = os.path.join("visualizations")


def _ensure_output_dir() -> None:
    """Create the visualizations/ directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_bar_chart(counts: pd.Series) -> str:
    """
    Create and save a sentiment distribution bar chart.

    Parameters
    ----------
    counts : pd.Series
        Series with sentiment labels as index and review counts as values.

    Returns
    -------
    str
        File path of the saved chart.
    """
    _ensure_output_dir()

    # Ensure consistent order
    labels = ["Positive", "Neutral", "Negative"]
    values = [counts.get(lbl, 0) for lbl in labels]
    colors = [COLORS[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    # Annotate bars with count values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:,}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#333333"
        )

    ax.set_title("Sentiment Distribution of Amazon Reviews", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "sentiment_bar_chart.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Bar chart saved -> {filepath}")
    return filepath


def plot_pie_chart(counts: pd.Series) -> str:
    """
    Create and save a sentiment distribution pie chart.

    Parameters
    ----------
    counts : pd.Series
        Series with sentiment labels as index and review counts as values.

    Returns
    -------
    str
        File path of the saved chart.
    """
    _ensure_output_dir()

    labels = ["Positive", "Neutral", "Negative"]
    values = [counts.get(lbl, 0) for lbl in labels]
    colors = [COLORS[lbl] for lbl in labels]

    # Explode the dominant slice slightly for emphasis
    max_idx = values.index(max(values))
    explode = [0.05 if i == max_idx else 0 for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(7, 7))

    # ax.pie() returns (wedges, texts) or (wedges, texts, autotexts) depending
    # on whether autopct is set. Since we always pass autopct, we always get 3
    # values — the type checker just can't infer that, so we suppress it.
    pie_result = ax.pie(
        values,
        labels=None,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.80,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    wedges, texts, autotexts = pie_result  # type: ignore[misc]


    # Style percentage labels
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    # Custom legend
    legend_patches = [
        mpatches.Patch(color=COLORS[lbl], label=f"{lbl}  ({counts.get(lbl, 0):,})")
        for lbl in labels
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=11,
        frameon=False,
    )

    ax.set_title("Sentiment Distribution of Amazon Reviews", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "sentiment_pie_chart.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Pie chart saved  -> {filepath}")
    return filepath


def generate_all_charts(counts: pd.Series) -> list:
    """
    Generate all visualizations and return their file paths.

    Parameters
    ----------
    counts : pd.Series
        Sentiment value counts.

    Returns
    -------
    list of str
        Paths to all saved chart images.
    """
    print("\n[Step 5] Generating visualizations...")
    bar_path = plot_bar_chart(counts)
    pie_path = plot_pie_chart(counts)
    print("  All charts saved successfully.")
    return [bar_path, pie_path]
