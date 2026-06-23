Sentiment Analysis on Amazon Product Reviews

## 📌 Project Overview

This project performs **lexicon-based sentiment analysis** on a sample of Amazon product reviews.  
Each review is classified as **Positive**, **Neutral**, or **Negative** using NLTK's VADER analyser.  
The results are visualised and distilled into actionable business insights.

No deep learning, APIs, or dashboards are used — the focus is on clean, readable Python that any
data analyst can understand and extend.

---

## 📁 Project Structure

```
CodeAlpha_Sentiment_Analysis/
├── README.md
├── requirements.txt
├── main.py                          ← Pipeline entry point

├── data/
│   ├── raw/                         ← Original downloaded CSV
│   └── processed/
│       └── cleaned_reviews.csv      ← Auto-generated after run

├── analysis/
│   └── sentiment_analysis.py        ← VADER scoring & classification

├── visualization/
│   └── create_visualizations.py     ← Bar chart & pie chart

├── visualizations/                  ← Saved PNG charts (auto-generated)
│   ├── sentiment_bar_chart.png
│   └── sentiment_pie_chart.png

└── reports/                         ← (Reserved for future reports)
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python main.py
```

The script will automatically:

1. Download the Amazon Reviews dataset via `kagglehub`
2. Clean and preprocess the text
3. Classify each review using NLTK VADER
4. Print a sentiment report in the terminal
5. Save two charts to `visualizations/`
6. Print business insights

---

## 📊 Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| 1 | Load dataset (5 000 reviews) | Console summary |
| 2 | Clean text (lowercase, remove symbols) | `data/processed/cleaned_reviews.csv` |
| 3 | VADER sentiment scoring | `sentiment` column added |
| 4 | Sentiment pattern analysis | Console report |
| 5 | Visualisations | `visualizations/*.png` |
| 6 | Business insights | Console summary |

---

## 🔍 Sentiment Classification Rules (VADER)

| Compound Score | Label |
|---------------|-------|
| ≥ 0.05 | **Positive** |
| ≤ −0.05 | **Negative** |
| Between −0.05 and 0.05 | **Neutral** |

---

## 💼 Business Insights Generated

- **Marketing:** High positive sentiment → leverage in campaigns & testimonials.  
- **Product Development:** Negative reviews flag areas for improvement.  
- **Customer Experience:** Neutral reviews signal opportunities to delight customers further.

---

## 📦 Dataset

**Source:** [Amazon Product Reviews – Kaggle](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)  
**Download:** Handled automatically by `kagglehub` at runtime.  
**Sample used:** 5 000 reviews (configurable via `SAMPLE_SIZE` in `main.py`).

---

## 🛠 Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading & cleaning |
| `nltk` (VADER) | Sentiment scoring |
| `matplotlib` | Visualisations |
| `kagglehub` | Dataset download |

---

## 📝 Notes

- All paths are **relative** — the project runs from its own root directory.
- No internet connection is needed after the first run (data & VADER lexicon are cached).
- Adjust `SAMPLE_SIZE` in `main.py` to analyse more or fewer reviews.
