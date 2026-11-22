# pyspark-movie-reviews

Basic data processing of the **IMDb 50K Movie Review dataset** with **Apache Spark (PySpark)**: load → explore → clean → analyze → transform → save.

> Module 1 – Homework 1.2 · Shashank Reddy Reddimalla (1RV23CD050)

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle), file `IMDB Dataset.csv`:

| column | type | description |
|---|---|---|
| `review` | string | full review text (can contain commas, quotes, HTML `<br />` tags and line breaks) |
| `sentiment` | string | `positive` or `negative` |

The CSV (~63 MB) is **not** committed to this repo. Download it from Kaggle, unzip it and put it in `data/`.

## Project structure

```
pyspark-movie-reviews/
├── movie_review_analysis.py     # main PySpark script (all 6 tasks)
├── movie_review_analysis.ipynb  # same pipeline as a notebook (runs in Google Colab)
├── requirements.txt
├── .gitignore
└── data/                        # put "IMDB Dataset.csv" here (git-ignored)
```

## What the code does

| # | Task | PySpark used |
|---|---|---|
| 1 | **Load** the CSV into a DataFrame | `spark.read.csv(...)` with an explicit schema and `multiLine` + `escape` options so reviews containing quotes and line breaks are parsed as one row |
| 2 | **Explore**: print schema, show first 5 rows | `printSchema()`, `show(5)` |
| 3 | **Clean**: remove null or empty reviews | `filter(col("review").isNotNull() & (trim(col("review")) != ""))`; sentiment labels are trimmed and lower-cased |
| 4 | **Analyze**: total reviews; positive vs negative counts | `count()`, `groupBy("sentiment").agg(count("*"))` |
| 5 | **Transform**: add `review_length`; show reviews longer than 500 characters | `withColumn("review_length", length("review"))`, `filter(col("review_length") > 500)` |
| 6 | **Save** the cleaned + transformed DataFrame as a new CSV | `coalesce(1).write.option("header", True).csv(...)` (falls back to pandas on Windows without `winutils.exe`) |

Extra: average / min / max review length per sentiment, to see whether positive or negative reviewers write more.

## How to run

### Option A – Google Colab (easiest, nothing to install)
1. Open [colab.research.google.com](https://colab.research.google.com) → *File → Upload notebook* → `movie_review_analysis.ipynb`.
2. *Runtime → Run all*. When prompted, upload `IMDB Dataset.csv`.

### Option B – Local machine
Requirements: Python 3.9+ and Java 8, 11 or 17 (`java -version`).

```bash
pip install -r requirements.txt
python movie_review_analysis.py
# or with custom paths
python movie_review_analysis.py --input "data/IMDB Dataset.csv" --output output/cleaned_reviews
```

Output: `output/cleaned_reviews/part-00000-*.csv` (columns `review, sentiment, review_length`).

## Notes on the design

- **Explicit schema** instead of `inferSchema`: avoids an extra full pass over the file and guarantees both columns are strings.
- **`cache()` after cleaning**: the cleaned DataFrame is reused by several actions (counts, group-bys, filters, write), so caching avoids re-reading and re-parsing the CSV each time.
- **`coalesce(1)` before writing**: produces a single CSV file instead of one file per partition (fine for 50K rows; for big data you would keep multiple partitions).
- **Windows**: the script sets `PYSPARK_PYTHON` to the current interpreter to avoid the common *"Python worker failed to connect back"* error.
