"""
pyspark-movie-reviews / movie_review_analysis.py
================================================
Module 1 - Homework 1.2 : basic PySpark processing of the IMDb 50K movie review dataset.

Author : Shashank Reddy Reddimalla (1RV23CD050)
Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
         (file "IMDB Dataset.csv", columns: review, sentiment)

Steps
  1. Load      - read the CSV into a Spark DataFrame
  2. Explore   - print the schema, show the first 5 rows
  3. Clean     - remove rows whose review is null or empty
  4. Analyze   - total reviews, positive vs negative counts
  5. Transform - add review_length (characters); show reviews longer than 500 chars
  6. Save      - write the cleaned + transformed DataFrame to a new CSV

Usage
  python movie_review_analysis.py                       # uses data/IMDB Dataset.csv
  python movie_review_analysis.py --input "path/to/IMDB Dataset.csv" --output output/cleaned_reviews
"""

import argparse
import os
import sys

# On Windows, Spark must launch its Python workers with the same interpreter as the
# driver; without this you often get "Python worker failed to connect back".
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

LONG_REVIEW_THRESHOLD = 500  # characters

# Explicit schema: faster than inferSchema (no extra pass over the file) and guarantees
# both columns are read as strings.
REVIEW_SCHEMA = StructType([
    StructField("review", StringType(), nullable=True),
    StructField("sentiment", StringType(), nullable=True),
])


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n {title}\n" + "=" * 70)


def create_spark_session(app_name: str = "IMDbMovieReviewAnalysis") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")                          # use all local CPU cores
        .config("spark.sql.shuffle.partitions", "8")  # small data -> fewer shuffle tasks
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")           # keep the console readable
    return spark


# ----------------------------------------------------------------------------- 1. LOAD
def load_reviews(spark: SparkSession, path: str) -> DataFrame:
    """Read the IMDb CSV. Reviews contain commas, double quotes and line breaks,
    so multiLine + quote/escape options are needed to parse every row correctly."""
    return (
        spark.read
        .option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .schema(REVIEW_SCHEMA)
        .csv(path)
    )


# ----------------------------------------------------------------------------- 3. CLEAN
def clean_reviews(df: DataFrame) -> DataFrame:
    """Drop rows whose review is null, empty or whitespace-only, and normalise the
    sentiment label (trim + lower-case) so 'Positive ' and 'positive' are counted together."""
    return (
        df.filter(F.col("review").isNotNull() & (F.trim(F.col("review")) != ""))
          .withColumn("sentiment", F.lower(F.trim(F.col("sentiment"))))
    )


# ----------------------------------------------------------------------------- 5. TRANSFORM
def add_review_length(df: DataFrame) -> DataFrame:
    """New column review_length = number of characters in the review."""
    return df.withColumn("review_length", F.length(F.col("review")))


# ----------------------------------------------------------------------------- 6. SAVE
def save_csv(df: DataFrame, output_dir: str) -> str:
    """Write the DataFrame as a single CSV file with a header.

    Spark writes a *folder* (output_dir/part-0000-....csv). coalesce(1) makes it one file.
    On Windows without Hadoop's winutils.exe, Spark's writer can fail; in that case we fall
    back to pandas so the result is still saved (50k rows fits easily in memory)."""
    try:
        (
            df.coalesce(1)
              .write.mode("overwrite")
              .option("header", True)
              .option("quote", '"')
              .option("escape", '"')
              .csv(output_dir)
        )
        return output_dir + os.sep + "part-*.csv"
    except Exception as err:  # noqa: BLE001 - we want any writer failure to trigger the fallback
        print(f"[warn] Spark CSV writer failed ({type(err).__name__}); saving with pandas instead.")
        fallback = output_dir.rstrip("/\\") + ".csv"
        os.makedirs(os.path.dirname(fallback) or ".", exist_ok=True)
        df.toPandas().to_csv(fallback, index=False)
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="PySpark analysis of the IMDb 50K movie review dataset")
    parser.add_argument("--input", default=os.path.join("data", "IMDB Dataset.csv"), help="path to IMDB Dataset.csv")
    parser.add_argument("--output", default=os.path.join("output", "cleaned_reviews"), help="output folder for the CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input file not found: {args.input}\nDownload it from Kaggle and place it in the data/ folder.")

    spark = create_spark_session()

    # 1. LOAD -----------------------------------------------------------------------------
    banner("1. LOAD THE DATASET")
    raw_df = load_reviews(spark, args.input)
    raw_count = raw_df.count()
    print(f"Loaded '{args.input}' -> {raw_count:,} rows")

    # 2. EXPLORE --------------------------------------------------------------------------
    banner("2. EXPLORE THE DATA")
    print("Schema:")
    raw_df.printSchema()
    print("First 5 rows (reviews truncated to 80 characters):")
    raw_df.show(5, truncate=80)

    # 3. CLEAN ----------------------------------------------------------------------------
    banner("3. CLEAN THE DATA")
    null_reviews = raw_df.filter(F.col("review").isNull()).count()
    empty_reviews = raw_df.filter(F.col("review").isNotNull() & (F.trim(F.col("review")) == "")).count()
    clean_df = clean_reviews(raw_df).cache()   # reused by every step below
    clean_count = clean_df.count()
    print(f"Null reviews removed  : {null_reviews:,}")
    print(f"Empty reviews removed : {empty_reviews:,}")
    print(f"Rows after cleaning   : {clean_count:,}  (removed {raw_count - clean_count:,})")

    # 4. ANALYZE --------------------------------------------------------------------------
    banner("4. ANALYZE THE DATA")
    print(f"Total number of reviews: {clean_count:,}\n")
    print("Reviews per sentiment label:")
    sentiment_counts = (
        clean_df.groupBy("sentiment")
                .agg(F.count("*").alias("num_reviews"))
                .withColumn("percent", F.round(F.col("num_reviews") * 100.0 / clean_count, 2))
                .orderBy(F.desc("num_reviews"))
    )
    sentiment_counts.show()
    counts = {row["sentiment"]: row["num_reviews"] for row in sentiment_counts.collect()}
    print(f"Positive reviews: {counts.get('positive', 0):,}")
    print(f"Negative reviews: {counts.get('negative', 0):,}")

    # 5. TRANSFORM ------------------------------------------------------------------------
    banner("5. TRANSFORM THE DATA")
    transformed_df = add_review_length(clean_df)
    print("New column 'review_length' added:")
    transformed_df.select("review", "sentiment", "review_length").show(5, truncate=60)

    long_reviews_df = transformed_df.filter(F.col("review_length") > LONG_REVIEW_THRESHOLD)
    long_count = long_reviews_df.count()
    print(f"Reviews longer than {LONG_REVIEW_THRESHOLD} characters: {long_count:,} "
          f"({long_count * 100.0 / clean_count:.1f}% of all reviews)")
    print("Sample of long reviews:")
    long_reviews_df.select("review", "sentiment", "review_length").show(10, truncate=60)

    # Extra insight: are positive or negative reviews longer?
    print("Review length statistics by sentiment:")
    (
        transformed_df.groupBy("sentiment")
        .agg(
            F.round(F.avg("review_length"), 1).alias("avg_length"),
            F.min("review_length").alias("min_length"),
            F.max("review_length").alias("max_length"),
            F.sum(F.when(F.col("review_length") > LONG_REVIEW_THRESHOLD, 1).otherwise(0)).alias("reviews_over_500"),
        )
        .orderBy("sentiment")
        .show()
    )

    # 6. SAVE -----------------------------------------------------------------------------
    banner("6. SAVE YOUR RESULTS")
    saved_to = save_csv(transformed_df, args.output)
    print(f"Cleaned + transformed DataFrame ({clean_count:,} rows, columns: "
          f"{', '.join(transformed_df.columns)}) saved to: {saved_to}")

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
