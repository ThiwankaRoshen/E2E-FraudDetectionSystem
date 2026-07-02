
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql import functions as F
import mlflow


def time_based_train_test_split(df, time_col, label_col, split_ratio=0.8):
    """
    Time-based split for PySpark DataFrame + class distribution stats.

    Returns:
        train_df, test_df, stats_dict
    """

    # 1. sort by time (critical for leakage-free split)
    df_sorted = df.orderBy(time_col)

    # 2. compute split index
    total_count = df_sorted.count()
    split_idx = int(total_count * split_ratio)

    # 3. add row index (safe sequential split)
    window_spec = (
        df_sorted
        .withColumn("row_idx", F.monotonically_increasing_id())
        .orderBy(time_col)
    )

    # IMPORTANT: monotonically_increasing_id is NOT strictly sequential,
    # so we create proper ordering using row_number

    w = Window.orderBy(time_col)

    df_indexed = df_sorted.withColumn("row_idx", row_number().over(w) - 1)

    train_df = df_indexed.filter(F.col("row_idx") < split_idx).drop("row_idx")
    test_df = df_indexed.filter(F.col("row_idx") >= split_idx).drop("row_idx")

    # 4. class distribution
    train_dist = (
        train_df.groupBy(label_col)
        .count()
        .withColumn("split", F.lit("train"))
    )

    test_dist = (
        test_df.groupBy(label_col)
        .count()
        .withColumn("split", F.lit("test"))
    )

    dist_df = train_dist.unionByName(test_dist)

    stats = {
        "train_size": train_df.count(),
        "test_size": test_df.count()
    }
    
    mlflow.log_metrics(stats)
    pdf = dist_df.toPandas()

    for _, row in pdf.iterrows():
        mlflow.log_metric(
            f"{row['split']}_class_{row['label']}_count",
            row["count"]
        )
    return train_df, test_df


FEATURE_COLS = [
    "time_velocity",
    "ip_user_share_count",
    "device_user_share_count",
    "day",
    "hour",
    "source_fr_enc",
    "browser_fr_enc",
    "sex_F",
    "sex_M",
    "age_manual_binned",
    "purchase_value_manual_binned"
]

LABEL_COL = "class"

def extract_xy(df):
    """
    Returns X, y as Spark DataFrames (model-ready)
    """

    X = df.select(*FEATURE_COLS)
    y = df.select(F.col(LABEL_COL))

    return X, y

def split_data(train_df, test_df):
    mlflow.log_param("features", FEATURE_COLS)
    mlflow.log_param("label", LABEL_COL)  
    X_train, Y_train = extract_xy(train_df)
    X_test, Y_test = extract_xy(test_df)

    return X_train, X_test, Y_train, Y_test

