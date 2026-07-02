from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import mlflow


class OutlierHandlingError(Exception):
    pass


def handle_outliers_iqr(
    df: DataFrame,
    strategy: str = "clip",  # "clip" | "remove" | "keep"
    multiplier: float = 1.5, 
):
    """
    Detects and handles outliers using IQR method.
    Logs results to MLflow.
    """

 
    summary = {}
    numeric_cols = ["purchase_value", "age"]

    for col in numeric_cols:

        # -------------------------
        # 1. Compute quantiles
        # -------------------------
        quantiles = df.approxQuantile(col, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # -------------------------
        # 2. Detect outliers
        # -------------------------
        outlier_condition = (
            (F.col(col) < lower_bound) |
            (F.col(col) > upper_bound)
        )

        outlier_count = df.filter(outlier_condition).count()
        total_count = df.count()
        outlier_pct = outlier_count / total_count

        summary[col] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers": outlier_count,
            "outlier_pct": outlier_pct
        }

        # -------------------------
        # 3. MLflow logging
        # -------------------------
        mlflow.log_metric(f"{col}_outliers", outlier_count)
        mlflow.log_metric(f"{col}_outlier_pct", outlier_pct)
        mlflow.log_param(f"{col}_lower_bound", lower_bound)
        mlflow.log_param(f"{col}_upper_bound", upper_bound)

        # -------------------------
        # 4. Handle outliers
        # -------------------------

        if strategy == "clip":
            df = df.withColumn(
                col,
                F.when(F.col(col) < lower_bound, lower_bound)
                    .when(F.col(col) > upper_bound, upper_bound)
                    .otherwise(F.col(col))
            )

        elif strategy == "remove":
            df = df.filter(~outlier_condition)

        elif strategy == "keep":
            pass  # only logging

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # -------------------------
    # 5. Log full report
    # -------------------------
    mlflow.log_dict(summary, "outlier_report.json")

    mlflow.set_tag("outlier_strategy", strategy)

    return df