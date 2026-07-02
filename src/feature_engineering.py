from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import mlflow


class FeatureEngineeringError(Exception):
    pass


def feature_engineering(df: DataFrame, ):
    """
    Builds time-based + leakage-safe aggregation features in PySpark.
    Logs feature stats to MLflow.
    """

    mlflow.set_tag("step", "feature_engineering")

    # =========================================================
    # 1. Ensure correct ordering (critical for time-based logic)
    # =========================================================
    df = df.withColumn("signup_time", F.col("signup_time")) \
            .withColumn("purchase_time", F.col("purchase_time"))

    df = df.sort(F.col("signup_time"))

    # =========================================================
    # 2. Time velocity feature
    # =========================================================
    df = df.withColumn(
        "time_velocity",
        (F.col("purchase_time").cast("long") - F.col("signup_time").cast("long"))
    )

    mlflow.log_metric("avg_time_velocity", df.select(F.avg("time_velocity")).collect()[0][0])

    # =========================================================
    # 3. Convert times for window ordering
    # =========================================================
    df = df.withColumn("event_time", F.col("signup_time").cast("long"))

    # =========================================================
    # 4. Leakage-safe cumulative counts (IP / Device)
    # =========================================================

    # Window: all past rows per key ordered by time
    w_ip = Window.partitionBy("ip_address").orderBy("event_time") \
                    .rowsBetween(Window.unboundedPreceding, -1)

    w_device = Window.partitionBy("device_id").orderBy("event_time") \
                        .rowsBetween(Window.unboundedPreceding, -1)

    # Count distinct users seen BEFORE current row
    df = df.withColumn(
        "ip_user_share_count",
        F.size(F.collect_set("user_id").over(w_ip))
    )

    df = df.withColumn(
        "device_user_share_count",
        F.size(F.collect_set("user_id").over(w_device))
    )

    # Fill nulls for first occurrence
    df = df.fillna({
        "ip_user_share_count": 0,
        "device_user_share_count": 0
    })

    # =========================================================
    # 5. Time features (day / hour)
    # =========================================================
    df = df.withColumn("day", F.dayofmonth("purchase_time"))
    df = df.withColumn("hour", F.hour("purchase_time"))

    # =========================================================
    # 6. Feature stats logging
    # =========================================================
    feature_stats = {
        "time_velocity_mean": df.select(F.avg("time_velocity")).collect()[0][0],
        "ip_share_mean": df.select(F.avg("ip_user_share_count")).collect()[0][0],
        "device_share_mean": df.select(F.avg("device_user_share_count")).collect()[0][0]
    }

    mlflow.log_dict(feature_stats, "feature_stats.json")

    # =========================================================
    # 7. Final validation
    # =========================================================
    if df.count() == 0:
        raise FeatureEngineeringError("Feature engineering produced empty dataframe")

    mlflow.set_tag("feature_status", "success")

    return df