from pyspark.sql import DataFrame
from pyspark.ml.feature import Bucketizer
import mlflow


class DataBinningError(Exception):
    pass

binning_config = {
    "age": {
        "splits": [-float("inf"), 25, 35, 45, 55, float("inf")],
        "output_col": "age_manual_binned"
    },
    "purchase_value": {
        "splits": [-float("inf"), 20, 40, 60, 80, float("inf")],
        "output_col": "purchase_value_manual_binned"
    }
}

def data_binning(
    df: DataFrame,  
):
    """
    Applies manual binning using PySpark Bucketizer and logs to MLflow.

    Parameters:
    - df: input DataFrame
    - binning_config: dict like:
        {
            "age": {
                "splits": [-inf, 25, 35, 45, 55, inf],
                "output_col": "age_binned"
            },
            "purchase_value": {
                "splits": [-inf, 20, 40, 60, 80, inf],
                "output_col": "purchase_value_binned"
            }
        }
    """


    mlflow.set_tag("step", "data_binning")

    summary = {}

    for col, config in binning_config.items():

        splits = config["splits"]
        output_col = config["output_col"]

        # -------------------------
        # 1. Validate splits
        # -------------------------
        if len(splits) < 2:
            raise DataBinningError(f"Invalid splits for {col}")

        # -------------------------
        # 2. Apply Bucketizer
        # -------------------------
        bucketizer = Bucketizer(
            splits=splits,
            inputCol=col,
            outputCol=output_col
        )

        df = bucketizer.transform(df)

        # -------------------------
        # 3. Compute bin distribution
        # -------------------------
        bin_counts = (
            df.groupBy(output_col)
            .count()
            .orderBy(output_col)
            .collect()
        )

        distribution = {str(row[0]): row[1] for row in bin_counts}

        summary[col] = {
            "splits": splits,
            "distribution": distribution
        }

        # -------------------------
        # 4. MLflow logging
        # -------------------------
        mlflow.log_param(f"{col}_splits", str(splits))
        mlflow.log_dict(distribution, f"{col}_bin_distribution.json")

        # -------------------------
        # 5. Log full summary
        # -------------------------
        mlflow.log_dict(summary, "binning_summary.json")

        return df