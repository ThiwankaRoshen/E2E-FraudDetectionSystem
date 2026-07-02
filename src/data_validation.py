from pyspark.sql import DataFrame
import logging
import mlflow
from pyspark.sql import functions as F


logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    pass

EXPECTED_SCHEMA = {
    "user_id": "bigint",
    "signup_time": "string",
    "purchase_time": "string",
    "purchase_value": "bigint",
    "device_id": "string",
    "source": "string",
    "browser": "string",
    "sex": "string",
    "age": "bigint",
    "ip_address": "double",
    "class": "bigint"
}


def validate_schema_data_pipeline(df: DataFrame) -> None:
    """
    Validate dataframe schema and log results to MLflow.

    Raises:
        ValueError: If schema validation fails.
    """

    errors = []

    actual_columns = df.columns
    expected_columns = list(EXPECTED_SCHEMA.keys())

    # Column count
    if len(actual_columns) != len(expected_columns):
        errors.append(
            f"Expected {len(expected_columns)} columns but found {len(actual_columns)}"
        )

    # Column names and order
    if actual_columns != expected_columns:
        missing = set(expected_columns) - set(actual_columns)
        extra = set(actual_columns) - set(expected_columns)

        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")

        if extra:
            errors.append(f"Unexpected columns: {sorted(extra)}")

        if (
            not missing
            and not extra
            and actual_columns != expected_columns
        ):
            errors.append(
                f"Column order mismatch.\n"
                f"Expected: {expected_columns}\n"
                f"Found:    {actual_columns}"
            )

    # Data types
    actual_types = dict(df.dtypes)

    for col, expected_type in EXPECTED_SCHEMA.items():
        if col in actual_types:
            actual_type = actual_types[col]

            if actual_type != expected_type:
                errors.append(
                    f"Column '{col}' type mismatch. "
                    f"Expected '{expected_type}', found '{actual_type}'"
                )

    # ---------- MLflow Logging ----------
    mlflow.log_metric("schema_expected_columns", len(expected_columns))
    mlflow.log_metric("schema_actual_columns", len(actual_columns))
    mlflow.log_metric("schema_error_count", len(errors))

    if errors:
        actual_schema = "\n".join(
            f"{name}: {dtype}"
            for name, dtype in df.dtypes
        )

        mlflow.log_text(
            actual_schema,
            "validation/actual_schema.txt"
        )
        
        expected_schema = "\n".join(
            f"{name}: {dtype}"
            for name, dtype in EXPECTED_SCHEMA.items()
        )

        mlflow.log_text(
            expected_schema,
            "validation/expected_schema.txt"
        )
        
        mlflow.log_param("schema_validation_passed", False)

        error_text = "\n".join(errors)

        # Useful for searching runs
        mlflow.set_tag("schema_validation_status", "failed")

        # Store full error message
        mlflow.log_text(
            error_text,
            "validation/schema_errors.txt"
        )

        raise DataValidationError(
            "Schema validation failed:\n"
            + "\n".join(f"- {err}" for err in errors)
        )

    mlflow.log_param("schema_validation_passed", True)
    mlflow.set_tag("schema_validation_status", "passed")
    

    logger.info("✅ Schema validation passed")
    

def validate_data_contract(df: DataFrame):
    """
    Validates dataset against known training data schema + value ranges.
    Logs violations to MLflow and fails if any rule is broken.
    """


    violations = {}

    # =========================
    # 1. NUMERIC VALIDATION
    # =========================

    numeric_rules = {
        "purchase_value": (9, 154),
        "age": (18, 76)
    }

    for col, (min_v, max_v) in numeric_rules.items():
        out_of_range_count = df.filter(
            (F.col(col) < min_v) | (F.col(col) > max_v)
        ).count()

        violations[col] = out_of_range_count
        mlflow.log_metric(f"{col}_out_of_range", out_of_range_count)

    # =========================
    # 2. CATEGORICAL VALIDATION
    # =========================

    categorical_rules = {
        "source": {"SEO", "Ads", "Direct"},
        "browser": {"Chrome", "Opera", "Safari", "IE", "FireFox"},
        "sex": {"M", "F"},
        "class": {0, 1}
    }

    for col, allowed_values in categorical_rules.items():
        invalid_count = df.filter(
            ~F.col(col).isin(list(allowed_values))
        ).count()

        violations[col] = invalid_count
        mlflow.log_metric(f"{col}_invalid_values", invalid_count)

    # =========================
    # 3. TOTAL VIOLATIONS
    # =========================

    total_violations = sum(violations.values())
    mlflow.log_metric("total_violations", total_violations)

    # Log full report
    mlflow.log_dict(violations, "data_contract_violations.json")

    # =========================
    # 4. FAIL FAST RULE
    # =========================

    if total_violations > 0:
        mlflow.set_tag("validation_status", "failed")

        raise DataValidationError(
            f"Data contract validation failed. Violations: {violations}"
        )

    mlflow.set_tag("validation_status", "passed")

    return True

def validate_no_nulls(df: DataFrame):
    """
    Validates that dataframe has no null values.
    Logs null stats to MLflow and raises error if any nulls exist.
    """
    # ---- 1. Compute null counts per column ----
    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ])

    null_dict = null_counts.collect()[0].asDict()

    # ---- 2. Compute total nulls ----
    total_nulls = sum(null_dict.values())

    # ---- 3. Log to MLflow ----
    for col, cnt in null_dict.items():
        mlflow.log_metric(f"nulls_{col}", cnt)

    mlflow.log_metric("total_nulls", total_nulls)

    # ---- 4. Validation rule ----
    if total_nulls > 0:
        message = (
            f"Data validation failed: found {total_nulls} null values. "
            f"Per column: {null_dict}"
        )
        mlflow.set_tag("validation_status", "failed")
        mlflow.log_param("null_columns", str(null_dict))

        raise DataValidationError(message)

    mlflow.set_tag("validation_status", "passed")

    return True