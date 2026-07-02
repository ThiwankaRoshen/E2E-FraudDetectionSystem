import os
import logging
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
import mlflow
from mlflow.tracking import MlflowClient
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def read_csv_from_s3(s3_uri: str) -> pd.DataFrame:
    """
    Read a CSV file from an S3 URI (s3://bucket/path/to/file.csv).
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("Input URI must start with s3://")
    # Parse bucket and key
    path_parts = s3_uri[5:].split('/', 1)
    bucket = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ''
    s3 = boto3.client('s3')
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])
    except ClientError as e:
        logger.error(f"Failed to read {s3_uri}: {e}")
        raise


def save_csv_to_s3(df: pd.DataFrame, s3_uri: str, index: bool = False):
    """
    Save a DataFrame as CSV to an S3 URI.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("Output URI must start with s3://")
    path_parts = s3_uri[5:].split('/', 1)
    bucket = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=index)
        tmp_path = tmp.name
    try:
        s3 = boto3.client('s3')
        s3.upload_file(tmp_path, bucket, key)
        logger.info(f"Saved predictions to {s3_uri}")
    finally:
        os.unlink(tmp_path)


def preprocess_data(df: pd.DataFrame, expected_features: list = None) -> pd.DataFrame:
    """
    Validate and preprocess the input DataFrame.
    Steps (customize as needed):
      - Drop rows with all nulls or critical missing columns.
      - Fill numeric missing values with median, categorical with mode.
      - Ensure feature set matches expected_features (order, existence).
    Returns preprocessed DataFrame ready for inference.
    """
    # Basic validation: require at least one row
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Example: drop rows where all values are missing
    df = df.dropna(how='all')
    
    # Separate features (we assume no target column present; if present, drop it)
    # Identify potential target column names (customize as needed)
    target_candidates = ['target', 'label', 'y', 'class']
    for col in target_candidates:
        if col in df.columns:
            logger.warning(f"Dropping target column '{col}' as it should not be in inference data")
            df = df.drop(columns=[col])
    
    # Handle missing values (example: numeric -> median, categorical -> mode)
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Filled missing values in numeric column '{col}' with median={median_val}")
    
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "MISSING"
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"Filled missing values in categorical column '{col}' with mode='{mode_val}'")
    
    # If expected_features is provided, reorder columns and check existence
    if expected_features:
        missing = set(expected_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected features in input data: {missing}")
        extra = set(df.columns) - set(expected_features)
        if extra:
            logger.warning(f"Extra columns in input data that will be ignored: {extra}")
        df = df[expected_features]
    
    return df


def load_model(model_uri: str = None, model_name: str = None, stage: str = "Production"):
    """
    Load an MLflow model either from a direct model URI or from the Model Registry.
    - model_uri: e.g., "runs:/<run_id>/model" or "models:/<model_name>/<stage/version>"
    - model_name: if provided along with stage, constructs models:/<model_name>/<stage>
    Returns a pyfunc model.
    """
    if model_uri is None and model_name is None:
        raise ValueError("Either model_uri or model_name must be provided")
    
    if model_uri is None:
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model from registry: {model_uri}")
    else:
        logger.info(f"Loading model from URI: {model_uri}")
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_uri}: {e}")
        raise


def inference_pipeline(input_csv_uri: str,
                       output_csv_uri: str,
                       model_uri: str = None,
                       model_name: str = None,
                       model_stage: str = "Production",
                       expected_features: list = None):
    """
    End-to-end inference pipeline:
      1. Read CSV from S3.
      2. Preprocess/validate.
      3. Load model (from URI or registry).
      4. Generate predictions.
      5. Append predictions to original data and save to output S3 URI.
    """
    logger.info("Starting inference pipeline")
    logger.info(f"Input: {input_csv_uri}")
    logger.info(f"Output: {output_csv_uri}")
    
    # Step 1: Load data
    df_raw = read_csv_from_s3(input_csv_uri)
    logger.info(f"Loaded {len(df_raw)} rows from {input_csv_uri}")
    
    # Step 2: Preprocess
    df_processed = preprocess_data(df_raw, expected_features=expected_features)
    logger.info(f"After preprocessing: {len(df_processed)} rows, features: {list(df_processed.columns)}")
    
    # Step 3: Load model
    model = load_model(model_uri=model_uri, model_name=model_name, stage=model_stage)
    
    # Step 4: Predict
    # Some models require data as numpy array; pyfunc predict accepts DataFrame.
    predictions = model.predict(df_processed)
    # If predictions are probabilities or multi-class, handle accordingly.
    # For binary classification, we often want both class and probability.
    # Here we add a column 'prediction' (class label).
    # If the model outputs probabilities (shape (n,2)), extract argmax.
    if hasattr(predictions, 'shape') and len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Multi-class probabilities -> class with highest prob
        pred_class = np.argmax(predictions, axis=1)
    else:
        pred_class = predictions
    
    # For binary classification, also log probability of positive class if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(df_processed)
        if proba.shape[1] == 2:
            df_raw['prediction_probability'] = proba[:, 1]
    
    df_raw['prediction'] = pred_class
    logger.info(f"Predictions generated. Unique values: {np.unique(pred_class)}")
    
    # Step 5: Save to S3
    save_csv_to_s3(df_raw, output_csv_uri, index=False)
    logger.info("Inference pipeline completed successfully")


if __name__ == "__main__":
    # Example usage (uncomment and modify as needed)
    # inference_pipeline(
    #     input_csv_uri="s3://my-bucket/inference_data.csv",
    #     output_csv_uri="s3://my-bucket/predictions.csv",
    #     model_name="my_xgboost_model",
    #     model_stage="Production",
    #     expected_features=["feature1", "feature2", "feature3"]
    # )
    pass