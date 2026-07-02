from datetime import datetime
from typing import Dict
import numpy as np 
import mlflow
from pyspark.sql import SparkSession
from src.data_splitter import split_data, time_based_train_test_split
from src.data_validation import validate_data_contract, validate_no_nulls, validate_schema_data_pipeline
from src.feature_binning import data_binning
from src.feature_encoding import data_encoder
from src.feature_engineering import feature_engineering
from src.outlier_detection import handle_outliers_iqr
import logging 
logger = logging.getLogger(__name__)
            


def data_pipeline() -> Dict[str, np.ndarray]:
 
        spark = SparkSession.builder \
        .appName("FraudDetectionPipeline") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.path.style.access", "false") \
        .getOrCreate()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA INGESTION STEP")
        logger.info(f"{'='*80}") 
        
        s3_bucket = "fraud-detection-raw-data"                     
        s3_key = "raw/Fraud_Data.csv"                   
        s3_path = f"s3a://{s3_bucket}/{s3_key}"         
        with mlflow.start_run(run_name="data_ingestion"):
            # Read CSV from S3 using Spark
            df = spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(s3_path)
         
         
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA VALIDATION STEP")
        logger.info(f"{'='*80}")
        
        with mlflow.start_run(run_name="data_validation"):
            validate_schema_data_pipeline(df)
            validate_data_contract(df)
            validate_no_nulls(df)
        
        
        logger.info(f"\n{'='*80}")
        logger.info(f"OUTLIER DETECTION STEP")
        logger.info(f"{'='*80}")
        
        with mlflow.start_run(run_name="outlier_detection"):
            df = handle_outliers_iqr(
                                        df,
                                        strategy="clip"
                                    )
        
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE BINNING STEP")
        logger.info(f"{'='*80}")
        
        with mlflow.start_run(run_name="feature_binning"):
            df = data_binning(df)

        
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE ENGINEERING STEP")
        logger.info(f"{'='*80}")
        
        with mlflow.start_run(run_name="feature_engineering"):
            df = feature_engineering(df)
        
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA SPLITTING STEP")
        logger.info(f"{'='*80}")
        with mlflow.start_run(run_name="data_splitting"):
            train_df, test_df = time_based_train_test_split(
                                                                df,
                                                                time_col="signup_time",
                                                                label_col="class",
                                                                split_ratio=0.8
                                                            )
        
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE ENCODING STEP")
        logger.info(f"{'='*80}")
        with mlflow.start_run(run_name="feature_encoding"):
            train_df, test_df = data_encoder(train_df, test_df)
        
        
        with mlflow.start_run(run_name="train-ready-datasets"):
            X_train, X_test, Y_train, Y_test = split_data(train_df, test_df)
            
        logger.info(f"\n{'='*80}")
        logger.info(f"PERSISTING DATASETS TO S3")
        logger.info(f"{'='*80}") 
        mlflow.log_dataframe(X_train.toPandas(), "train/X_train.csv")
        mlflow.log_dataframe(Y_train.toPandas(), "train/Y_train.csv")
        mlflow.log_dataframe(X_test.toPandas(), "test/X_test.csv")
        mlflow.log_dataframe(Y_test.toPandas(), "test/Y_test.csv")
        
        spark.stop()


if __name__ == "__main__":
    data_pipeline()