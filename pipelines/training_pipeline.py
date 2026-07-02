import os
import sys
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tempfile

logger = logging.getLogger(__name__)


def read_df_csv(artifact_uri: str) -> pd.DataFrame:
    """
    Download a CSV artifact from MLflow and read it as a DataFrame.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri, dst_path=tmpdir)
        return pd.read_csv(local_path)


def training_pipeline():
    # Start an MLflow run
    with mlflow.start_run(run_name="xgboost_training"):
        # ------------------------------------------------------------------
        # 1. Load processed data from previous pipeline (logged as CSV artifacts)
        # ------------------------------------------------------------------
        run_artifact_uri = settings.train_artifacts_uri
        base_uri = run_artifact_uri.rstrip('/')
        X_train_uri = f"{base_uri}/train/X_train.csv"
        Y_train_uri = f"{base_uri}/train/Y_train.csv"
        X_test_uri  = f"{base_uri}/test/X_test.csv"
        Y_test_uri  = f"{base_uri}/test/Y_test.csv"

        X_train = read_df_csv(X_train_uri)
        Y_train = read_df_csv(Y_train_uri)
        X_test  = read_df_csv(X_test_uri)
        Y_test  = read_df_csv(Y_test_uri)
 
        y_train = Y_train.values.ravel()
        y_test  = Y_test.values.ravel()

        # ------------------------------------------------------------------
        # 2. Define hyperparameters and train XGBoost model
        # ------------------------------------------------------------------
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',      # for classification
            'use_label_encoder': False     # silence warning in newer xgboost
        }

        # Log hyperparameters to MLflow
        mlflow.log_params(hyperparams)

        # Train model
        model = xgb.XGBClassifier(**hyperparams)
        model.fit(X_train, y_train, 
                  verbose=False)

        # ------------------------------------------------------------------
        # 3. Evaluate and log metrics
        # ------------------------------------------------------------------
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of positive class

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        mlflow.log_metrics(metrics)

        # ------------------------------------------------------------------
        # 4. Log the model
        # ------------------------------------------------------------------
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
 

if __name__ == "__main__":
    training_pipeline()