import numpy as np
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json
import os
import yaml
import logging
from logger import get_logger

# Initialize logger for this script
logger = get_logger(
    "model_evaluation",
    s3_bucket="mlops-dagshub-dvc-mlflow-experimenttracking",
    s3_prefix="logs/model_evaluation"
)

# --- Initialize DagsHub MLflow tracking ---
dagshub.init(
    repo_owner="krishnauppalapatiaws",
    repo_name="MLOpsProductionModel",
    mlflow=True
)
mlflow.set_experiment("tweet_emotions_experiment")

# --- Load parameters from params.yaml ---
repo_root = os.path.dirname(os.path.dirname(__file__))
params_path = os.path.join(repo_root, "params.yaml")

with open(params_path) as f:
    params = yaml.safe_load(f)

gb_params = params["gb"]
N_ESTIMATORS = gb_params.get("n_estimators", 100)
LEARNING_RATE = gb_params.get("learning_rate", 0.1)
MAX_DEPTH = gb_params.get("max_depth", 3)

try:
    # --- Load train/test features locally ---
    train_data = pd.read_csv("artifacts/features/tweet_emotions_train_features.csv")
    test_data = pd.read_csv("artifacts/features/tweet_emotions_test_features.csv")

    X_train = train_data.drop(columns=["sentiment"]).values
    y_train = train_data["sentiment"].values

    X_test = test_data.drop(columns=["sentiment"]).values
    y_test = test_data["sentiment"].values

    # --- Train model ---
    clf = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=42
    )
    clf.fit(X_train, y_train)
    logger.info("Gradient Boosting model trained successfully.")

    # --- Train Metrics ---
    y_train_pred = clf.predict(X_train)
    y_train_proba = clf.predict_proba(X_train)[:, 1]

    train_metrics = {
        "accuracy": accuracy_score(y_train, y_train_pred),
        "precision": precision_score(y_train, y_train_pred, average="weighted"),
        "recall": recall_score(y_train, y_train_pred, average="weighted"),
        "auc": roc_auc_score(y_train, y_train_proba)
    }

    # --- Eval Metrics ---
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    eval_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, average="weighted"),
        "recall": recall_score(y_test, y_test_pred, average="weighted"),
        "auc": roc_auc_score(y_test, y_test_proba)
    }

    logger.info(f"Train Metrics: {train_metrics}")
    logger.info(f"Eval Metrics: {eval_metrics}")

    # --- Save metrics locally ---
    os.makedirs("artifacts/metrics", exist_ok=True)
    train_path = "artifacts/metrics/gb_train_metrics.json"
    eval_path = "artifacts/metrics/gb_eval_metrics.json"

    with open(train_path, "w") as f:
        json.dump(train_metrics, f)
    with open(eval_path, "w") as f:
        json.dump(eval_metrics, f)

    logger.info("Saved Gradient Boosting metrics locally.")

    # --- Save model locally ---
    os.makedirs("artifacts/models", exist_ok=True)
    model_out = "artifacts/models/gb_model.pkl"
    with open(model_out, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Gradient Boosting model saved locally at {model_out}")

    # --- Log to MLflow/Dagshub ---
    with mlflow.start_run(run_name="GradientBoosting"):

        # Log metrics
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.items()})

        # Log parameters
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("model_type", "GradientBoostingClassifier")

        # Log model
        mlflow.sklearn.log_model(clf, "gb_model")

        # Log individual metric files
        mlflow.log_artifact(train_path)
        mlflow.log_artifact(eval_path)

        # ⭐ CRITICAL FIX ⭐
        # Ensures Dagshub marks run as SUCCESS ✓
        mlflow.log_artifacts("artifacts/")

        # Register model in MLflow Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/gb_model"
        result = mlflow.register_model(model_uri, "TweetSentimentModel")

        logger.info(f"Model registered as {result.name}, version {result.version}")

except Exception as e:
    logger.error(f"Pipeline failed: {e}")

# Upload logs to S3 and close
logger.upload_to_s3()
logging.shutdown()
