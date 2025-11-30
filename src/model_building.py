import os
import pandas as pd
import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
import logging
from logger import get_logger

# Initialize logger for this script
logger = get_logger(
    "model_building",
    s3_bucket="mlops-dagshub-dvc-mlflow-experimenttracking",
    s3_prefix="logs/model_building"
)

# --- Initialize DagsHub + MLflow ---
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

logreg_params = params["logreg"]
MAX_ITER = logreg_params.get("max_iter", 500)
C = logreg_params.get("C", 1.0)

# --- Train model ---
def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    try:
        X_train = train_df.drop(columns=["sentiment"])
        y_train = train_df["sentiment"]

        X_test = test_df.drop(columns=["sentiment"])
        y_test = test_df["sentiment"]

        model = LogisticRegression(max_iter=MAX_ITER, C=C)
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)[:, 1]

        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        train_metrics = {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred),
            "recall": recall_score(y_train, y_train_pred),
            "auc": roc_auc_score(y_train, y_train_prob)
        }

        eval_metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "auc": roc_auc_score(y_test, y_test_prob)
        }

        logger.info(f"Model trained successfully. Train metrics: {train_metrics}, Eval metrics: {eval_metrics}")
        return model, train_metrics, eval_metrics
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

# --- Save model locally ---
def save_model_local(model):
    try:
        os.makedirs("artifacts/models", exist_ok=True)
        model_path = "artifacts/models/logreg_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved Logistic Regression model locally at {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

# --- Save metrics locally ---
def save_metrics(train_metrics, eval_metrics):
    try:
        os.makedirs("artifacts/metrics", exist_ok=True)
        train_path = "artifacts/metrics/logreg_train_metrics.json"
        eval_path = "artifacts/metrics/logreg_eval_metrics.json"

        pd.DataFrame([train_metrics]).to_json(train_path, orient="records", lines=True)
        pd.DataFrame([eval_metrics]).to_json(eval_path, orient="records", lines=True)

        logger.info(f"Saved train metrics at {train_path}")
        logger.info(f"Saved eval metrics at {eval_path}")
        return train_path, eval_path
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

# --- Log to MLflow/DagsHub ---
def log_to_mlflow(model, train_metrics, eval_metrics, train_path, eval_path):
    try:
        with mlflow.start_run(run_name="LogisticRegression"):
            mlflow.log_params({"model_type": "LogisticRegression", "max_iter": MAX_ITER, "C": C})
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.items()})
            mlflow.sklearn.log_model(model, "logreg_model")
            mlflow.log_artifact(train_path)
            mlflow.log_artifact(eval_path)
        logger.info("Logged model and metrics to MLflow/DagsHub")
    except Exception as e:
        logger.error(f"Error logging to MLflow/DagsHub: {e}")
        raise

# --- Main ---
def main():
    features_dir = "artifacts/features"
    if not os.path.exists(features_dir):
        logger.warning("No feature files found locally. Run feature_engineering first.")
        return

    try:
        train_df = pd.read_csv(os.path.join(features_dir, "tweet_emotions_train_features.csv"))
        test_df = pd.read_csv(os.path.join(features_dir, "tweet_emotions_test_features.csv"))

        model, train_metrics, eval_metrics = train_model(train_df, test_df)

        # Save model locally
        model_path = save_model_local(model)

        # Save metrics locally
        train_path, eval_path = save_metrics(train_metrics, eval_metrics)

        # Log to MLflow/DagsHub
        log_to_mlflow(model, train_metrics, eval_metrics, train_path, eval_path)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

    # Upload log file to S3 and flush logs locally
    logger.upload_to_s3()
    logging.shutdown()

if __name__ == "__main__":
    main()