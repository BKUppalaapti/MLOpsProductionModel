import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from logger import get_logger

# Initialize logger for this script
logger = get_logger(
    "feature_engineering",
    s3_bucket="mlops-dagshub-dvc-mlflow-experimenttracking",
    s3_prefix="logs/feature_engineering"
)

# --- Feature engineering ---
def vectorize_text(train_df: pd.DataFrame, test_df: pd.DataFrame):
    try:
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(train_df["content"])
        X_test = vectorizer.transform(test_df["content"])

        train_features = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
        test_features = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())

        train_features["sentiment"] = train_df["sentiment"].values
        test_features["sentiment"] = test_df["sentiment"].values

        logger.info(f"Vectorized text: train {train_features.shape}, test {test_features.shape}")
        return train_features, test_features
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

# --- Save features locally (for DVC tracking) ---
def save_features_local(train_features: pd.DataFrame, test_features: pd.DataFrame, base_name: str):
    try:
        os.makedirs("artifacts/features", exist_ok=True)
        train_path = f"artifacts/features/{base_name}_train_features.csv"
        test_path = f"artifacts/features/{base_name}_test_features.csv"

        train_features.to_csv(train_path, index=False)
        test_features.to_csv(test_path, index=False)

        logger.info(f"Saved feature files locally under artifacts/features for {base_name}")
    except Exception as e:
        logger.error(f"Error saving features for {base_name}: {e}")
        raise

# --- Main ---
def main():
    processed_dir = "artifacts/processed"
    if not os.path.exists(processed_dir):
        logger.warning("No processed splits found locally. Run preprocessing first.")
        return

    train_files = [f for f in os.listdir(processed_dir) if f.endswith("_train.csv")]
    test_files = [f for f in os.listdir(processed_dir) if f.endswith("_test.csv")]

    for train_file in train_files:
        try:
            base_name = train_file.replace("_train.csv", "")
            test_file = [f for f in test_files if base_name in f]
            if not test_file:
                logger.warning(f"No matching test file for {train_file}")
                continue

            train_path = os.path.join(processed_dir, train_file)
            test_path = os.path.join(processed_dir, test_file[0])

            logger.info(f"Processing {base_name}...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_features, test_features = vectorize_text(train_df, test_df)
            save_features_local(train_features, test_features, base_name=base_name)
        except Exception as e:
            logger.error(f"Failed to process {train_file}: {e}")

    # Upload log file to S3 at the end
    logger.upload_to_s3()

if __name__ == "__main__":
    main()