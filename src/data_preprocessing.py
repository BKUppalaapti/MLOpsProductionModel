import os
import pandas as pd
import re
from logger import get_logger

# Initialize logger for this script
logger = get_logger(
    "data_preprocessing",
    s3_bucket="mlops-dagshub-dvc-mlflow-experimenttracking",
    s3_prefix="logs/data_preprocessing"
)

# --- Text cleaning ---
def clean_text(text: str) -> str:
    try:
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)   # remove punctuation/numbers
        text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise

# --- Preprocess dataframe ---
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if "content" in df.columns:
            df["content"] = df["content"].apply(clean_text)
        df["sentiment"] = df["sentiment"].astype(int)
        logger.info(f"Preprocessed dataframe with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing dataframe: {e}")
        raise

# --- Save processed splits locally (for DVC tracking) ---
def save_processed_local(df: pd.DataFrame, filename: str) -> None:
    try:
        base_name = os.path.splitext(os.path.basename(filename))[0]
        os.makedirs("artifacts/processed", exist_ok=True)
        save_path = f"artifacts/processed/{base_name}.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Saved processed file locally at {save_path}")
    except Exception as e:
        logger.error(f"Error saving processed file {filename}: {e}")
        raise

# --- Main ---
def main():
    raw_dir = "artifacts/raw"
    if not os.path.exists(raw_dir):
        logger.warning("No raw splits found locally. Run ingestion first.")
        return

    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not files:
        logger.warning("No raw CSV files found in artifacts/raw.")
        return

    for file in files:
        try:
            file_path = os.path.join(raw_dir, file)
            logger.info(f"Processing {file_path}...")
            df = pd.read_csv(file_path)
            processed_df = preprocess_df(df)
            save_processed_local(processed_df, filename=file)
        except Exception as e:
            logger.error(f"Failed to process {file}: {e}")

    # Upload log file to S3 at the end
    logger.upload_to_s3()

if __name__ == "__main__":
    main()