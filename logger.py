import logging
import os
import boto3

def get_logger(name, s3_bucket="mlops-dagshub-dvc-mlflow-experimenttracking", s3_prefix="logs"):
    """
    Create a logger that writes to local file and can upload to S3.
    Each script gets its own log file under logs/<script>.log locally,
    and uploads to s3://<bucket>/<prefix>/<script>.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    # Avoid duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def upload_to_s3():
        try:
            s3 = boto3.client("s3")
            s3_key = f"{s3_prefix}/{name}.log"
            s3.upload_file(log_path, s3_bucket, s3_key)
            print(f"✅ Uploaded {log_path} to s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"❌ Failed to upload log to S3: {e}")

    # Attach helper method to logger
    logger.upload_to_s3 = upload_to_s3
    return logger