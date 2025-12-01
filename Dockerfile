# --------------------------------------------------------
# 1. Base image
# --------------------------------------------------------
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --------------------------------------------------------
# 2. System dependencies
# --------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    build-essential \
    && apt-get clean

# --------------------------------------------------------
# 3. Install Python dependencies
# --------------------------------------------------------

# Copy only requirements first to leverage Docker caching
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# 4. Copy project files
# --------------------------------------------------------
COPY . .

# --------------------------------------------------------
# 5. Avoid interactive DVC/MLflow logins in CI
# --------------------------------------------------------
ENV DAGSHUB_SKIP_HTTP_CLIENT_AUTH=true

# --------------------------------------------------------
# 6. Entrypoint: run full DVC lifecycle
# --------------------------------------------------------
CMD ["bash", "-c", \
    "dvc remote default s3remote && \
     dvc pull -v && \
     dvc repro -v && \
     dvc push -v" ]
