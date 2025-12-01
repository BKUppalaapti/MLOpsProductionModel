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
# 3. Set working directory
# --------------------------------------------------------
WORKDIR /app

# --------------------------------------------------------
# 4. Install all dependencies from requirements.txt
# --------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# 5. Copy the entire project
# --------------------------------------------------------
COPY . .

# --------------------------------------------------------
# 6. Run DVC pipeline
# --------------------------------------------------------
CMD ["bash", "-c", "dvc repro -v && dvc push -v"]
