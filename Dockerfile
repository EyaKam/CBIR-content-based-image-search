FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Mount your image folder and the index:  -v /path/Objects:/data/Objects -v /path/index:/app/index
ENV CBIR_IMAGE_ROOT=/data/Objects CBIR_INDEX_DIR=/app/index TF_CPP_MIN_LOG_LEVEL=2
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
