FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN mv /app/bt_result /app/bt_result_init \
    && mv /app/ml_results /app/ml_results_init \
	&& mv /app/models /app/models_init \
    && mv /app/output /app/output_init \
    && mv /app/.cache /app/.cache_init \
    && mkdir /app/bt_result /app/ml_results /app/models /app/output /app/.cache

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
CMD ["python", "run.py"]
