FROM python:3.10-slim

# ========== 1. 安装系统依赖（不常变） ==========
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*
	
# ========== 2. 设置工作目录 ==========
WORKDIR /app

# ========== 3. 先 COPY 低频更新的文件 ==========
COPY requirements.txt ./
COPY run.py ./
COPY api_key.py ./
COPY docker-compose.yml ./
COPY README.md ./
COPY scripts ./scripts
COPY appendix ./appendix

# ========== 4. 安装 Python 依赖（最慢但尽量复用缓存） ==========
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m playwright install --with-deps chromium

# 挂载目录预创建（会被 bind mount 覆盖）
RUN mkdir -p bt_result ml_results models

# ========== 6. COPY 高频更新内容 ==========
COPY app ./app
COPY tests ./tests

# ========== 7. 环境变量和启动 ==========
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
CMD ["python", "run.py"]
