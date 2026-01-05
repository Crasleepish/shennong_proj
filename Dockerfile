FROM python:3.10-slim

# ========== 1. 安装系统依赖（不常变） ==========
# - curl/ca-certificates: 安装 uv 用
# - build-essential/gcc/libpq-dev: 你原本需要
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ========== 2. 安装 uv ==========
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ========== 3. 设置工作目录 ==========
WORKDIR /app

# ========== 4. 先 COPY 低频更新文件（用于依赖层缓存） ==========
# 只要依赖不变，这层就能命中缓存
COPY pyproject.toml uv.lock ./
# COPY .python-version ./

# ========== 5. 安装 Python 依赖（用 uv，尽量复用缓存） ==========
# 让 uv 创建并使用项目内 .venv（默认也是项目内 venv）
# --frozen: 严格按 uv.lock 安装，不允许隐式更新 lock（更可复现）
RUN uv sync --frozen

# 让后续命令默认用 venv（避免每条命令都 uv run）
ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"

# ========== 6. Playwright 浏览器与依赖 ==========
# --with-deps 会尝试安装系统依赖（可能触发 apt-get），在 slim 上很常见
RUN python -m playwright install --with-deps chromium

# ========== 7. COPY 其它低频文件（非代码但可能不常变） ==========
COPY run.py ./
COPY api_key.py ./
COPY docker-compose.yml ./
COPY README.md ./
COPY scripts ./scripts
COPY appendix ./appendix

# 挂载目录预创建（会被 bind mount 覆盖）
RUN mkdir -p bt_result ml_results models

# ========== 8. COPY 高频更新内容（代码与测试） ==========
COPY app ./app
COPY tests ./tests

# ========== 9. 环境变量和启动 ==========
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
CMD ["python", "run.py"]
