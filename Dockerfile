# Doctor - 医疗问诊与临床决策AI系统

FROM python:3.11-slim

LABEL maintainer="tswangli-cyber"
LABEL description="Medical Consultation AI System based on HuatuoGPT"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/

# 复制应用代码
COPY app.py .
COPY .env.example .env

# 创建模型缓存目录
RUN mkdir -p /root/.cache/huggingface

# 暴露端口
EXPOSE 8000 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认启动命令
CMD ["python", "app.py"]
