FROM python:3.11-slim

WORKDIR /app

# ── 系统工具 & 时区 & 编码 ──────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 基础工具
    curl ca-certificates \
    vim less \
    # 网络诊断
    iputils-ping telnet net-tools dnsutils \
    # 进程查看
    procps \
    # 时区
    tzdata \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# ── 编码与时区 ───────────────────────────────────────
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai

# ── pip 清华镜像 ─────────────────────────────────────
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8800

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8800", "--reload"]
