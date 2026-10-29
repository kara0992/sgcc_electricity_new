# 基于原作者已构建好的镜像叠加补丁脚本，构建极快（无需重装 Playwright/Chromium）
ARG BUILD_FROM
FROM ${BUILD_FROM}
COPY scripts/data_fetcher.py /app/data_fetcher.py
WORKDIR /app
