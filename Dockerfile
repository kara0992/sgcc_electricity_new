ARG BUILD_FROM
FROM ${BUILD_FROM}

# 只覆盖被修改的两个脚本，基础镜像里的 Playwright/Chromium 全部复用，构建极快
COPY scripts/data_fetcher.py /app/data_fetcher.py
COPY scripts/main.py /app/main.py

WORKDIR /app
