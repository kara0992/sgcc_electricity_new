ARG BUILD_FROM
FROM ${BUILD_FROM}

# 只覆盖被修改的脚本，基础镜像里的 Playwright/Chromium 全部复用，构建极快
COPY scripts/data_fetcher.py /app/data_fetcher.py
COPY scripts/main.py /app/main.py
COPY scripts/sensor_updator.py /app/sensor_updator.py
COPY scripts/click_captcha_solver.py /app/click_captcha_solver.py
COPY scripts/captcha_playwright.py /app/captcha_playwright.py

WORKDIR /app
