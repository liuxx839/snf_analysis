#!/usr/bin/env bash
# 启动 FastAPI + 原生 HTML 前端
#   http://localhost:8000
set -e
cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "启动 SNF 前端, 访问 http://${HOST}:${PORT}"
python3 -m uvicorn web.app:app --host "$HOST" --port "$PORT" --app-dir .
