@echo off
title ChurnIQ — API Server
echo ============================================================
echo  ChurnIQ FastAPI Server — Starting on http://127.0.0.1:8000
echo ============================================================
cd /d "%~dp0"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload --log-level info
pause
