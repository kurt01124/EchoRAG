@echo off
start uvicorn app:app --host 0.0.0.0 --port 8000 --reload

cd frontend
start python -m http.server 3000