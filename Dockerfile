FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.api.server:app --host 0.0.0.0 --port 8000 & streamlit run src/ui/dashboard.py --server.port 8501 --server.headless true"]
