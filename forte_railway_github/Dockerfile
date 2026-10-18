FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-railway.txt /app/
RUN pip install --no-cache-dir -r requirements-railway.txt
COPY backend /app/backend
COPY web /app/web
COPY config /app/config
# /data is a Railway persistent Volume (configured in dashboard).
RUN mkdir -p /data && chmod 700 /data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
