FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --create-home app

COPY pyproject.toml README.md alembic.ini ./
COPY backend_api ./backend_api
COPY migrations ./migrations
RUN pip install --upgrade pip && pip install ".[ml,production]"

RUN mkdir -p /app/data /app/model && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready', timeout=3)"

CMD ["sh", "-c", "alembic upgrade head && uvicorn backend_api.main:app --host 0.0.0.0 --port 8000"]
