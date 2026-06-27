FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY enrichment_agent.py /app/enrichment_agent.py
COPY model_router.py /app/model_router.py

ENTRYPOINT ["python", "/app/enrichment_agent.py"]
