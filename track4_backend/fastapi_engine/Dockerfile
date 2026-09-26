FROM python:3.11-slim

WORKDIR /opt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev g++ && \
    rm -rf /var/lib/apt/lists/*

COPY track4_backend/fastapi_engine/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY track4_backend/fastapi_engine /opt/fastapi_engine
COPY track3_validation_engine /opt/track3_validation_engine

EXPOSE 8000
CMD ["uvicorn", "fastapi_engine.app:app", "--host", "0.0.0.0", "--port", "8000"]
