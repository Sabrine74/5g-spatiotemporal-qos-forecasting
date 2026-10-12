FROM python:3.12-slim
# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY realtime/requirements.txt /app/realtime/requirements.txt
RUN pip install --no-cache-dir -r realtime/requirements.txt
COPY src /app/src
COPY realtime /app/realtime
COPY models /app/models
ENV PYTHONPATH=/app/src
