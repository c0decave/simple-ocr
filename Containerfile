FROM docker.io/library/python:3.13-slim
# No LM Studio host is baked in: it comes from .env (see .env.example) via --env-file.
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    DATA_DIR=/data
# PDF reports: the DejaVu fonts, and fpdf2 with its dependencies, pinned with hashes, wheels only
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt ./
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r requirements.txt
COPY app/ ./app/
COPY VERSION ./
RUN mkdir /data && chown 65534:65534 /data
USER 65534:65534
EXPOSE 8080
CMD ["python", "-m", "app.server"]
