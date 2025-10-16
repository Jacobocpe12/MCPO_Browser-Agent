FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r /requirements.txt

WORKDIR /app
COPY app.py /app/app.py

CMD ["python", "app.py"]
