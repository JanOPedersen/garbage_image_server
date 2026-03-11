FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m pip install --no-cache-dir --no-build-isolation \
    "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@v0.6"

COPY . .

RUN chmod +x /app/start.sh

EXPOSE 5000

CMD ["/app/start.sh"]