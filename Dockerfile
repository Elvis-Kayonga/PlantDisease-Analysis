FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

EXPOSE 8000

RUN cd frontend && npm install && npm run build

CMD ["python", "main.py"]
