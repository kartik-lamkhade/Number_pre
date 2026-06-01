FROM python:3.11.9-slim
WORKDIR /app

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY . .

RUN apt-get update && apt-get install -y dos2unix && rm -rf /var/lib/apt/lists/*
RUN dos2unix start.sh && chmod +x start.sh

EXPOSE 8000 8501

CMD ["./start.sh"]
