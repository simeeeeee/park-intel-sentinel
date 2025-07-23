# edge/Dockerfile
FROM python:3.10-slim

WORKDIR /edge

# healthcheck용
RUN apt-get update && apt-get install -y curl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py .

CMD ["python", "run.py"]
