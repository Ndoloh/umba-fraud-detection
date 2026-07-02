FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "8000"]