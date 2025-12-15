FROM python:3.11-slim


# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
ffmpeg libgl1 libsm6 libxext6 && rm -rf /var/lib/apt/lists/*


WORKDIR /app


# Copy requirement first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt


# Copy app
COPY app /app/app
COPY .env /app/.env


ENV PYTHONUNBUFFERED=1


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]