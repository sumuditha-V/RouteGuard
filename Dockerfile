# RouteGuard dashboard image (single process; imports the pipeline directly).
FROM python:3.11-slim

WORKDIR /app

# install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code (model_v1 is included; raw data is gitignored — see README deploy notes)
COPY . .

ENV PYTHONPATH=/app/src
EXPOSE 8501

# Streamlit needs to bind 0.0.0.0 to be reachable from outside the container
CMD ["streamlit", "run", "dashboard/Home.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
