FROM python:3.12-slim

WORKDIR /app

# Install git and essential tools for the agent's Bash tool
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir \
    openai>=2.15.0 \
    fastapi>=0.110.0 \
    uvicorn>=0.28.0 \
    pydantic>=2.0.0

# Copy application files
COPY . /app/

# Railway automatically sets PORT
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn web:app --host 0.0.0.0 --port ${PORT:-8080}"]
