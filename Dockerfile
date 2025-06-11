FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    build-essential \
    libpq-dev \
    netcat-traditional \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (for Prisma)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install prisma

# Copy prisma schema for client generation
COPY prisma ./prisma
RUN prisma generate

# Copy start script
COPY start.sh .
RUN chmod +x start.sh && sed -i 's/\r//' start.sh

# Environment setup
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONMALLOC=malloc
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["./start.sh"]