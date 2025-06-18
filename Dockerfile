FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    build-essential \
    libpq-dev \
    netcat-traditional \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка Rust (для Prisma)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Установка рабочей директории
WORKDIR /app

# Копирование зависимостей и установка Python-библиотек
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Установка Prisma Python
RUN pip install prisma

# Копирование остального кода (включая start.sh)
COPY . .

# Исправление прав и переносов строк (важно для Windows пользователей)
RUN chmod +x start.sh && sed -i 's/\r//' start.sh

# Генерация Prisma клиента
RUN prisma generate

# Настройка переменных окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONMALLOC=malloc
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV PYTHONDONTWRITEBYTECODE=1

# Открываем порт
EXPOSE 8000

CMD ["./start.sh"]