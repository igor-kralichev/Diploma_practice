# syntax=docker/dockerfile:1

########################################
# 1) Stage: установить зависимости    #
########################################
FROM python:3.11-slim AS builder

WORKDIR /app

# Системные библиотеки для сборки psycopg2 и проч.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc \
      libpq-dev \
      python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Кэшируем установку Python-зависимостей
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --user -r requirements.txt

########################################
# 2) Stage: сам код приложения         #
########################################
FROM python:3.11-slim

WORKDIR /app

# Добавляем локально установленные пакеты в PATH
ENV PATH=/root/.local/bin:$PATH

# Копируем зависимости и исходники
COPY --from=builder /root/.local /root/.local
COPY . .

# Создаём папки под данные, output и логи
RUN mkdir -p /app/logs /app/data /app/output

# Переменные окружения для Flask
ENV FLASK_APP=run.py \
    FLASK_ENV=production

EXPOSE 5000

# По умолчанию запускаем веб-сервер
CMD ["flask", "run", "--host=0.0.0.0"]
