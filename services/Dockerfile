FROM python:3.11-slim

LABEL author=${AUTHOR}

# Назначаем рабочую директорию сразу
WORKDIR /app

# Копируем всё содержимое текущей папки проекта внутрь /app
COPY . .

# Устанавливаем зависимости
RUN pip3 install --no-cache-dir -r requirements.txt

# Открываем порт
EXPOSE ${APP_PORT}

# Если вам нужно монтировать модели снаружи (опционально), указывать правильный путь:
# VOLUME /app/models

CMD ["uvicorn", "ml_service.app_fast_api:app", "--host", "0.0.0.0", "--port", "1702"]
