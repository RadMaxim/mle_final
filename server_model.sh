#!/bin/sh

set -e

# Загружаем переменные из .env
set -a
. ./.env
set +a


# --------------------------------------------------
# S3
# --------------------------------------------------

export MLFLOW_S3_ENDPOINT_URL="https://storage.yandexcloud.net"


# --------------------------------------------------
# PostgreSQL
# --------------------------------------------------

DB_DESTINATION_HOST="rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net"
DB_DESTINATION_PORT="6432"
DB_DESTINATION_NAME="playground_mle_20260525_e31877c13e"

DB_URI="postgresql://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"


# --------------------------------------------------
# MLflow
# --------------------------------------------------

mlflow server \
  --registry-store-uri "$DB_URI" \
  --backend-store-uri "$DB_URI" \
  --default-artifact-root "s3://${S3_BUCKET_NAME}" \
  --no-serve-artifacts \
  --host 127.0.0.1 \
  --port 5000