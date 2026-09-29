#!/bin/sh

set -e

if [ -f ./.env ]; then
    set -a
    . ./.env
    set +a
fi

export MLFLOW_S3_ENDPOINT_URL="https://storage.yandexcloud.net"

DB_URI="postgresql://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"

mlflow server \
  --registry-store-uri "$DB_URI" \
  --backend-store-uri "$DB_URI" \
  --default-artifact-root "s3://${S3_BUCKET_NAME}" \
  --no-serve-artifacts \
  --host 0.0.0.0 \
  --port 5000