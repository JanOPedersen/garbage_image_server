#!/usr/bin/env bash
set -euo pipefail

echo "Starting Render/Docker bootstrap..."

MODEL_ARTIFACTS_DIR="${MODEL_ARTIFACTS_DIR:-/tmp/model_artifacts}"
MODEL_SUBDIR="${MODEL_SUBDIR:-cigarette-butt}"
MODEL_FILENAME="${MODEL_FILENAME:-model_final.pth}"
MODEL_URL="${MODEL_URL:-}"

MODEL_DIR="${MODEL_ARTIFACTS_DIR}/${MODEL_SUBDIR}"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILENAME}"

mkdir -p "${MODEL_DIR}"

if [ -n "${MODEL_URL}" ]; then
  if [ ! -f "${MODEL_PATH}" ]; then
    echo "Testing asset URL headers..."
    curl -I -L \
      --connect-timeout 20 \
      --max-time 60 \
      "${MODEL_URL}"

    echo "Downloading model from: ${MODEL_URL}"
    curl -fL \
      --connect-timeout 20 \
      --max-time 600 \
      --retry 3 \
      --retry-delay 5 \
      -o "${MODEL_PATH}" \
      "${MODEL_URL}"

    echo "Downloaded model to: ${MODEL_PATH}"
    ls -lh "${MODEL_PATH}"
  else
    echo "Model already exists at: ${MODEL_PATH}"
    ls -lh "${MODEL_PATH}"
  fi
else
  echo "MODEL_URL not set; assuming model already exists at: ${MODEL_PATH}"
fi

export MODEL_ARTIFACTS_DIR
export MODEL_PATH

echo "Starting Gunicorn on 0.0.0.0:${PORT:-5000}"
exec gunicorn "run:app" -c gunicorn.conf.py