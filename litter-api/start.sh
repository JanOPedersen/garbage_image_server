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
    echo "Downloading model from: ${MODEL_URL}"
    curl -L --fail "${MODEL_URL}" -o "${MODEL_PATH}"
    echo "Model downloaded to: ${MODEL_PATH}"
  else
    echo "Model already exists at: ${MODEL_PATH}"
  fi
else
  echo "MODEL_URL not set; assuming model already exists at: ${MODEL_PATH}"
fi

export MODEL_ARTIFACTS_DIR
export MODEL_PATH

echo "MODEL_ARTIFACTS_DIR=${MODEL_ARTIFACTS_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PORT=${PORT:-5000}"

exec gunicorn "run:app" -c gunicorn.conf.py