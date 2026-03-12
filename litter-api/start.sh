#!/usr/bin/env bash
set -euo pipefail

echo "Starting Render/Docker bootstrap..."

MODEL_ARTIFACTS_DIR="${MODEL_ARTIFACTS_DIR:-/tmp/model_artifacts}"
DEFAULT_MANIFEST_FILE="${DEFAULT_MANIFEST_FILE:-models/manifests/cigarette-butt-v1.yaml}"
MODEL_URL="${MODEL_URL:-}"

echo "[boot] MODEL_ARTIFACTS_DIR  = ${MODEL_ARTIFACTS_DIR}"
echo "[boot] DEFAULT_MANIFEST_FILE = ${DEFAULT_MANIFEST_FILE}"
echo "[boot] MODEL_URL             = ${MODEL_URL:-(not set)}"

MANIFEST_PATH="${DEFAULT_MANIFEST_FILE}"
if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "Manifest not found: ${MANIFEST_PATH}"
  exit 1
fi

WEIGHTS_PATH_RAW="$(grep -E '^[[:space:]]*weights_path:' "${MANIFEST_PATH}" | head -n1 | sed -E 's/^[[:space:]]*weights_path:[[:space:]]*//')"
if [ -z "${WEIGHTS_PATH_RAW}" ]; then
  echo "weights_path missing in manifest: ${MANIFEST_PATH}"
  exit 1
fi

WEIGHTS_PATH_RAW="${WEIGHTS_PATH_RAW%\"}"
WEIGHTS_PATH_RAW="${WEIGHTS_PATH_RAW#\"}"
WEIGHTS_PATH_RAW="${WEIGHTS_PATH_RAW%\'}"
WEIGHTS_PATH_RAW="${WEIGHTS_PATH_RAW#\'}"
WEIGHTS_FILENAME="$(basename "${WEIGHTS_PATH_RAW}")"

if [[ "${WEIGHTS_PATH_RAW}" = /* ]]; then
  MODEL_PATH="${WEIGHTS_PATH_RAW}"
else
  MODEL_PATH="${MODEL_ARTIFACTS_DIR%/}/${WEIGHTS_PATH_RAW#./}"
fi

mkdir -p "$(dirname "${MODEL_PATH}")"

echo "[boot] WEIGHTS_FILENAME      = ${WEIGHTS_FILENAME}"
echo "[boot] MODEL_PATH            = ${MODEL_PATH}"

if [ -n "${MODEL_URL}" ]; then
  MODEL_DOWNLOAD_URL="${MODEL_URL%/}/${WEIGHTS_FILENAME}"
  echo "[boot] MODEL_DOWNLOAD_URL    = ${MODEL_DOWNLOAD_URL}"

  if [ ! -f "${MODEL_PATH}" ]; then
    echo "Testing asset URL headers..."
    curl -I -L \
      --connect-timeout 20 \
      --max-time 60 \
      "${MODEL_DOWNLOAD_URL}"

    echo "Downloading model from: ${MODEL_DOWNLOAD_URL}"
    curl -fL \
      --connect-timeout 20 \
      --max-time 600 \
      --retry 3 \
      --retry-delay 5 \
      -o "${MODEL_PATH}" \
      "${MODEL_DOWNLOAD_URL}"

    echo "Downloaded model to: ${MODEL_PATH}"
    ls -lh "${MODEL_PATH}"
  else
    echo "Model already exists at: ${MODEL_PATH}"
    ls -lh "${MODEL_PATH}"
  fi
else
  echo "[boot] MODEL_URL not set; assuming model already exists at: ${MODEL_PATH}"
  if [ ! -f "${MODEL_PATH}" ]; then
    echo "[boot] ERROR: model file does not exist and MODEL_URL is not set. Aborting."
    exit 1
  fi
fi

export MODEL_ARTIFACTS_DIR
export DEFAULT_MANIFEST_FILE
export MODEL_PATH

echo "Starting Gunicorn on 0.0.0.0:${PORT:-5000}"
exec gunicorn "run:app" -c gunicorn.conf.py