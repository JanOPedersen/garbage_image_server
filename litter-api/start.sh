#!/usr/bin/env bash
set -euo pipefail

echo "Starting Render/Docker bootstrap..."

MODEL_ARTIFACTS_DIR="${MODEL_ARTIFACTS_DIR:-/tmp/model_artifacts}"
MODEL_MANIFEST_DIR="${MODEL_MANIFEST_DIR:-models/manifests}"
DEFAULT_MODEL_ID="${DEFAULT_MODEL_ID:-cigarette-butt-v1}"
MODEL_URL="${MODEL_URL:-}"

MANIFEST_PATH="${MODEL_MANIFEST_DIR%/}/${DEFAULT_MODEL_ID}.yaml"
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

if [[ "${WEIGHTS_PATH_RAW}" = /* ]]; then
  MODEL_PATH="${WEIGHTS_PATH_RAW}"
else
  MODEL_PATH="${MODEL_ARTIFACTS_DIR%/}/${WEIGHTS_PATH_RAW#./}"
fi

mkdir -p "$(dirname "${MODEL_PATH}")"

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
export MODEL_MANIFEST_DIR
export DEFAULT_MODEL_ID
export MODEL_PATH

echo "Starting Gunicorn on 0.0.0.0:${PORT:-5000}"
exec gunicorn "run:app" -c gunicorn.conf.py