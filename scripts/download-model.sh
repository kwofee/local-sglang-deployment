#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
# Pin the revision used to build and test this project. Override deliberately
# with MODEL_REVISION when testing a newer upstream revision.
MODEL_REVISION="${MODEL_REVISION:-c1899de289a04d12100db370d81485cdf75e47ca}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${REPO_DIR}/models/Qwen3-0.6B}"

mkdir -p "${MODEL_DIR}"

if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --local-dir "${MODEL_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --local-dir "${MODEL_DIR}"
else
  echo "The Hugging Face CLI is required." >&2
  echo "Install it with: python3 -m pip install --user 'huggingface_hub[cli]'" >&2
  exit 1
fi

for required_file in config.json model.safetensors tokenizer.json; do
  if [[ ! -s "${MODEL_DIR}/${required_file}" ]]; then
    echo "Download is incomplete: ${MODEL_DIR}/${required_file} is missing." >&2
    exit 1
  fi
done

echo "Downloaded ${MODEL_ID}@${MODEL_REVISION} to ${MODEL_DIR}"
echo "Deploy with: MODEL_PATH=${MODEL_DIR} ./scripts/deploy.sh"
