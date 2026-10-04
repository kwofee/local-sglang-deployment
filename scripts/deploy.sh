#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${KUBECTL:-}" ]]; then
  read -r -a KUBECTL_CMD <<<"${KUBECTL}"
elif [[ -r "${HOME}/.kube/config" ]]; then
  export KUBECONFIG="${HOME}/.kube/config"
  KUBECTL_CMD=(kubectl)
else
  KUBECTL_CMD=(sudo k3s kubectl)
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/models/Qwen3-0.6B}"

if [[ "${MODEL_PATH}" != /* ]]; then
  echo "MODEL_PATH must be an absolute path: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "No model found at ${MODEL_PATH}." >&2
  echo "Run ./scripts/download-model.sh or set MODEL_PATH to a downloaded model." >&2
  exit 1
fi

"${KUBECTL_CMD[@]}" kustomize deploy/k3s \
  | python3 scripts/render-manifests.py --model-path "${MODEL_PATH}" \
  | "${KUBECTL_CMD[@]}" apply -f -
"${KUBECTL_CMD[@]}" label node --all nvidia.com/gpu.present=true --overwrite
"${KUBECTL_CMD[@]}" -n nvidia-device-plugin rollout status daemonset/nvidia-device-plugin --timeout=300s
"${KUBECTL_CMD[@]}" get node -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
"${KUBECTL_CMD[@]}" -n llm rollout status deployment/sglang --timeout=15m
"${KUBECTL_CMD[@]}" -n llm rollout status deployment/cloudflared --timeout=5m
"${KUBECTL_CMD[@]}" -n llm get pods -o wide

echo
echo "Internal endpoint: http://sglang.llm.svc.cluster.local:30000"
echo "Quick Tunnel URL:"
"${KUBECTL_CMD[@]}" -n llm logs deployment/cloudflared \
  | grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' \
  | tail -n 1
