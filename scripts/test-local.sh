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
LOCAL_PORT="${LOCAL_PORT:-30000}"

if [[ -z "${SGLANG_API_KEY:-}" ]]; then
  read -r -s -p "SGLang API key: " SGLANG_API_KEY
  echo
fi

"${KUBECTL_CMD[@]}" -n llm port-forward service/sglang "${LOCAL_PORT}:30000" >/tmp/sglang-port-forward.log 2>&1 &
PORT_FORWARD_PID=$!
trap 'kill "${PORT_FORWARD_PID}" 2>/dev/null || true' EXIT

for _ in {1..30}; do
  if curl -fsS "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null; then
    break
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/chat/completions" \
  -H "Authorization: Bearer ${SGLANG_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Reply with exactly: deployment works"}],
    "max_tokens": 32,
    "temperature": 0
  }'
echo
