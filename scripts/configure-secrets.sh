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

"${KUBECTL_CMD[@]}" apply -f deploy/k3s/namespace.yaml

if [[ -z "${SGLANG_API_KEY:-}" ]]; then
  read -r -s -p "SGLang API key (leave blank to generate one): " SGLANG_API_KEY
  echo
fi
if [[ -z "${SGLANG_API_KEY}" ]]; then
  SGLANG_API_KEY="$(openssl rand -hex 32)"
  echo "Generated SGLang API key (save this now): ${SGLANG_API_KEY}"
fi

"${KUBECTL_CMD[@]}" -n llm create secret generic sglang-api \
  --from-literal="api-key=${SGLANG_API_KEY}" \
  --dry-run=client -o yaml | "${KUBECTL_CMD[@]}" apply -f -

echo "SGLang API secret configured. Continue with: ./scripts/deploy.sh"
