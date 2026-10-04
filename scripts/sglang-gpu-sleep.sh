#!/usr/bin/env bash
set -euo pipefail

K3S_BIN="${K3S_BIN:-/usr/local/bin/k3s}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-/usr/bin/nvidia-smi}"
SLEEP_BIN="${SLEEP_BIN:-/usr/bin/sleep}"
NAMESPACE="${SGLANG_NAMESPACE:-llm}"
DEPLOYMENT="${SGLANG_DEPLOYMENT:-sglang}"
PLUGIN_NAMESPACE="${NVIDIA_PLUGIN_NAMESPACE:-nvidia-device-plugin}"
PLUGIN_DAEMONSET="${NVIDIA_PLUGIN_DAEMONSET:-nvidia-device-plugin}"
SGLANG_SELECTOR="${SGLANG_SELECTOR:-app=sglang}"

log() {
  logger -t sglang-gpu-sleep -- "$*" 2>/dev/null || true
  printf '%s\n' "$*"
}

wait_for_k3s() {
  for _ in $(seq 1 60); do
    if "${K3S_BIN}" kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
      return 0
    fi
    "${SLEEP_BIN}" 2
  done
  log "k3s API did not become ready within 120 seconds"
  return 1
}

wait_for_sglang_exit() {
  local pods

  # A Deployment rollout can report success as soon as replicas is set to zero,
  # while its terminating pod still owns CUDA resources. Do not let the NVIDIA
  # driver suspend until the pod has actually disappeared.
  for _ in $(seq 1 120); do
    pods="$("${K3S_BIN}" kubectl -n "${NAMESPACE}" get pods \
      -l "${SGLANG_SELECTOR}" -o name)"
    if [[ -z "${pods}" ]]; then
      return 0
    fi
    "${SLEEP_BIN}" 1
  done

  log "SGLang pods did not exit within 120 seconds; refusing an unsafe GPU suspend"
  return 1
}

wait_for_compute_idle() {
  local processes

  for _ in $(seq 1 30); do
    if processes="$("${NVIDIA_SMI_BIN}" \
      --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" && \
      [[ -z "${processes//[[:space:]]/}" ]]; then
      return 0
    fi
    "${SLEEP_BIN}" 1
  done

  log "CUDA processes still own the GPU after 30 seconds; refusing an unsafe GPU suspend"
  return 1
}

wait_for_nvidia() {
  for _ in $(seq 1 60); do
    if "${NVIDIA_SMI_BIN}" --query-gpu=uuid --format=csv,noheader \
      >/dev/null 2>&1; then
      return 0
    fi
    "${SLEEP_BIN}" 2
  done

  log "NVIDIA driver did not recover within 120 seconds; leaving SGLang scaled to zero"
  return 1
}

case "${1:-}" in
  suspend)
    log "Stopping SGLang before NVIDIA suspend"
    wait_for_k3s
    "${K3S_BIN}" kubectl -n "${NAMESPACE}" scale \
      deployment/"${DEPLOYMENT}" --replicas=0
    wait_for_sglang_exit
    wait_for_compute_idle
    log "SGLang pod exited and the GPU is compute-idle; NVIDIA suspend may continue"
    ;;
  resume)
    log "Recovering SGLang after NVIDIA resume"
    wait_for_k3s
    wait_for_nvidia

    # Re-register the GPU with kubelet only after the host driver is healthy.
    "${K3S_BIN}" kubectl -n "${PLUGIN_NAMESPACE}" rollout restart \
      daemonset/"${PLUGIN_DAEMONSET}"
    "${K3S_BIN}" kubectl -n "${PLUGIN_NAMESPACE}" rollout status \
      daemonset/"${PLUGIN_DAEMONSET}" --timeout=180s

    "${K3S_BIN}" kubectl -n "${NAMESPACE}" scale \
      deployment/"${DEPLOYMENT}" --replicas=1
    log "SGLang restart requested; Kubernetes will wait for its health probes"
    ;;
  *)
    echo "Usage: $0 {suspend|resume}" >&2
    exit 2
    ;;
esac
