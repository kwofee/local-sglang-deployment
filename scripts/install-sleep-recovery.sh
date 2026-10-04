#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer with sudo: sudo ./scripts/install-sleep-recovery.sh" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

install -m 0755 \
  "${REPO_DIR}/scripts/sglang-gpu-sleep.sh" \
  /usr/local/sbin/sglang-gpu-sleep

install -d -m 0755 \
  /etc/systemd/system/nvidia-suspend.service.d \
  /etc/systemd/system/nvidia-hibernate.service.d \
  /etc/systemd/system/nvidia-resume.service.d \
  /etc/systemd/system/systemd-suspend.service.d \
  /etc/systemd/system/systemd-hibernate.service.d

install -m 0644 \
  "${REPO_DIR}/deploy/systemd/nvidia-suspend-sglang.conf" \
  /etc/systemd/system/nvidia-suspend.service.d/sglang.conf
install -m 0644 \
  "${REPO_DIR}/deploy/systemd/nvidia-suspend-sglang.conf" \
  /etc/systemd/system/nvidia-hibernate.service.d/sglang.conf
install -m 0644 \
  "${REPO_DIR}/deploy/systemd/nvidia-resume-sglang.conf" \
  /etc/systemd/system/nvidia-resume.service.d/sglang.conf
install -m 0644 \
  "${REPO_DIR}/deploy/systemd/systemd-suspend-require-nvidia.conf" \
  /etc/systemd/system/systemd-suspend.service.d/require-nvidia.conf
install -m 0644 \
  "${REPO_DIR}/deploy/systemd/systemd-hibernate-require-nvidia.conf" \
  /etc/systemd/system/systemd-hibernate.service.d/require-nvidia.conf

systemctl daemon-reload
systemctl enable nvidia-suspend.service nvidia-hibernate.service nvidia-resume.service

echo "SGLang sleep recovery installed."
echo "Before sleep: SGLang fully exits and the GPU becomes compute-idle."
echo "After wake: the NVIDIA driver is health-checked before the GPU plugin and SGLang restart."
