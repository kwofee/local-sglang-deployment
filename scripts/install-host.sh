#!/usr/bin/env bash
set -euo pipefail

TOOLKIT_VERSION="${TOOLKIT_VERSION:-1.20.0-1}"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is missing; install a supported NVIDIA driver first." >&2
  exit 1
fi
nvidia-smi >/dev/null

if ! command -v nvidia-ctk >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y \
    "nvidia-container-toolkit=${TOOLKIT_VERSION}" \
    "nvidia-container-toolkit-base=${TOOLKIT_VERSION}" \
    "libnvidia-container-tools=${TOOLKIT_VERSION}" \
    "libnvidia-container1=${TOOLKIT_VERSION}"
fi

# Keep Docker usable for local image checks as well as configuring the runtime
# binary that k3s detects when its bundled containerd starts.
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

if ! command -v k3s >/dev/null; then
  curl -sfL https://get.k3s.io \
    | INSTALL_K3S_EXEC="server --disable traefik --disable servicelb" sh -
else
  sudo systemctl restart k3s
fi

echo "Waiting for the k3s node to register..."
for _ in $(seq 1 60); do
  if sudo k3s kubectl get node --no-headers 2>/dev/null | grep -q .; then
    break
  fi
  sleep 2
done

if ! sudo k3s kubectl get node --no-headers 2>/dev/null | grep -q .; then
  echo "k3s did not register a node within 120 seconds." >&2
  echo "Inspect it with: sudo systemctl status k3s && sudo journalctl -u k3s -n 100" >&2
  exit 1
fi

sudo k3s kubectl wait --for=condition=Ready node --all --timeout=180s
echo "Host setup complete. Continue with: ./scripts/configure-secrets.sh"
