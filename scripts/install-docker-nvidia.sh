#!/usr/bin/env bash
# Install Docker Engine and the NVIDIA Container Toolkit on Ubuntu, so that
# `nawat sandbox` can run containers with the GPU.
#
#   sudo bash scripts/install-docker-nvidia.sh
#
# Follows the official instructions:
#   https://docs.docker.com/engine/install/ubuntu/
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# Safe to run again: each step skips what is already in place.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run it with sudo: sudo bash $0" >&2
  exit 1
fi
TARGET_USER="${SUDO_USER:-}"
. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-$VERSION_CODENAME}"
ARCH="$(dpkg --print-architecture)"

echo "==> Docker Engine (Ubuntu $CODENAME, $ARCH)"
apt-get update
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
if [ ! -s /etc/apt/keyrings/docker.asc ]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
fi
echo "deb [arch=$ARCH signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $CODENAME stable" \
  > /etc/apt/sources.list.d/docker.list

echo "==> NVIDIA Container Toolkit"
if [ ! -s /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
fi
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit

echo "==> Let Docker use the GPU"
nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker

if [ -n "$TARGET_USER" ]; then
  usermod -aG docker "$TARGET_USER"
  echo "==> $TARGET_USER added to the docker group (log out and in, or run: newgrp docker)"
fi

echo "==> Test: a CUDA container sees the GPU"
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

echo
echo "Done. Next: newgrp docker   (or log out and in), then: nawat sandbox build"
