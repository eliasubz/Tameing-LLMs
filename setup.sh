#!/usr/bin/env bash

set -e  # Exit immediately if a command exits with a non-zero status

git config --global user.email eliasls2002@yahoo.com	
git config --global user.name Colab_elias

echo "Installing PyTorch (CUDA 12.4 build)..."
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

echo "Installing Triton..."
uv pip install triton

echo "Installing Flash Attention (prebuilt wheel)..."
uv pip install \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl

echo "Installing Flash Linear Attention (FLA)..."
uv pip install git+https://github.com/fla-org/flash-linear-attention.git

echo "Installing experiment tracking utilities..."
uv pip install wandb tiktoken einops

# echo "Downloading VS Code CLI..."
# curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' \
#   --output vscode_cli.tar.gz

# echo "Extracting VS Code CLI..."
# tar -xf vscode_cli.tar.gz

# echo "Starting VS Code tunnel..."
# ./code tunnel --accept-server-license-terms

echo "Setup complete."