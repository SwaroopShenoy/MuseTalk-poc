#!/bin/bash

# Create the checkpoints directory structure
echo "Creating directory structure..."
mkdir -p ./checkpoints/MuseTalk/musetalk
mkdir -p ./checkpoints/MuseTalk/sd-vae-ft-mse
mkdir -p ./checkpoints/MuseTalk/whisper

echo "Starting MuseTalk model downloads..."

# Download MuseTalk main model (largest file ~500MB)
echo "📥 Downloading MuseTalk main model (pytorch_model.bin)..."
curl -L -o ./checkpoints/MuseTalk/musetalk/pytorch_model.bin \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin"

# Download MuseTalk config
echo "📥 Downloading MuseTalk config (musetalk.json)..."
curl -L -o ./checkpoints/MuseTalk/musetalk/musetalk.json \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json"

# Download VAE model (required for latent space operations ~300MB)
echo "📥 Downloading VAE model (diffusion_pytorch_model.bin)..."
curl -L -o ./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.bin \
  "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin"

# Download VAE config
echo "📥 Downloading VAE config (config.json)..."
curl -L -o ./checkpoints/MuseTalk/sd-vae-ft-mse/config.json \
  "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json"

# Download Whisper model (~40MB)
echo "📥 Downloading Whisper tiny model (tiny.pt)..."
curl -L -o ./checkpoints/MuseTalk/whisper/tiny.pt \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"

echo "✅ All models downloaded successfully!"

# Verify downloads
echo ""
echo "📊 Downloaded file sizes:"
echo "MuseTalk model: $(du -h ./checkpoints/MuseTalk/musetalk/pytorch_model.bin | cut -f1)"
echo "VAE model: $(du -h ./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.bin | cut -f1)"
echo "Whisper model: $(du -h ./checkpoints/MuseTalk/whisper/tiny.pt | cut -f1)"
echo "Total size: $(du -sh ./checkpoints/MuseTalk | cut -f1)"

echo ""
echo "🚀 Models ready! You can now run: docker-compose up"