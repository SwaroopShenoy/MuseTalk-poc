#!/bin/bash

# Function to check if file exists and has reasonable size
check_file() {
    local file_path="$1"
    local min_size_kb="$2"
    
    if [ -f "$file_path" ]; then
        local file_size_kb=$(du -k "$file_path" | cut -f1)
        if [ "$file_size_kb" -gt "$min_size_kb" ]; then
            return 0  # File exists and is large enough
        else
            echo "⚠️ File $file_path exists but is too small (${file_size_kb}KB), re-downloading..."
            rm -f "$file_path"
            return 1
        fi
    else
        return 1  # File doesn't exist
    fi
}

# Function to download file only if needed
download_if_needed() {
    local url="$1"
    local output_path="$2"
    local description="$3"
    local min_size_kb="$4"
    
    if check_file "$output_path" "$min_size_kb"; then
        echo "✅ $description already exists ($(du -h "$output_path" | cut -f1)), skipping download"
        return 0
    fi
    
    echo "📥 Downloading $description..."
    curl -L --fail --show-error --silent -o "$output_path" "$url"
    
    if [ $? -eq 0 ] && check_file "$output_path" "$min_size_kb"; then
        echo "✅ Successfully downloaded $description ($(du -h "$output_path" | cut -f1))"
        return 0
    else
        echo "❌ Failed to download $description"
        rm -f "$output_path"
        return 1
    fi
}

# Create the checkpoints directory structure
echo "Creating directory structure..."
mkdir -p ./checkpoints/MuseTalk/musetalk
mkdir -p ./checkpoints/MuseTalk/sd-vae-ft-mse
mkdir -p ./checkpoints/MuseTalk/whisper

echo "Starting MuseTalk model downloads (idempotent)..."

# Download MuseTalk main model (largest file ~500MB, min 400MB)
download_if_needed \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin" \
    "./checkpoints/MuseTalk/musetalk/pytorch_model.bin" \
    "MuseTalk main model (pytorch_model.bin)" \
    2709600

# Download MuseTalk config (small file, min 1KB)
download_if_needed \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" \
    "./checkpoints/MuseTalk/musetalk/musetalk.json" \
    "MuseTalk config (musetalk.json)" \
    1

# Download VAE model (required for latent space operations ~300MB, min 250MB)
download_if_needed \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" \
    "./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.bin" \
    "VAE model (diffusion_pytorch_model.bin)" \
    256000

# Download VAE config (small file, min 1KB)
download_if_needed \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" \
    "./checkpoints/MuseTalk/sd-vae-ft-mse/config.json" \
    "VAE config (config.json)" \
    1

# Download Whisper model (~40MB, min 30MB)
download_if_needed \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin" \
    "./checkpoints/MuseTalk/whisper/tiny.pt" \
    "Whisper tiny model (tiny.pt)" \
    30720

echo ""
echo "✅ All models verified/downloaded successfully!"

# Verify all required files exist
required_files=(
    "./checkpoints/MuseTalk/musetalk/pytorch_model.bin"
    "./checkpoints/MuseTalk/musetalk/musetalk.json"
    "./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.bin"
    "./checkpoints/MuseTalk/sd-vae-ft-mse/config.json"
    "./checkpoints/MuseTalk/whisper/tiny.pt"
)

echo "🔍 Verifying all required files..."
all_present=true
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        all_present=false
    fi
done

if [ "$all_present" = true ]; then
    echo "✅ All required files present!"
    
    # Display final file sizes
    echo ""
    echo "📊 Final file sizes:"
    if [ -f "./checkpoints/MuseTalk/musetalk/pytorch_model.bin" ]; then
        echo "MuseTalk model: $(du -h ./checkpoints/MuseTalk/musetalk/pytorch_model.bin | cut -f1)"
    fi
    if [ -f "./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.bin" ]; then
        echo "VAE model: $(du -h ./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.bin | cut -f1)"
    fi
    if [ -f "./checkpoints/MuseTalk/whisper/tiny.pt" ]; then
        echo "Whisper model: $(du -h ./checkpoints/MuseTalk/whisper/tiny.pt | cut -f1)"
    fi
    echo "Total size: $(du -sh ./checkpoints/MuseTalk | cut -f1)"
    
    echo ""
    echo "🚀 Models ready! You can now run: docker-compose up"
    exit 0
else
    echo "❌ Some files are missing. Please check the download errors above."
    exit 1
fi