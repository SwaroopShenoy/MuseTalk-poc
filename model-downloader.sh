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

# Function to download file with progress bar and multiple URLs
download_with_fallback() {
    local urls=("$@")
    local output_path="${urls[-2]}"
    local description="${urls[-1]}"
    local min_size_kb="${urls[-3]}"
    
    # Remove last 3 elements to get just URLs
    unset urls[-1] urls[-1] urls[-1]
    
    if check_file "$output_path" "$min_size_kb"; then
        echo "✅ $description already exists ($(du -h "$output_path" | cut -f1)), skipping download"
        return 0
    fi
    
    echo "📥 Downloading $description..."
    
    # Try each URL until one succeeds
    for url in "${urls[@]}"; do
        local repo_name=$(echo "$url" | sed 's|.*/\([^/]*\)/resolve/.*|\1|')
        echo "  Trying from $repo_name..."
        
        # Download with progress bar
        if curl -L --fail --show-error --progress-bar -o "$output_path" "$url"; then
            if check_file "$output_path" "$min_size_kb"; then
                echo "✅ Successfully downloaded $description from $repo_name ($(du -h "$output_path" | cut -f1))"
                return 0
            else
                echo "❌ Downloaded file from $repo_name is too small, trying next URL..."
                rm -f "$output_path"
            fi
        else
            echo "❌ Failed to download from $repo_name, trying next URL..."
            rm -f "$output_path"
        fi
    done
    
    echo "❌ Failed to download $description from all sources"
    return 1
}

# Create the checkpoints directory structure
echo "Creating directory structure..."
mkdir -p ./checkpoints/MuseTalk/musetalk
mkdir -p ./checkpoints/MuseTalk/sd-vae-ft-mse
mkdir -p ./checkpoints/MuseTalk/whisper

echo "Starting MuseTalk model downloads with fallback URLs..."
echo "This script is idempotent - existing files will be verified and skipped"
echo ""

# Download MuseTalk main model (largest file ~3.2GB)
download_with_fallback \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin" \
    "https://huggingface.co/camenduru/MuseTalk/resolve/main/musetalk/pytorch_model.bin" \
    "https://huggingface.co/ameerazam08/MuseTalk/resolve/main/musetalk/pytorch_model.bin" \
    "2800000" \
    "./checkpoints/MuseTalk/musetalk/pytorch_model.bin" \
    "MuseTalk main model (pytorch_model.bin)"

# Download MuseTalk config (small file)
download_with_fallback \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" \
    "https://huggingface.co/camenduru/MuseTalk/resolve/main/musetalk/musetalk.json" \
    "https://huggingface.co/ameerazam08/MuseTalk/resolve/main/musetalk/musetalk.json" \
    "1" \
    "./checkpoints/MuseTalk/musetalk/musetalk.json" \
    "MuseTalk config (musetalk.json)"

# Download VAE model (try safetensors first, then bin)
if ! download_with_fallback \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors" \
    "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors" \
    "250000" \
    "./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.safetensors" \
    "VAE model (safetensors format)"; then
    
    echo "Safetensors failed, trying .bin format..."
    download_with_fallback \
        "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" \
        "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.bin" \
        "250000" \
        "./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.bin" \
        "VAE model (bin format)"
fi

# Download VAE config
download_with_fallback \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" \
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/config.json" \
    "1" \
    "./checkpoints/MuseTalk/sd-vae-ft-mse/config.json" \
    "VAE config (config.json)"

# Download Whisper model (~144MB)
download_with_fallback \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin" \
    "https://huggingface.co/openai/whisper-base/resolve/main/pytorch_model.bin" \
    "30720" \
    "./checkpoints/MuseTalk/whisper/tiny.pt" \
    "Whisper tiny model (tiny.pt)"

echo ""
echo "🔍 Verifying all required files..."

# Verify all required files exist
required_files=(
    "./checkpoints/MuseTalk/musetalk/pytorch_model.bin"
    "./checkpoints/MuseTalk/musetalk/musetalk.json"
    "./checkpoints/MuseTalk/sd-vae-ft-mse/config.json"
    "./checkpoints/MuseTalk/whisper/tiny.pt"
)

# Check for either VAE format
vae_present=false
if [ -f "./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.safetensors" ]; then
    vae_present=true
    echo "✅ VAE model found (safetensors format)"
elif [ -f "./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.bin" ]; then
    vae_present=true
    echo "✅ VAE model found (bin format)"
else
    echo "❌ VAE model missing (neither safetensors nor bin format found)"
fi

all_present=true
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        all_present=false
    else
        echo "✅ Found: $(basename "$file")"
    fi
done

if [ "$all_present" = true ] && [ "$vae_present" = true ]; then
    echo ""
    echo "✅ All required files present!"
    
    # Display final file sizes
    echo ""
    echo "📊 Final file sizes:"
    if [ -f "./checkpoints/MuseTalk/musetalk/pytorch_model.bin" ]; then
        echo "  MuseTalk model: $(du -h ./checkpoints/MuseTalk/musetalk/pytorch_model.bin | cut -f1)"
    fi
    if [ -f "./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.safetensors" ]; then
        echo "  VAE model (safetensors): $(du -h ./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.safetensors | cut -f1)"
    elif [ -f "./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.bin" ]; then
        echo "  VAE model (bin): $(du -h ./checkpoints/MuseTalk/sd-vae-ft-mse/diffusion_pytorch_model.bin | cut -f1)"
    fi
    if [ -f "./checkpoints/MuseTalk/whisper/tiny.pt" ]; then
        echo "  Whisper model: $(du -h ./checkpoints/MuseTalk/whisper/tiny.pt | cut -f1)"
    fi
    echo "  Total size: $(du -sh ./checkpoints/MuseTalk | cut -f1)"
    
    echo ""
    echo "🚀 Models ready! You can now run your MuseTalk API:"
    echo "   docker-compose up"
    echo "   OR"
    echo "   python main.py"
    echo ""
    echo "💡 Tip: This script is idempotent - run it anytime to verify/re-download missing files"
    exit 0
else
    echo ""
    echo "❌ Some files are missing. Please check the download errors above."
    echo "💡 Try running this script again - it will retry failed downloads."
    exit 1
fi