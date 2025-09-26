# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TORCH_HOME=/app/checkpoints/torch_cache
ENV HF_HOME=/app/checkpoints/huggingface_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Upgrade pip and install wheel
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.6 support first (specific versions to avoid conflicts)
RUN pip3 install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126

# Install xformers (for efficient attention)
RUN pip3 install --no-cache-dir \
    xformers==0.0.30 \
    --index-url https://download.pytorch.org/whl/cu126

# Install other Python dependencies (excluding torch variants already installed)
RUN pip3 install --no-cache-dir -r requirements.txt \
    --find-links https://download.pytorch.org/whl/cu126

# Install additional MuseTalk specific packages
RUN pip3 install --no-cache-dir \
    git+https://github.com/openai/whisper.git \
    face-alignment \
    mediapipe

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/output /app/input \
    /app/checkpoints/MuseTalk \
    /app/checkpoints/torch_cache \
    /app/checkpoints/huggingface_cache

# Copy application files
COPY main.py .

# Set permissions
RUN chmod +x main.py

# Create non-root user for security
RUN useradd -m -u 1000 musetalk && chown -R musetalk:musetalk /app
USER musetalk

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=300s --timeout=15s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python3", "main.py"]