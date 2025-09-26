# Use NVIDIA CUDA base image compatible with CUDA 12.6
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV TORCH_HOME=/app/checkpoints/torch_cache
ENV HF_HOME=/app/checkpoints/huggingface_cache

# Install system dependencies including mmcv requirements
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
    ninja-build \
    libopencv-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Upgrade pip and install wheel
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support (best compatibility for CUDA 12.6)
RUN pip3 install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install xformers (for efficient attention) - compatible version
RUN pip3 install --no-cache-dir xformers==0.0.28

# Install mmcv and related packages first (they need special handling)
RUN pip3 install --no-cache-dir \
    openmim \
    mmengine

# Install mmcv with CUDA support
RUN mim install "mmcv>=2.0.1"

# Install mmdet and mmpose
RUN mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0"

# Install core Python dependencies
RUN pip3 install --no-cache-dir \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    transformers>=4.30.0 \
    diffusers>=0.25.0 \
    accelerate>=0.20.0 \
    omegaconf>=2.3.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    resampy>=0.4.0 \
    opencv-python>=4.8.0 \
    face-alignment>=1.4.0 \
    mediapipe>=0.10.0 \
    Pillow>=9.5.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.3.0 \
    matplotlib>=3.7.0 \
    requests>=2.31.0 \
    psutil>=5.9.0 \
    pyyaml>=6.0 \
    tqdm>=4.65.0 \
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.4.8 \
    einops \
    ipython>=8.14.0

# Install whisper from OpenAI
RUN pip3 install --no-cache-dir git+https://github.com/openai/whisper.git

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