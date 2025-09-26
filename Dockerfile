# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/checkpoints/torch_cache
ENV HF_HOME=/app/checkpoints/huggingface_cache

# Install ONLY essential system dependencies (much faster)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Upgrade pip and install wheel
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install minimal Python dependencies for MuseTalk
RUN pip3 install --no-cache-dir -r requirements.txt

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