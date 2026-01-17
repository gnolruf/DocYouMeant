FROM nvcr.io/nvidia/tensorrt:25.01-py3 AS base

# Install system dependencies (LLVM 14 is available by default in Ubuntu 22.04)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    clang \
    libclang-dev \
    llvm-dev \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    libomp-dev \
    libgomp1 \
    git \
    libopencv-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install ONNX Runtime with CUDA support
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz \
    && tar -xzf onnxruntime-linux-x64-gpu-1.22.0.tgz \
    && cp onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so* /usr/lib/ \
    && cp onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime_providers_*.so* /usr/lib/ || true \
    && rm -rf onnxruntime-linux-x64-gpu-1.22.0*

# Download and install Pdfium binaries
RUN mkdir -p /tmp/pdfium \
    && cd /tmp/pdfium \
    && wget https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-x64.tgz \
    && tar -xzf pdfium-linux-x64.tgz \
    && cp lib/libpdfium.so /usr/lib/ \
    && cp -r include/* /usr/include/ \
    && cd / \
    && rm -rf /tmp/pdfium

# Copy and install Python requirements first for better caching
COPY scripts/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ENV LLVM_CONFIG_PATH=/usr/bin/llvm-config
ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib
ENV CLANG_PATH=/usr/bin/clang
ENV PKG_CONFIG_PATH=/usr/lib/pkgconfig
ENV ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
ENV PDFIUM_DYNAMIC_LIB_PATH=/usr/lib

# Development image
FROM base AS development
RUN apt-get update && apt-get install -y \
    git \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create developer user with same UID/GID that can access cargo
RUN useradd -m -s /bin/bash developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install Rust for the developer user
USER developer
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.92.0
ENV PATH="/home/developer/.cargo/bin:${PATH}"

WORKDIR /app

# Production build stage
FROM base AS builder

# Install Rust for building
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.92.0
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY . .
RUN cargo build --release

# Copy all ONNX Runtime libraries to a single location for easier copying
RUN mkdir -p /tmp/ort-libs && \
    cp /usr/lib/libonnxruntime.so* /tmp/ort-libs/ && \
    cp /usr/lib/libonnxruntime_providers_*.so* /tmp/ort-libs/ 2>/dev/null || true

# Production runtime with TensorRT support (25.01+ required for Blackwell/RTX 50 series)
FROM nvcr.io/nvidia/tensorrt:25.01-py3 AS runtime
RUN apt-get update && apt-get install -y \
    libomp-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/docyoumeant /usr/local/bin/
COPY --from=builder /tmp/ort-libs/* /usr/lib/
COPY --from=builder /usr/lib/libpdfium.so* /usr/lib/
ENV ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
ENV PDFIUM_DYNAMIC_LIB_PATH=/usr/lib
CMD ["docyoumeant"]
