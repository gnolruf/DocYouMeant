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

# Download and install ONNX Runtime
ARG TARGETARCH
RUN ARCH=$(uname -m) && \
    HAS_CUDA=false && \
    if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
        HAS_CUDA=true; \
    fi && \
    if [ "$TARGETARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then \
        ORT_FILE="onnxruntime-linux-aarch64-1.22.0.tgz"; \
        ORT_DIR="onnxruntime-linux-aarch64-1.22.0"; \
    elif [ "$HAS_CUDA" = "true" ]; then \
        ORT_FILE="onnxruntime-linux-x64-gpu-1.22.0.tgz"; \
        ORT_DIR="onnxruntime-linux-x64-gpu-1.22.0"; \
    else \
        ORT_FILE="onnxruntime-linux-x64-1.22.0.tgz"; \
        ORT_DIR="onnxruntime-linux-x64-1.22.0"; \
    fi && \
    echo "Installing ONNX Runtime: $ORT_FILE (CUDA available: $HAS_CUDA)" && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/${ORT_FILE} \
    && tar -xzf ${ORT_FILE} \
    && cp ${ORT_DIR}/lib/libonnxruntime.so* /usr/lib/ \
    && cp ${ORT_DIR}/lib/libonnxruntime_providers_*.so* /usr/lib/ 2>/dev/null || true \
    && rm -rf ${ORT_DIR}*

# Download and install Pdfium binaries
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        PDFIUM_ARCH="arm64"; \
    else \
        PDFIUM_ARCH="x64"; \
    fi && \
    mkdir -p /tmp/pdfium \
    && cd /tmp/pdfium \
    && wget https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-${PDFIUM_ARCH}.tgz \
    && tar -xzf pdfium-linux-${PDFIUM_ARCH}.tgz \
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

# Production runtime with TensorRT support
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
