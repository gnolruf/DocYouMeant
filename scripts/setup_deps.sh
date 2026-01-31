#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="1.23.2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPS_DIR="$PROJECT_ROOT/deps"
DEPS_LIB="$DEPS_DIR/lib"
DEPS_INCLUDE="$DEPS_DIR/include"

echo "==> DocYouMeant dependency setup"
echo "    Project root: $PROJECT_ROOT"
echo "    Install dir:  $DEPS_DIR"
echo ""

# --- Detect OS and architecture ---
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)  PLATFORM="linux" ;;
    Darwin) PLATFORM="mac" ;;
    *)      echo "Error: Unsupported OS '$OS'. Use setup_deps.ps1 for Windows." >&2; exit 1 ;;
esac

case "$ARCH" in
    x86_64|amd64)  ARCH_TAG="x64" ;;
    aarch64|arm64) ARCH_TAG="arm64" ;;
    *)             echo "Error: Unsupported architecture '$ARCH'." >&2; exit 1 ;;
esac

echo "    Platform:     $PLATFORM ($ARCH_TAG)"

# --- Detect CUDA availability (Linux only) ---
HAS_CUDA=false
if [ "$PLATFORM" = "linux" ]; then
    if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then
        HAS_CUDA=true
    fi
fi
echo "    CUDA:         $HAS_CUDA"
echo ""

mkdir -p "$DEPS_LIB" "$DEPS_INCLUDE"

# --- Helper: download and extract ---
download_and_extract() {
    local url="$1"
    local dest="$2"
    local filename
    filename="$(basename "$url")"

    echo "    Downloading $filename ..."
    if command -v wget > /dev/null 2>&1; then
        wget -q --show-progress -O "$dest/$filename" "$url"
    elif command -v curl > /dev/null 2>&1; then
        curl -fSL --progress-bar -o "$dest/$filename" "$url"
    else
        echo "Error: neither wget nor curl found." >&2
        exit 1
    fi

    echo "    Extracting ..."
    case "$filename" in
        *.tgz|*.tar.gz) tar -xzf "$dest/$filename" -C "$dest" ;;
        *.zip)          unzip -qo "$dest/$filename" -d "$dest" ;;
        *)              echo "Error: unknown archive format '$filename'" >&2; exit 1 ;;
    esac

    rm -f "$dest/$filename"
}

# ============================================================
# OnnxRuntime
# ============================================================
echo "==> Installing OnnxRuntime v${ORT_VERSION}"

ORT_BASE_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}"

if [ "$PLATFORM" = "linux" ]; then
    if [ "$ARCH_TAG" = "arm64" ]; then
        ORT_NAME="onnxruntime-linux-aarch64-${ORT_VERSION}"
    elif [ "$HAS_CUDA" = "true" ]; then
        ORT_NAME="onnxruntime-linux-x64-gpu-${ORT_VERSION}"
    else
        ORT_NAME="onnxruntime-linux-x64-${ORT_VERSION}"
    fi
elif [ "$PLATFORM" = "mac" ]; then
    if [ "$ARCH_TAG" = "arm64" ]; then
        ORT_NAME="onnxruntime-osx-arm64-${ORT_VERSION}"
    else
        ORT_NAME="onnxruntime-osx-x86_64-${ORT_VERSION}"
    fi
fi

ORT_FILE="${ORT_NAME}.tgz"
ORT_URL="${ORT_BASE_URL}/${ORT_FILE}"

TMP_ORT="$(mktemp -d)"
download_and_extract "$ORT_URL" "$TMP_ORT"

# Copy libraries
if [ "$PLATFORM" = "linux" ]; then
    cp "$TMP_ORT/$ORT_NAME/lib"/libonnxruntime.so* "$DEPS_LIB/"
    cp "$TMP_ORT/$ORT_NAME/lib"/libonnxruntime_providers_*.so* "$DEPS_LIB/" 2>/dev/null || true
elif [ "$PLATFORM" = "mac" ]; then
    cp "$TMP_ORT/$ORT_NAME/lib"/libonnxruntime.*.dylib "$DEPS_LIB/"
    # Create a symlink without version for convenience
    (cd "$DEPS_LIB" && ln -sf libonnxruntime.${ORT_VERSION}.dylib libonnxruntime.dylib 2>/dev/null || true)
fi

# Copy headers
if [ -d "$TMP_ORT/$ORT_NAME/include" ]; then
    cp -r "$TMP_ORT/$ORT_NAME/include/"* "$DEPS_INCLUDE/"
fi

rm -rf "$TMP_ORT"
echo "    OnnxRuntime installed to $DEPS_LIB"
echo ""

# ============================================================
# Pdfium
# ============================================================
echo "==> Installing pdfium (latest)"

PDFIUM_BASE_URL="https://github.com/bblanchon/pdfium-binaries/releases/latest/download"

if [ "$PLATFORM" = "linux" ]; then
    PDFIUM_FILE="pdfium-linux-${ARCH_TAG}.tgz"
elif [ "$PLATFORM" = "mac" ]; then
    PDFIUM_FILE="pdfium-mac-${ARCH_TAG}.tgz"
fi

PDFIUM_URL="${PDFIUM_BASE_URL}/${PDFIUM_FILE}"

TMP_PDFIUM="$(mktemp -d)"
download_and_extract "$PDFIUM_URL" "$TMP_PDFIUM"

# Copy library
if [ "$PLATFORM" = "linux" ]; then
    cp "$TMP_PDFIUM/lib"/libpdfium.so* "$DEPS_LIB/" 2>/dev/null || true
elif [ "$PLATFORM" = "mac" ]; then
    cp "$TMP_PDFIUM/lib"/libpdfium.dylib "$DEPS_LIB/" 2>/dev/null || true
fi

# Copy headers
if [ -d "$TMP_PDFIUM/include" ]; then
    cp -r "$TMP_PDFIUM/include/"* "$DEPS_INCLUDE/"
fi

rm -rf "$TMP_PDFIUM"
echo "    pdfium installed to $DEPS_LIB"
echo ""

# ============================================================
# Environment variables
# ============================================================
if [ "$PLATFORM" = "linux" ]; then
    ORT_LIB_FILE="$DEPS_LIB/libonnxruntime.so"
elif [ "$PLATFORM" = "mac" ]; then
    ORT_LIB_FILE="$DEPS_LIB/libonnxruntime.dylib"
fi

ENV_FILE="$PROJECT_ROOT/.env"

cat > "$ENV_FILE" <<EOF
ORT_DYLIB_PATH=${ORT_LIB_FILE}
PDFIUM_DYNAMIC_LIB_PATH=${DEPS_LIB}
EOF

# Write .cargo/config.toml so cargo commands pick up the env vars automatically
CARGO_CONFIG_DIR="$PROJECT_ROOT/.cargo"
mkdir -p "$CARGO_CONFIG_DIR"
cat > "$CARGO_CONFIG_DIR/config.toml" <<EOF
[env]
ORT_DYLIB_PATH = "${ORT_LIB_FILE}"
PDFIUM_DYNAMIC_LIB_PATH = "${DEPS_LIB}"
EOF

echo "==> Environment variables configured"
echo ""
echo "    ORT_DYLIB_PATH=${ORT_LIB_FILE}"
echo "    PDFIUM_DYNAMIC_LIB_PATH=${DEPS_LIB}"
echo ""
echo "    Written to .env and .cargo/config.toml"
echo "    cargo build/run will pick these up automatically."
echo ""
echo "==> Setup complete!"
