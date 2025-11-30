<div align="center">
  <img src="logo.png" alt="DocYouMeant Logo" width="600">
  
  # DocYouMeant
  
  *A lightweight Rust-based document understanding API*

  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Rust](https://img.shields.io/badge/rust-1.81%2B-orange.svg)](https://www.rust-lang.org)
  [![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
</div>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Model Setup](#model-setup)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

DocYouMeant is a high-performance document understanding API server built with Rust. Via an ONNX Runtime model pipeline, it provides OCR capabilities, layout detection, and document analysis through a RESTful API. Docyoumeant supports multiple document formats and languages, and is configurable depending on what document analysis task it needs to perform.

## Features

- **RESTful API**: HTTP API for document analysis with JSON responses
- **Text Detection & Recognition**: OCR capabilities powered by PaddleOCR models
- **Multi-format Support**: Process images (PNG, JPG), PDFs, Office documents (DOCX, XLSX), CSV, and text files
- **Multi-language Support**: English, Chinese, and extensible language support
- **Layout Analysis**: Document structure detection and region classification
- **High Performance**: Built with Rust and async runtime (Tokio) for speed and efficiency
- **Docker Support**: Easy deployment with Docker containers
- **Flexible Architecture**: Modular design for extensibility

## Quick Start

### Prerequisites

- Rust 1.81 or later
- Python 3.x (for model management)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gnolruf/DocYouMeant.git
   cd DocYouMeant
   ```

2. **Set up models:**
   
   Before running the application, you need to download and set up the required ONNX models:

   ```bash
   cd scripts
   pip install -r requirements.txt
   
   # Download all models
   python download_models.py
   
   # Or download for a specific language
   python download_models.py --language english
   
   cd ..
   ```

   The script downloads PaddleOCR models and converts them to ONNX format. See [Model Setup](#model-setup) for more details.

3. **Configure environment (optional):**
   ```bash
   cp .env.example .env
   # Edit .env to customize server address and logging
   ```

4. **Build the application:**
   ```bash
   cargo build --release
   ```

### Running the Application

Start the API server:

```bash
# Default (127.0.0.1:3000)
cargo run --release

# Custom address
DOCYOUMEANT_ADDR="0.0.0.0:8080" cargo run --release
```

Or use the compiled binary:

```bash
./target/release/docyoumeant
```

#### Using Docker

For development:
```bash
docker-compose up dev
```

For production build:
```bash
docker build -t docyoumeant .
docker run -p 3000:3000 docyoumeant
```

### Testing the API

**Health Check:**
```bash
curl http://127.0.0.1:3000/health | jq
```

**Using the Python Client:**
```bash
# Health check
python scripts/client.py health

# Analyze a document
python scripts/client.py tests/fixtures/png/test.png

# Analyze with questions
python scripts/client.py tests/fixtures/pdf/test.pdf 'What is the document about?'
```

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check endpoint |
| POST | `/api/v1/analyze` | Document analysis endpoint |

### Analyze Document

**Endpoint:** `POST /api/v1/analyze`

**Request Body:**
```json
{
  "data": "base64_encoded_document_data",
  "filename": "document.png",
  "language": "english",
  "questions": ["optional question 1", "optional question 2"]
}
```

**Response:**
```json
{
  "filename": "document.png",
  "document_type": "png",
  "regions": [
    {
      "region_type": "text|table|figure|title",
      "bbox": [x1, y1, x2, y2],
      "text_boxes": [
        {
          "bbox": [x1, y1, x2, y2],
          "text": "detected text",
          "confidence": 0.95
        }
      ]
    }
  ],
  "answers": ["answer 1", "answer 2"]
}
```

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy"
}
```

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Available variables:

- `DOCYOUMEANT_ADDR`: Server bind address (default: `127.0.0.1:3000`)
- `RUST_LOG`: Logging level (default: `docyoumeant=info,tower_http=debug`)
  - Levels: `trace`, `debug`, `info`, `warn`, `error`

### Running with Custom Configuration

```bash
# Custom address and debug logging
DOCYOUMEANT_ADDR="0.0.0.0:8080" RUST_LOG=debug cargo run --release

# Or using .env file
cargo run --release
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request (will provide contribution guidelines in the future).

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project would not be possible without the excellent work of:

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**: The foundational OCR models and techniques that power DocYouMeant's text detection and recognition capabilities
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)**: Instrumental in providing insights and approaches for efficient OCR implementation
- **[ort](https://github.com/pykeio/ort)**: Rust crate that enables ONNX model inference