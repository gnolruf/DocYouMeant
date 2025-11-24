# Scripts

This directory contains utility scripts for managing ML models and a Python client for testing the DocYouMeant API.

> **Note:** For the main application documentation, please refer to the [root README](../README.md).

## Setup

```bash
pip install -r requirements.txt
```

## Prerequisites

The script requires `paddle2onnx` for model conversion. This is included in the requirements.txt file and will be installed automatically.

## Download Models

To download and convert all PaddleOCR models to ONNX format:

```bash
python download_models.py
```

To download models for a specific language only, include a language arg:

```bash
python download_models.py --language english
```

**Note**: Models without a language property (like text detection and angle classification) are always downloaded regardless of the language filter, as they are universal models used by all languages.

## Command Line Options

The script supports the following command line options:

- `--language LANGUAGE`: Download models for a specific language only (e.g., 'english', 'chinese'). When specified, only models with the matching language property will be downloaded. Models without a language property (universal models like text detection and angle classification) are always downloaded.

## What the Script Does

The script performs the following operations for each model:

1. **Downloads** PaddleOCR models from official Paddle repositories
2. **Extracts** the downloaded tar archives
3. **Converts** models from PaddlePaddle format to ONNX format using paddle2onnx
4. **Generates** language model configuration file
5. **Cleans up** temporary files and downloads
6. **Organizes** models in structured directories

## Language Models Configuration

The script automatically generates a `lang_models.json` configuration file that maps language identifiers to their corresponding model and dictionary files. This configuration is used by the application to dynamically load the appropriate models for different languages.

Example configuration:
```json
{
  "english": {
    "name": "english",
    "model_file": "models/onnx/text_recognition_en.onnx",
    "dict_file": "models/dict/en_dict.txt"
  },
  "chinese": {
    "name": "chinese", 
    "model_file": "models/onnx/text_recognition_ch.onnx",
    "dict_file": "models/dict/chinese_cht_dict.txt"
  }
}
```

## Python Client

A full-featured Python client for interacting with the DocYouMeant API.

### Installation

The client requires the `requests` library, which is included in the requirements.txt file.

### Usage

**Health Check:**
```bash
python scripts/client.py health
```

**Analyze a Document:**
```bash
python scripts/client.py path/to/document.pdf
python scripts/client.py tests/fixtures/png/test.png
python scripts/client.py tests/fixtures/pdf/test.pdf
```

**Analyze with Questions:**
```bash
python scripts/client.py tests/fixtures/pdf/test.pdf 'What is the document about?'
python scripts/client.py tests/fixtures/pdf/test.pdf 'What is the title?' 'Who is the author?'
```

## API Documentation

For detailed API documentation, supported document types, and server configuration, please refer to the [main project README](../README.md).
