import os
import tarfile
import requests
import subprocess
import shutil
import json
import argparse
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from huggingface_hub import snapshot_download

def load_models_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)

class ModelProcessor(ABC):
    def __init__(self, model_name: str, config: Dict[str, Any], base_dir: Path):
        self.model_name = model_name
        self.config = config
        self.base_dir = base_dir
        self.models_dir = base_dir / "models"
        self.download_dir = self.models_dir / "download"
        self.onnx_dir = self.models_dir / "onnx"
        self.tokenizer_dir = self.models_dir / "tokenizer"
        
        for d in [self.models_dir, self.download_dir, self.onnx_dir, self.tokenizer_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def process(self):
        pass

    def cleanup(self, path: Path):
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

class PaddleModelProcessor(ModelProcessor):
    def process(self):
        print(f"Processing {self.model_name}...")
        tar_path = self.download_dir / f"{self.model_name}.tar"
        print(f"Downloading {self.config['url']}...")
        self._download_file(self.config['url'], tar_path)
        
        print("Extracting...")
        extract_path = self.download_dir / self.model_name
        self._extract_tar(tar_path, extract_path)
        
        print("Converting to ONNX...")
        model_dir = extract_path / self.config['dir']
        output_path = self.onnx_dir / self.config['output']
        self._convert_to_onnx(
            model_dir, 
            output_path, 
            self.config['model_filename'], 
            self.config['params_filename']
        )

        print("Cleaning up...")
        self.cleanup(tar_path)
        self.cleanup(extract_path)
        print(f"Successfully processed {self.model_name}")

    def _download_file(self, url: str, dest_path: Path) -> None:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def _extract_tar(self, tar_path: Path, extract_path: Path) -> None:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_path)

    def _convert_to_onnx(self, model_dir: Path, output_file: Path, model_filename: str, params_filename: str) -> None:
        cmd = [
            "paddle2onnx",
            "--model_dir", str(model_dir),
            "--model_filename", model_filename,
            "--params_filename", params_filename,
            "--save_file", str(output_file)
        ]
        subprocess.run(cmd, check=True)

class HuggingFaceModelProcessor(ModelProcessor):
    def process(self):
        print(f"Downloading from Hugging Face: {self.config['repo']}...")
        
        variants = self.config.get("variants", [])
        if not variants:
            variants = [{
                "name": "default",
                "include_pattern": self.config.get("include_pattern"),
                "source_subdir": self.config.get("source_subdir", ""),
                "onnx_files": self.config.get("onnx_files", [])
            }]
        
        for variant in variants:
            self._process_variant(variant)
            
        if "tokenizer_files" in self.config and "tokenizer_dir" in self.config:
            self._process_tokenizer()

    def _process_variant(self, variant: Dict[str, Any]):
        variant_name = variant.get("name", "default")
        print(f"\nProcessing {variant_name.upper()} variant...")
        
        if variant_name == "default":
             temp_output_dir = self.download_dir / self.config['output_dir']
        else:
             temp_output_dir = self.download_dir / f"{self.config['output_dir']}_{variant_name}"

        self._download_huggingface_model(
            self.config['repo'],
            variant['include_pattern'],
            temp_output_dir
        )
        print(f"Successfully downloaded {self.model_name} ({variant_name})")
        
        onnx_files = variant.get('onnx_files', self.config.get('onnx_files', []))
        if onnx_files:
            print(f"Moving {variant_name.upper()} ONNX files to models/onnx/...")
            
            if "onnx_subdir" in self.config:
                if variant_name == "default":
                    model_onnx_dir = self.onnx_dir / self.config['onnx_subdir']
                else:
                    model_onnx_dir = self.onnx_dir / self.config['onnx_subdir'] / variant_name
                model_onnx_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Created subdirectory: {model_onnx_dir.relative_to(self.onnx_dir)}")
            else:
                if variant_name == "default":
                    model_onnx_dir = self.onnx_dir
                else:
                    model_onnx_dir = self.onnx_dir / variant_name
                    model_onnx_dir.mkdir(parents=True, exist_ok=True)
            
            source_subdir = variant.get('source_subdir', '')
            for onnx_file in onnx_files:
                src_file = temp_output_dir / source_subdir / onnx_file
                if src_file.exists():
                    dst_file = model_onnx_dir / onnx_file
                    shutil.copy2(src_file, dst_file)
                    print(f"  Copied {onnx_file} -> {dst_file.relative_to(self.onnx_dir)}")
                else:
                    print(f"  Warning: {onnx_file} not found at {src_file}")
        
        self.cleanup(temp_output_dir)

    def _process_tokenizer(self):
        print("\nMoving tokenizer files to models/tokenizer/...")
        model_tokenizer_dir = self.tokenizer_dir / self.config['tokenizer_dir']
        model_tokenizer_dir.mkdir(exist_ok=True)
        
        tokenizer_source = self.config.get('tokenizer_source_subdir')
        if tokenizer_source is None:
             variants = self.config.get("variants", [])
             if variants:
                 tokenizer_source = variants[0].get('source_subdir', '')
             else:
                 tokenizer_source = self.config.get("source_subdir", "")

        temp_tokenizer_dir = self.download_dir / f"{self.config['output_dir']}_tokenizer"
        
        if tokenizer_source:
            combined_pattern = f"{tokenizer_source}/*.json"
        else:
            combined_pattern = "*.json"
        
        self._download_huggingface_model(
            self.config['repo'],
            combined_pattern,
            temp_tokenizer_dir
        )
        
        for tokenizer_file in self.config['tokenizer_files']:
            if tokenizer_source:
                src_file = temp_tokenizer_dir / tokenizer_source / tokenizer_file
            else:
                src_file = temp_tokenizer_dir / tokenizer_file

            if src_file.exists():
                dst_file = model_tokenizer_dir / tokenizer_file
                shutil.copy2(src_file, dst_file)
                print(f"  Copied {tokenizer_file} -> {model_tokenizer_dir.relative_to(self.tokenizer_dir)}/{tokenizer_file}")
            else:
                print(f"  Warning: {tokenizer_file} not found at {src_file}")
        
        self.cleanup(temp_tokenizer_dir)

    def _download_huggingface_model(self, repo: str, include_pattern: str, output_dir: Path) -> None:
        snapshot_download(
            repo_id=repo,
            allow_patterns=include_pattern,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False
        )

def generate_ocr_lang_models_config(models_dir: Path, models: dict) -> None:
    """Generate a JSON config file for language models.
    
    For each language, prioritizes models where is_script_model is false
    over script-based models that may technically support the language.
    """
    ocr_lang_models = {}
    
    for model_name, model_info in models.items():
        if "languages" in model_info and "dict_filename" in model_info:
            is_script_model = model_info.get("is_script_model", False)
            
            for language in model_info["languages"]:
                if language not in ocr_lang_models:
                    ocr_lang_models[language] = {
                        "name": language,
                        "model_file": f"models/onnx/{model_info['output']}",
                        "dict_file": f"models/dict/{model_info['dict_filename']}",
                        "is_script_model": is_script_model
                    }
                elif not is_script_model and ocr_lang_models[language].get("is_script_model", False):
                    ocr_lang_models[language] = {
                        "name": language,
                        "model_file": f"models/onnx/{model_info['output']}",
                        "dict_file": f"models/dict/{model_info['dict_filename']}",
                        "is_script_model": is_script_model
                    }
    
    config_file = models_dir / "ocr_lang_models.json"
    with open(config_file, 'w') as f:
        json.dump(ocr_lang_models, f, indent=2)
    
    print(f"Generated language models config: {config_file}")

def filter_models_by_language(models: dict, target_language: str = None) -> dict:
    if target_language is None:
        return models
    
    filtered_models = {}
    for model_name, model_info in models.items():
        if "languages" not in model_info:
            filtered_models[model_name] = model_info
        elif target_language in model_info.get("languages", []):
            filtered_models[model_name] = model_info
    
    return filtered_models

def get_processor(model_name: str, config: Dict[str, Any], base_dir: Path) -> ModelProcessor:
    if config.get("type") == "huggingface":
        return HuggingFaceModelProcessor(model_name, config, base_dir)
    return PaddleModelProcessor(model_name, config, base_dir)

def main():
    parser = argparse.ArgumentParser(description="Download and convert PaddleOCR models to ONNX format")
    parser.add_argument(
        "--language", 
        type=str, 
        help="Download models for specific language only (e.g., 'english', 'chinese'). Models without language property are always downloaded."
    )
    args = parser.parse_args()
    
    models_config_path = Path(__file__).parent.parent / "models" / "models.json"
    models = load_models_config(models_config_path)
    
    models_to_process = filter_models_by_language(models, args.language)
    
    if args.language:
        print(f"Filtering models for language: {args.language}")
        print(f"Models to process: {list(models_to_process.keys())}")
    else:
        print("Processing all models")
    
    base_dir = Path(__file__).parent.parent
    
    for model_name, model_info in models_to_process.items():
        processor = get_processor(model_name, model_info, base_dir)
        processor.process()
    
    models_dir = base_dir / "models"
    generate_ocr_lang_models_config(models_dir, models_to_process)

    download_dir = models_dir / "download"
    if download_dir.exists():
        shutil.rmtree(download_dir)

if __name__ == "__main__":
    main()
