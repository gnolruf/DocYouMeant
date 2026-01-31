#Requires -Version 5.1
<#
.SYNOPSIS
    Installs OnnxRuntime and pdfium dependencies for DocYouMeant on Windows.
.DESCRIPTION
    Downloads and extracts OnnxRuntime and pdfium binaries into deps/ directory.
    Auto-detects CUDA availability for GPU-accelerated OnnxRuntime.
#>

$ErrorActionPreference = "Stop"

$ORT_VERSION = "1.22.0"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DepsDir = Join-Path $ProjectRoot "deps"
$DepsLib = Join-Path $DepsDir "lib"
$DepsInclude = Join-Path $DepsDir "include"

Write-Host "==> DocYouMeant dependency setup"
Write-Host "    Project root: $ProjectRoot"
Write-Host "    Install dir:  $DepsDir"
Write-Host ""

# --- Detect CUDA ---
$HasCuda = $false
try {
    $null = Get-Command nvidia-smi -ErrorAction Stop
    $null = & nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) { $HasCuda = $true }
} catch {}

Write-Host "    Platform:     windows (x64)"
Write-Host "    CUDA:         $HasCuda"
Write-Host ""

# Create directories
New-Item -ItemType Directory -Force -Path $DepsLib | Out-Null
New-Item -ItemType Directory -Force -Path $DepsInclude | Out-Null

# ============================================================
# OnnxRuntime
# ============================================================
Write-Host "==> Installing OnnxRuntime v${ORT_VERSION}"

$OrtBaseUrl = "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}"

if ($HasCuda) {
    $OrtName = "onnxruntime-win-x64-gpu-${ORT_VERSION}"
} else {
    $OrtName = "onnxruntime-win-x64-${ORT_VERSION}"
}

$OrtFile = "${OrtName}.zip"
$OrtUrl = "${OrtBaseUrl}/${OrtFile}"

$TmpOrt = Join-Path ([System.IO.Path]::GetTempPath()) "ort_download"
if (Test-Path $TmpOrt) { Remove-Item -Recurse -Force $TmpOrt }
New-Item -ItemType Directory -Force -Path $TmpOrt | Out-Null

$OrtZipPath = Join-Path $TmpOrt $OrtFile
Write-Host "    Downloading $OrtFile ..."
Invoke-WebRequest -Uri $OrtUrl -OutFile $OrtZipPath -UseBasicParsing

Write-Host "    Extracting ..."
Expand-Archive -Path $OrtZipPath -DestinationPath $TmpOrt -Force

# Copy libraries
$OrtLibDir = Join-Path $TmpOrt $OrtName "lib"
Copy-Item -Path (Join-Path $OrtLibDir "onnxruntime.dll") -Destination $DepsLib -Force
Copy-Item -Path (Join-Path $OrtLibDir "onnxruntime.lib") -Destination $DepsLib -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $OrtLibDir -Filter "onnxruntime_providers_*" -ErrorAction SilentlyContinue |
    Copy-Item -Destination $DepsLib -Force

# Copy headers
$OrtIncDir = Join-Path $TmpOrt $OrtName "include"
if (Test-Path $OrtIncDir) {
    Copy-Item -Path "$OrtIncDir\*" -Destination $DepsInclude -Recurse -Force
}

Remove-Item -Recurse -Force $TmpOrt
Write-Host "    OnnxRuntime installed to $DepsLib"
Write-Host ""

# ============================================================
# Pdfium
# ============================================================
Write-Host "==> Installing pdfium (latest)"

$PdfiumUrl = "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-x64.tgz"

$TmpPdfium = Join-Path ([System.IO.Path]::GetTempPath()) "pdfium_download"
if (Test-Path $TmpPdfium) { Remove-Item -Recurse -Force $TmpPdfium }
New-Item -ItemType Directory -Force -Path $TmpPdfium | Out-Null

$PdfiumTgzPath = Join-Path $TmpPdfium "pdfium-win-x64.tgz"
Write-Host "    Downloading pdfium-win-x64.tgz ..."
Invoke-WebRequest -Uri $PdfiumUrl -OutFile $PdfiumTgzPath -UseBasicParsing

Write-Host "    Extracting ..."
# Use tar (available on Windows 10+)
tar -xzf $PdfiumTgzPath -C $TmpPdfium

# Copy library
$PdfiumDll = Join-Path $TmpPdfium "bin" "pdfium.dll"
if (Test-Path $PdfiumDll) {
    Copy-Item -Path $PdfiumDll -Destination $DepsLib -Force
} else {
    # Some releases put it in lib/
    Get-ChildItem -Path $TmpPdfium -Filter "pdfium.dll" -Recurse |
        Select-Object -First 1 |
        Copy-Item -Destination $DepsLib -Force
}

# Copy headers
$PdfiumIncDir = Join-Path $TmpPdfium "include"
if (Test-Path $PdfiumIncDir) {
    Copy-Item -Path "$PdfiumIncDir\*" -Destination $DepsInclude -Recurse -Force
}

Remove-Item -Recurse -Force $TmpPdfium
Write-Host "    pdfium installed to $DepsLib"
Write-Host ""

# ============================================================
# Environment variables
# ============================================================
$OrtDllPath = Join-Path $DepsLib "onnxruntime.dll"
$EnvFile = Join-Path $ProjectRoot ".env"

@"
ORT_DYLIB_PATH=${OrtDllPath}
PDFIUM_DYNAMIC_LIB_PATH=${DepsLib}
"@ | Set-Content -Path $EnvFile -Encoding UTF8

# Write .cargo/config.toml so cargo commands pick up the env vars automatically
$CargoConfigDir = Join-Path $ProjectRoot ".cargo"
New-Item -ItemType Directory -Force -Path $CargoConfigDir | Out-Null
@"
[env]
ORT_DYLIB_PATH = "${OrtDllPath}"
PDFIUM_DYNAMIC_LIB_PATH = "${DepsLib}"
"@ | Set-Content -Path (Join-Path $CargoConfigDir "config.toml") -Encoding UTF8

Write-Host "==> Environment variables configured"
Write-Host ""
Write-Host "    ORT_DYLIB_PATH=${OrtDllPath}"
Write-Host "    PDFIUM_DYNAMIC_LIB_PATH=${DepsLib}"
Write-Host ""
Write-Host "    Written to .env and .cargo/config.toml"
Write-Host "    cargo build/run will pick these up automatically."
Write-Host ""
Write-Host "==> Setup complete!"
