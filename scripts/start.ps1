param(
  [string]$Port = "8000",
  [string]$DataDir = ""
)

$ErrorActionPreference = "Stop"

# Move to script root
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -Path (Resolve-Path "..")

if (!(Test-Path .venv)) {
  py -m venv .venv
}

# Activate venv
$venv = Join-Path (Get-Location) ".venv/Scripts/Activate.ps1"
. $venv

python -m pip install --upgrade pip setuptools wheel cmake
pip install -r requirements.txt

if ($DataDir -ne "") {
  $env:SMARTATTENDANCE_DATA_DIR = $DataDir
}

python -m uvicorn web.main:app --host 0.0.0.0 --port $Port