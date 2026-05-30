# setup.ps1 - Create venv and install pipeline dependencies
# Usage: .\setup.ps1

$ErrorActionPreference = "Stop"
$venvDir = ".venv"

if (Test-Path $venvDir) {
    Write-Host "[OK] .venv already exists" -ForegroundColor Green
} else {
    Write-Host "[..] Creating .venv" -ForegroundColor Yellow
    python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv" }
    Write-Host "[OK] .venv created" -ForegroundColor Green
}

$activate = Join-Path $venvDir "Scripts\Activate.ps1"
& $activate

Write-Host "[..] Upgrading pip" -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "[..] Installing dependencies from requirements.txt" -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements" }

Write-Host "`n[DONE] Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "       Run pipeline:  python run_all.py pdf_idk.pdf --skip 3 4" -ForegroundColor Cyan
