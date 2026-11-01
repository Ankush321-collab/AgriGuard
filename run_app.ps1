# AgriGuard-ML PowerShell Launcher
Write-Host "========================================================" -ForegroundColor Green
Write-Host "      AgriGuard-ML: Multi-Agent Precision System       " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""

# Check .venv
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[ERROR] .venv\Scripts\python.exe not found!" -ForegroundColor Red
    Write-Host "Please ensure the Python virtual environment exists in this directory."
    Exit 1
}

# Check npm
if (-not (Get-Command "npm" -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] npm was not found in your PATH." -ForegroundColor Red
    Write-Host "Please install Node.js (https://nodejs.org)."
    Exit 1
}

Write-Host "[*] Verifying Python ML environment..." -ForegroundColor Cyan
& .venv\Scripts\python.exe -c "import joblib, sklearn, lightgbm; print('    [OK] Python ML environment verified.')"

Write-Host "[*] Launching Next.js frontend on http://localhost:3000 ..." -ForegroundColor Cyan
Start-Process "http://localhost:3000"

Set-Location "frontend"
npm run dev
