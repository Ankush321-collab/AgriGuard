@echo off
title AgriGuard-ML - Precision Apple Leaf Disease Detection
echo ========================================================
echo       AgriGuard-ML: Multi-Agent Precision System
echo ========================================================
echo.

:: Verify Python Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found in current folder!
    echo Please make sure .venv exists in this directory.
    pause
    exit /b 1
)

:: Verify Node.js / npm
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] npm was not found in your PATH.
    echo Please install Node.js (https://nodejs.org) to run the frontend.
    pause
    exit /b 1
)

echo [*] Testing Python multi-agent ML backend...
.venv\Scripts\python.exe -c "import joblib, sklearn, lightgbm; print('    [OK] Python ML environment verified.')"

echo [*] Starting AgriGuard Next.js Frontend Server...
cd frontend

:: Open browser after 2 seconds
start "" http://localhost:3000

:: Start Next.js development server
npm run dev
