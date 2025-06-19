@echo off
REM Simple startup script for RAG Evaluation Pipeline on Windows
REM This script sets up and runs the pipeline without Docker

echo.
echo ========================================
echo  RAG Evaluation Pipeline - Local Setup
echo ========================================
echo.

REM Change to the pipeline directory
cd /d "%~dp0"
echo Working directory: %CD%

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found!
python --version

echo.
echo Starting automated setup...
echo.

REM Run the Python setup script
python start_local.py

echo.
echo ========================================
echo Setup complete!
echo.
echo To run the pipeline manually:
echo   python run_pipeline.py --config config/simple_config.yaml
echo.
echo To generate testset only:
echo   python generate_dataset_configurable.py
echo.
echo Check the 'outputs' folder for results
echo ========================================
echo.

pause
