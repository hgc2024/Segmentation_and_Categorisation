@echo off
echo ===================================================
echo Setting up Python Environment for DiffusionDet
echo ===================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Create Virtual Environment
if not exist "venv" (
    echo Creating virtual environment 'venv'...
    python -m venv venv
) else (
    echo Virtual environment 'venv' already exists.
)

REM Activate Environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA 12.1 support (Recommended for RTX 40 series)
echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install other requirements
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Warning: requirements.txt not found! Installing basics manually...
    pip install numpy opencv-python tqdm scipy matplotlib pillow
)

echo.
echo ===================================================
echo Setup Complete!
echo ===================================================
echo To run the diffusion detector:
echo 1. call venv\Scripts\activate
echo 2. python Source/diffusion_detector.py
echo.
pause
