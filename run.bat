@echo off
echo ===================================================
echo   Handwritten Digit Recognizer - Launcher
echo ===================================================
echo.
echo Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH!
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo.
echo Installing dependencies from requirements.txt (using CPU-only PyTorch)...
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

if %errorlevel% neq 0 (
    echo.
    echo Warning: Failed to install with extra-index-url. Retrying standard installation...
    python -m pip install -r requirements.txt
)

echo.
echo Launching Streamlit web application...
python -m streamlit run app.py
pause
