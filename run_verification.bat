@echo off
REM Verification script runner for Windows
REM This ensures the script runs in the virtual environment

echo ======================================================================
echo FACE DETECTION MODULE VERIFICATION
echo ======================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then run: venv\Scripts\activate
    echo Then run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment and run verification
call venv\Scripts\activate.bat
python verify_face_detection.py

echo.
pause
