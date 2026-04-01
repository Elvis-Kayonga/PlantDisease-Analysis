@echo off
REM Quick start script for Plant Disease Detection System (Windows)

setlocal enabledelayedexpansion
title Plant Disease Detection System - Setup

echo.
echo ============================================
echo Plant Disease Detection System - Setup ^& Run
echo ============================================
echo.

REM Colors using ANSI codes
for /F %%A in ('echo prompt $E ^| cmd') do set "ESC=%%A"
set "GREEN=%ESC%[32m"
set "BLUE=%ESC%[34m"
set "YELLOW=%ESC%[33m"
set "NC=%ESC%[0m"

REM Check Python
echo %BLUE%Checking Python installation...%NC%
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3 is required but not installed.
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set "PYTHON_VERSION=%%i"
echo %GREEN%✓ %PYTHON_VERSION%%NC%

REM Create virtual environment
echo.
echo %BLUE%Setting up virtual environment...%NC%
if not exist "venv" (
    python -m venv venv
    echo %GREEN%✓ Virtual environment created%NC%
) else (
    echo %GREEN%✓ Virtual environment already exists%NC%
)

REM Activate virtual environment
call venv\Scripts\activate.bat
echo %GREEN%✓ Virtual environment activated%NC%

REM Install dependencies
echo.
echo %BLUE%Installing dependencies...%NC%
pip install --upgrade pip setuptools wheel > nul 2>&1
pip install -r requirements.txt

echo %GREEN%✓ Dependencies installed%NC%

REM Create necessary directories
echo.
echo %BLUE%Creating project directories...%NC%
if not exist "data\train" mkdir data\train
if not exist "data\validation" mkdir data\validation
if not exist "data\test" mkdir data\test
if not exist "models" mkdir models
if not exist "uploaded_data" mkdir uploaded_data
if not exist "logs" mkdir logs
echo %GREEN%✓ Directories created%NC%

REM Menu
echo.
echo %YELLOW%What would you like to do?%NC%
echo 1) Run Jupyter Notebook (Training)
echo 2) Start API Backend (FastAPI)
echo 3) Start Web UI (React)
echo 4) Run Both API and UI
echo 5) Run Load Tests (Locust)
echo 6) Docker Deployment
echo 7) Exit
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo %BLUE%Starting Jupyter Notebook...%NC%
    echo Open: http://localhost:8888
    jupyter notebook notebook\plant_disease_detection.ipynb
) else if "%choice%"=="2" (
    echo %BLUE%Starting FastAPI Backend...%NC%
    echo API running at: http://localhost:8000
    echo API docs at: http://localhost:8000/docs
    echo.
    python main.py
) else if "%choice%"=="3" (
    echo %BLUE%Starting React UI...%NC%
    echo UI running at: http://localhost:5173
    echo.
    echo %YELLOW%Note: Make sure API is running on another terminal!%NC%
    echo.
    cd frontend
    call npm install
    call npm run dev -- --host 0.0.0.0 --port 5173
) else if "%choice%"=="4" (
    echo %BLUE%Starting API and UI...%NC%
    echo.
    echo %YELLOW%Starting API in background...%NC%
    start "Plant Disease API" cmd /k python main.py
    
    echo Waiting for API to start...
    timeout /t 3 /nobreak
    
    echo.
    echo %BLUE%Starting React UI...%NC%
    echo UI running at: http://localhost:5173
    cd frontend
    call npm install
    call npm run dev -- --host 0.0.0.0 --port 5173
) else if "%choice%"=="5" (
    echo %BLUE%Starting Locust Load Testing...%NC%
    echo.
    echo %YELLOW%Make sure API is running on http://localhost:8000%NC%
    echo.
    echo Locust UI running at: http://localhost:8089
    locust -f locustfile.py --host=http://localhost:8000 --web --port=8089
) else if "%choice%"=="6" (
    echo %BLUE%Docker Deployment Options:%NC%
    echo.
    echo 1) Build Docker image
    echo 2) Run single container
    echo 3) Run with docker-compose
    echo.
    set /p docker_choice="Enter docker choice (1-3): "
    
    if "!docker_choice!"=="1" (
        echo %BLUE%Building Docker image...%NC%
        docker build -t plant-disease-detector:latest .
        echo %GREEN%✓ Docker image built%NC%
    ) else if "!docker_choice!"=="2" (
        echo %BLUE%Running Docker container...%NC%
        docker run -p 8000:8000 -p 5173:5173 ^
            -v %cd%\models:/app/models ^
            -v %cd%\uploaded_data:/app/uploaded_data ^
            plant-disease-detector:latest
    ) else if "!docker_choice!"=="3" (
        echo %BLUE%Starting docker-compose...%NC%
        docker-compose up
    ) else (
        echo Invalid choice
    )
) else if "%choice%"=="7" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice
    exit /b 1
)

echo.
echo %GREEN%Done!%NC%
pause
