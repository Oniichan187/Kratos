@echo off
REM ============================================================================
REM  Kratos one-shot installer (Windows).  Right-click -> "Run as administrator"
REM  (it will self-elevate if you don't).
REM
REM  Prepares EVERYTHING:
REM    1. WSL2 + Ubuntu
REM    2. checks the NVIDIA Windows driver (CUDA for WSL comes from it)
REM    3. Python 3 on Windows + Kratos Python dependencies
REM    4. inside WSL: Ollama (GPU/CUDA) + downloads all abliterated models
REM    5. builds the 'kratos-planner' compressor model + saves the config
REM
REM  Idempotent: safe to run again, it only fixes what is missing.
REM ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
title Kratos Installer

REM ---- self-elevate to Administrator -----------------------------------------
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo.
echo ============================================================
echo   KRATOS INSTALLER
echo   Repo: %REPO%
echo ============================================================
echo.

REM ---- 1. WSL2 ----------------------------------------------------------------
echo [1/5] Checking WSL2...
wsl --status >nul 2>&1
if %errorlevel% neq 0 (
    echo   WSL is not installed. Installing WSL2 + Ubuntu...
    wsl --install -d Ubuntu
    echo.
    echo   ============================================================
    echo   A RESTART IS REQUIRED to finish installing WSL2.
    echo   After rebooting, Ubuntu will open once to create your Linux
    echo   user/password. Then run this install.bat again.
    echo   ============================================================
    pause
    exit /b
)
echo   Making sure the kernel is current...
wsl --update >nul 2>&1
REM Ensure a usable Ubuntu distro exists
wsl -l -q 2>nul | findstr /i "Ubuntu" >nul
if %errorlevel% neq 0 (
    echo   Installing the Ubuntu distribution...
    wsl --install -d Ubuntu
    echo   If Ubuntu just opened to set up a username/password, finish that,
    echo   close it, then run install.bat again.
    pause
    exit /b
)
echo   OK: WSL2 ready.
echo.

REM ---- 2. NVIDIA driver (CUDA for WSL comes from the Windows driver) ----------
echo [2/5] Checking NVIDIA GPU driver...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo   OK: GPU driver present. WSL/Ollama will use CUDA.
) else (
    echo   WARNING: nvidia-smi not found on Windows.
    echo   Install the NVIDIA "Game Ready" or "Studio" driver ^(^>= 535^) from
    echo   https://www.nvidia.com/download/index.aspx  then re-run this script.
    echo   ^(Kratos still runs on CPU, just slower.^)
)
echo.

REM ---- 3. Python on Windows + dependencies -----------------------------------
echo [3/5] Checking Python on Windows...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo   Python not found. Installing via winget...
    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
        echo   Python installed. Please CLOSE this window and run install.bat again
        echo   so the new PATH takes effect.
        pause
        exit /b
    ) else (
        echo   winget is unavailable. Install Python 3.10+ from https://www.python.org/downloads/
        echo   ^(tick "Add python.exe to PATH"^), then re-run install.bat.
        pause
        exit /b
    )
)
echo   Installing Kratos Python dependencies...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r "%REPO%\requirements.txt"
if %errorlevel% neq 0 (
    echo   ERROR: pip install failed. See the output above.
    pause
    exit /b
)
echo   OK: Python dependencies installed.
echo.

REM ---- 4. WSL provisioning: Ollama + CUDA + model downloads -------------------
echo [4/5] Provisioning WSL ^(Ollama + CUDA + downloading models^)...
echo   This downloads several GB of models and can take a while.
for /f "usebackq delims=" %%i in (`wsl wslpath -a "%REPO%"`) do set "REPO_WSL=%%i"
wsl -e bash -lc "cd '%REPO_WSL%' && bash ./setup_wsl.sh"
if %errorlevel% neq 0 (
    echo   ERROR: WSL provisioning failed. See the output above.
    pause
    exit /b
)
echo   OK: WSL provisioning complete.
echo.

REM ---- 5. Build compressor model + save config -------------------------------
echo [5/5] Building the 'kratos-planner' compressor and saving config...
python "%REPO%\setup_models.py"
if %errorlevel% neq 0 (
    echo   WARNING: setup_models.py reported an issue. Check the output above.
)
echo.

echo ============================================================
echo   DONE. Start Kratos from any project folder with:  kratos
echo        ^(or:  python "%REPO%\kratos.py"^)
echo ============================================================
echo.
pause
endlocal
