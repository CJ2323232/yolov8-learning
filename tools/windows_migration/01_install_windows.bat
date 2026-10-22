@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
echo Create a fresh Windows Python environment. Do not copy the Mac .venv.
if exist ".venv\Scripts\python.exe" goto install
py -3.14 --version >nul 2>&1
if errorlevel 1 goto try312
py -3.14 -m venv .venv
if errorlevel 1 goto failed
goto install
:try312
py -3.12 --version >nul 2>&1
if errorlevel 1 goto nopython
py -3.12 -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-windows.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -B resume_windows.py --check
if errorlevel 1 goto failed
echo Ready. Installation did not start training. Next: double-click 02_resume_training.bat
pause
exit /b 0
:nopython
echo Install 64-bit Python 3.14 from https://www.python.org/downloads/windows/
echo Include the Python launcher, then run this file again.
pause
exit /b 1
:failed
echo Setup/check failed. Keep this window open and share the last error.
echo If CUDA is unavailable, update the NVIDIA driver and check the CUDA PyTorch installation.
pause
exit /b 1
