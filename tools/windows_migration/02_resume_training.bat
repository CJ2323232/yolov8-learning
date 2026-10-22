@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist ".venv\Scripts\python.exe" (
  echo Run 01_install_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -B resume_windows.py
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" echo Training stopped. Keep the error visible; do not start a fresh run.
pause
exit /b %TRAIN_EXIT%
