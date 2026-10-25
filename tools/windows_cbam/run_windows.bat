@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0\..\.."
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
if defined YOLO_PYTHON goto selected
if exist ".venv\Scripts\python.exe" (
  set "YOLO_PYTHON=%CD%\.venv\Scripts\python.exe"
  goto selected
)
if exist "venv\Scripts\python.exe" (
  set "YOLO_PYTHON=%CD%\venv\Scripts\python.exe"
  goto selected
)
set "YOLO_PYTHON=python"
:selected
echo Python: %YOLO_PYTHON%
echo Use the SAME environment that trained the previous P2. No packages will be upgraded.
"%YOLO_PYTHON%" -B train_p2_cbam.py %*
if errorlevel 1 goto failed
echo Completed successfully.
pause
exit /b 0
:failed
echo Stopped. Do not delete old results or change training parameters to bypass this error.
echo Keep this window open and share the error above.
pause
exit /b 1
