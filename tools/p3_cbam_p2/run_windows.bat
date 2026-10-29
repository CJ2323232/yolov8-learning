@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
if defined YOLO_PYTHON goto selected
py -3.10 --version >nul 2>&1
if errorlevel 1 goto fallback
py -3.10 -B "%~dp0launch.py" %*
goto result
:selected
"%YOLO_PYTHON%" -B "%~dp0launch.py" %*
goto result
:fallback
python -B "%~dp0launch.py" %*
:result
if errorlevel 1 goto failed
echo Completed. Keep previous experiment files.
pause
exit /b 0
:failed
echo Stopped. Read the error and keep this window open. Do not change batch or parameters to bypass it.
pause
exit /b 1
