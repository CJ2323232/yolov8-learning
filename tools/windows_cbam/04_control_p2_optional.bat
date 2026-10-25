@echo off
echo This starts an OPTIONAL matched P2 control run. It does not contain CBAM.
echo It is not required to start the P2+CBAM run. Close this window to cancel.
choice /C YN /N /M "Start a NEW 50-epoch P2 control run? [Y/N] "
if errorlevel 2 exit /b 0
call "%~dp0run_windows.bat" --control-p2
