@echo off
chcp 65001 >nul
title Batscribe - captioning
if "%~1"=="" (
  echo Drag a course folder ^(or video files^) onto this file.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0batscribe.py" %*
echo.
pause
