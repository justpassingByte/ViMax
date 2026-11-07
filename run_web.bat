@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
title ViMax Studio - Web UI
echo ===================================================
echo     KHOI DONG VIMAX STUDIO - XUONG PHIM AI
echo ===================================================
echo.

cd /d "%~dp0\web"
echo Dang khoi dong Web server tai http://127.0.0.1:4173 ...
npm run dev
pause
