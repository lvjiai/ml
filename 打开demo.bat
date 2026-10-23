@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

rem Double-click this file to start the demo.
rem Checks Python + dependencies, picks a free port, opens the browser,
rem and runs Streamlit in this window. Close the window to stop.

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
    where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
    echo.
    echo [ERROR] Python not found. Install Python 3.10+ and tick "Add to PATH".
    echo.
    pause
    exit /b 1
)

%PY% -c "import streamlit, lightgbm, sklearn, pandas, numpy, torch" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Missing dependencies. Run from the project folder:
    echo     %PY% -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

set "PORT=8501"
%PY% -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1',8501)); s.close(); sys.exit(0 if r==0 else 1)" >nul 2>nul
if not errorlevel 1 set "PORT=8502"
%PY% -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1',8502)); s.close(); sys.exit(0 if r==0 else 1)" >nul 2>nul
if not errorlevel 1 set "PORT=8503"

if "%~1"=="--check" (
    echo OK: %PY%
    echo PORT: %PORT%
    exit /b 0
)

if not defined DEMO_NO_BROWSER (
    start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 6; Start-Process 'http://127.0.0.1:%PORT%'"
)

echo.
echo Demo: http://127.0.0.1:%PORT%
echo Close this window (or press Ctrl+C) to stop the server.
echo.
%PY% -m streamlit run app\streamlit_app.py --server.headless true --server.port %PORT% --browser.gatherUsageStats false

echo.
echo Server stopped.
pause
