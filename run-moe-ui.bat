@echo off
setlocal

rem Launch the MoE Training UI. Works from any cwd; paths are relative to this script.
set "ROOT=%~dp0"

cd /d "%ROOT%"

rem Prefer the project venv; fall back to the Windows Python launcher.
if exist "%ROOT%.venv\Scripts\python.exe" (
    "%ROOT%.venv\Scripts\python.exe" scripts\moe-training-UI\run.py
    set "ERR=%ERRORLEVEL%"
) else (
    py -3.12 scripts\moe-training-UI\run.py
    set "ERR=%ERRORLEVEL%"
)

if not "%ERR%"=="0" (
    echo.
    echo [error] MoE Training UI exited with code %ERR%.
    pause
)
endlocal
