@echo off
setlocal

rem Launch Cfish in a new terminal. Works from any cwd; paths are relative to this script.
set "ROOT=%~dp0"
set "SRC=%ROOT%src\cfish"
set "EXE=%SRC%\cfish.exe"

if not exist "%EXE%" (
    echo [error] cfish.exe not found:
    echo   %EXE%
    echo.
    echo Build it first:
    echo   cd src\cfish
    echo   make build numa=no ARCH=x86-64
    echo.
    pause
    exit /b 1
)

if not exist "%SRC%\nn-62ef826d1a6d.nnue" (
    echo [warn] NNUE file not found next to cfish.exe:
    echo   %SRC%\nn-62ef826d1a6d.nnue
    echo Hybrid/Pure eval may be unavailable. Run "make net" in src\cfish if needed.
    echo.
)

rem New console, stay in src\cfish so EvalFile resolves; /k keeps the window after quit.
start "Cfish" /D "%SRC%" cmd /k cfish.exe
endlocal
