@echo off
setlocal

set CC=gcc
set INCLUDE_DIR=..\include
set SRC_DIR=..\src

for %%f in (test_*.c) do (
    echo Compiling %%f ...
    %CC% -I%INCLUDE_DIR% -o %%~nf.exe %%f %SRC_DIR%\*.c -lm -DDEBUG_TENSOR=1
    if errorlevel 1 (
        echo Failed to compile %%f
        pause
        exit /b 1
    )
)
echo All tests compiled successfully.
pause