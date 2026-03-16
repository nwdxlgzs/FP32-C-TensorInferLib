@echo off
setlocal

REM 设置编译器（如果 gcc 不在 PATH 中，请指定完整路径）
set CC=gcc

REM 设置头文件路径和库路径
set INCLUDE_DIR=..\include
set SRC_DIR=..\src

REM 编译命令
echo Compiling test_nn_ops.c ...
%CC% -I%INCLUDE_DIR% -o test_nn_ops.exe test_nn_ops.c %SRC_DIR%\*.c -lm -DDEBUG_TENSOR=1

REM 检查编译是否成功
if errorlevel 1 (
    echo Compilation failed!
    pause
    exit /b %errorlevel%
)

REM 运行测试
echo Running test_nn_ops.exe ...
test_nn_ops.exe

REM 暂停以查看结果（可选）
pause