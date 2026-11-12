@echo off
chcp 65001 >nul
setlocal
set "PYTORCH_ENV=%USERPROFILE%\.conda\envs\aiinfra-pytorch"
if not exist "%PYTORCH_ENV%\python.exe" (
    echo 未找到 aiinfra-pytorch 环境。
    pause
    exit /b 1
)
set "PATH=%PYTORCH_ENV%;%PYTORCH_ENV%\Library\bin;%PYTORCH_ENV%\Scripts;%PATH%"
pushd "%~dp0docs\guides\模块一-前置知识\pyroch"
if errorlevel 1 (
    echo 无法进入 Notebook 目录。
    pause
    exit /b 1
)
"%PYTORCH_ENV%\python.exe" -m jupyterlab "PyTorch实用入门-AIInfra版.ipynb" --ServerApp.ip=127.0.0.1
set "JUPYTER_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%JUPYTER_EXIT_CODE%"=="0" pause
exit /b %JUPYTER_EXIT_CODE%
