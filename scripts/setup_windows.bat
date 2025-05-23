@echo off
setlocal EnableDelayedExpansion

:: --- Configuration ---
set "current_script_dir=%~dp0"
pushd "%current_script_dir%.."
set "project_path=%CD%"
popd

:: Define environment path (using prefix)
set "env_prefix=%project_path%\.venv"
set "env_file=%project_path%\environment.yml"
set "scripts_dir=%project_path%\scripts"
set "log_dir=%project_path%\log"
set "data_dir=%project_path%\data"
set "cache_dir=%project_path%\cache"
set "anim_tmp_dir=%cache_dir%\animation_tmp"
set "anim_output_dir=%project_path%\animations"

:: --- Verification ---
echo Project Root: %project_path%
:: Check if conda command exists
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: 'conda' command not found in PATH.
    echo Please install Anaconda or Miniconda and add it to your PATH.
    goto :error
)
if not exist "%env_file%" (
    echo ERROR: environment.yml not found at: %env_file%
    goto :error
)
echo Conda environment file verified: %env_file%

:: --- Setup ---
echo Ensuring required directories exist...
if not exist "%log_dir%" mkdir "%log_dir%" || (echo Warning: Failed to create log directory & goto :error)
if not exist "%data_dir%" mkdir "%data_dir%" || (echo Warning: Failed to create data directory & goto :error)
if not exist "%scripts_dir%" mkdir "%scripts_dir%" || (echo Warning: Failed to create scripts directory & goto :error)
if not exist "%cache_dir%" mkdir "%cache_dir%" || (echo Warning: Failed to create cache directory & goto :error)
if not exist "%anim_tmp_dir%" mkdir "%anim_tmp_dir%" || (echo Warning: Failed to create animation temp directory & goto :error)
if not exist "%anim_output_dir%" mkdir "%anim_output_dir%" || (echo Warning: Failed to create animation output directory & goto :error)

:: Remove existing environment directory if it exists
if exist "%env_prefix%\" (
    echo Removing existing environment directory: %env_prefix%
    rmdir /S /Q "%env_prefix%" || (echo WARNING: Failed to remove existing .venv directory.)
)

:: Create Conda environment using the environment.yml file and prefix
echo Creating Conda environment at %env_prefix% from %env_file%...
conda env create --prefix "%env_prefix%" --file "%env_file%" || (
    echo ERROR: Failed to create Conda environment. Check %env_file%.
    goto :error
)
echo Conda environment created successfully.

:: --- Apply Huey DB PRAGMAs (Requires the environment) ---
set "PRAGMA_SCRIPT=%project_path%\scripts\apply_pragmas.py"
if exist "%PRAGMA_SCRIPT%" (
    echo Attempting to apply Huey DB PRAGMAs...
    conda run -p "%env_prefix%" python "%PRAGMA_SCRIPT%" || (echo Warning: Failed to apply PRAGMAs automatically.)
) else (
    echo Optional: Create scripts\apply_pragmas.py to automatically configure Huey DB.
)


:: --- Create Wrapper Scripts using conda run ---
echo Creating wrapper scripts...

:: Wrapper for CLI ('frad-proc.bat')
set "WRAPPER_CLI_PATH=%scripts_dir%\frad-proc.bat"
set "PYTHON_MODULE_CLI=radproc.cli.main"
echo Creating: %WRAPPER_CLI_PATH%
(
    echo @echo off
    echo setlocal
    echo set "SCRIPT_ROOT_DIR=%%~dp0"
    echo set "PROJECT_ROOT=%%SCRIPT_ROOT_DIR%%.."
    echo set "ENV_PREFIX=%%PROJECT_ROOT%%\.venv"
    echo call conda run -p "%%ENV_PREFIX%%" python -m %PYTHON_MODULE_CLI% %%*
    echo endlocal
) > "%WRAPPER_CLI_PATH%" || (echo ERROR: Failed to create CLI wrapper & goto :error)

:: Wrapper for API ('run-api.bat')
set "WRAPPER_API_PATH=%scripts_dir%\run-api.bat"
set "PYTHON_MODULE_API=radproc.api.main:app"
set "DEFAULT_HOST_API=127.0.0.1"
set "DEFAULT_PORT_API=8001"
echo Creating: %WRAPPER_API_PATH%
(
    echo @echo off
    echo setlocal
    echo set "SCRIPT_ROOT_DIR=%%~dp0"
    echo set "PROJECT_ROOT=%%SCRIPT_ROOT_DIR%%.."
    echo set "ENV_PREFIX=%%PROJECT_ROOT%%\.venv"
    echo echo Starting Uvicorn in %%ENV_PREFIX%%...
    echo call conda run -p "%%ENV_PREFIX%%" python -m uvicorn %PYTHON_MODULE_API% --host %DEFAULT_HOST_API% --port %DEFAULT_PORT_API% %%*
    echo endlocal
) > "%WRAPPER_API_PATH%" || (echo ERROR: Failed to create API wrapper & goto :error)

:: Wrapper for Worker ('run-worker.bat')
set "WRAPPER_WORKER_PATH=%scripts_dir%\run-worker.bat"
set "HUEY_INSTANCE=radproc.huey_config.huey"
set "WORKER_TYPE=thread"
set "WORKER_COUNT=2"
set "LOG_FILE_WORKER=%log_dir%\huey_worker.log"
set "LOG_FILE_FLAG=-l "%LOG_FILE_WORKER%""
set "LOG_LEVEL_FLAG=-q"
echo Creating: %WRAPPER_WORKER_PATH%
(
    echo @echo off
    echo setlocal
    echo set "SCRIPT_ROOT_DIR=%%~dp0"
    echo set "PROJECT_ROOT=%%SCRIPT_ROOT_DIR%%.."
    echo set "ENV_PREFIX=%%PROJECT_ROOT%%\.venv"
    echo echo Starting Huey Consumer in %%ENV_PREFIX%%...
    echo call conda run -p "%%ENV_PREFIX%%" python -m huey_consumer %HUEY_INSTANCE% -k %WORKER_TYPE% -w %WORKER_COUNT% %LOG_LEVEL_FLAG% %LOG_FILE_FLAG% %%*
    echo endlocal
) > "%WRAPPER_WORKER_PATH%" || (echo ERROR: Failed to create worker wrapper & goto :error)


:: --- Completion ---
echo.
echo Setup complete (using Conda).
echo Environment created at: %env_prefix%
echo Wrapper scripts created in: %scripts_dir%
echo.
echo To use:
echo 1. Set required environment variables (e.g., FTP_PASSWORD_...).
echo 2. Add "%scripts_dir%" to your system PATH variable (optional).
echo 3. Run commands using wrappers from any directory (if PATH is set) or from scripts dir:
echo    frad-proc run
echo    run-api --port 8000
echo    run-worker -w 1 -v
echo 4. For services (NSSM/Task Scheduler): Configure them to run the .bat wrapper scripts.
goto :eof

:error
echo.
echo !!!!! SETUP FAILED !!!!!
pause
exit /b 1

:eof
endlocal