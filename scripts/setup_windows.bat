@echo off
setlocal EnableDelayedExpansion

:: --- Configuration ---
:: Assuming this batch file is in the 'scripts' directory
set "current_script_dir=%~dp0"
:: Go up one level to get the project root
pushd "%current_script_dir%.."
set "project_path=%CD%"
popd

:: Define key paths relative to project root
set "radproc_module_dir=%project_path%\radproc"
set "venv_path=%project_path%\.venv"
set "scripts_dir=%project_path%\scripts" REM Use scripts_dir name consistently
set "requirements_path=%project_path%\requirements.txt"
set "log_dir=%project_path%\log" REM Define log dir path
set "data_dir=%project_path%\data" REM Define data dir (for queue DB)

:: Python module path to run
set "python_module=radproc.cli.main"

:: --- Verification ---
echo Project Root: %project_path%
if not exist "%radproc_module_dir%" (
    echo Error: Module directory not found at: %radproc_module_dir%
    goto :error
)
if not exist "%requirements_path%" (
    echo Error: requirements.txt not found at: %requirements_path%
    goto :error
)

:: --- Setup ---

:: Create essential directories if they don't exist
if not exist "%log_dir%" mkdir "%log_dir%" || (echo Warning: Failed to create log directory & goto :error)
if not exist "%data_dir%" mkdir "%data_dir%" || (echo Warning: Failed to create data directory & goto :error)
if not exist "%scripts_dir%" mkdir "%scripts_dir%" || (echo Warning: Failed to create scripts directory & goto :error)


:: Create virtual environment if it doesn't exist
if not exist "%venv_path%" (
    echo Creating virtual environment at %venv_path%...
    python -m venv "%venv_path%" || (
        echo Error: Failed to create virtual environment
        goto :error
    )
)

:: Activate virtual environment (correct way in batch)
echo Activating virtual environment...
call "%venv_path%\Scripts\activate.bat" || (
    echo Error: Failed to activate virtual environment
    goto :error
)

:: Install/Upgrade requirements (use --upgrade for potential updates)
echo Installing/Updating requirements from %requirements_path%...
pip install --upgrade -r "%requirements_path%" || (
    echo Error: Failed to install requirements
    goto :error
)

:: --- Create Wrapper Script ---
set "wrapper_script_path=%scripts_dir%\frad-proc.bat"
echo Creating wrapper script: %wrapper_script_path%
(
    echo @echo off
    echo :: Wrapper for running the radproc application
    echo :: Activates venv and runs the correct module from project root
    echo.
    echo set "SCRIPT_ROOT_DIR=%~dp0"
    echo set "PROJECT_ROOT=%SCRIPT_ROOT_DIR%.."
    echo set "VENV_PATH=%PROJECT_ROOT%\.venv"
    echo.
    echo :: Activate venv
    echo call "%VENV_PATH%\Scripts\activate.bat"
    echo.
    echo :: Run python module from project root
    echo pushd "%PROJECT_ROOT%"
    echo "%VENV_PATH%\Scripts\python.exe" -m %python_module% %%*
    echo popd
) > "%wrapper_script_path%" || (
    echo Error: Failed to create wrapper script %wrapper_script_path%
    goto :error
)


:: --- Completion ---
echo.
echo Setup complete.
echo Wrapper script created at: %wrapper_script_path%
echo.
echo To use:
echo 1. Set required environment variables (e.g., FTP_PASSWORD_...).
echo 2. Add "%scripts_dir%" to your system PATH variable.
echo 3. Run 'frad-proc run' or 'frad-proc reprocess ...' from any directory.
echo    OR Configure Windows Scheduled Tasks to run "%wrapper_script_path%" run
goto :eof

:error
echo.
echo !!!!! SETUP FAILED !!!!!
pause
exit /b 1

:eof
endlocal