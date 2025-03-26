@echo off
setlocal EnableDelayedExpansion

:: Set up paths
set "rel_project_path=..\"

set "current_script_dir=%~dp0"

pushd "%current_script_dir%"
pushd "%rel_project_path%"
set "project_path=%CD%"
popd
popd

set "radproc_path=%project_path%\radproc"
set "cli_path=%radproc_path%\cli\main.py"
set "venv_path=%project_path%\.venv"
set "scripts_path=%project_path%\scripts"
set "requirements_path=%project_path%\requirements.txt"

:: Verify directories exist
if not exist "%radproc_path%" (
    echo Error: radproc directory not found at: %radproc_path%
    exit /b 1
)

:: Change to radproc directory
cd /d "%radproc_path%" || (
    echo Error: Failed to change to directory: %radproc_path%
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "%venv_path%" (
    echo Creating virtual environment...
    python -m venv "%venv_path%" || (
        echo Error: Failed to create virtual environment
        exit /b 1
    )
)

:: Create scripts directory if it doesn't exist
if not exist "%scripts_path%" mkdir "%scripts_path%"

:: Activate virtual environment (correct way in batch)
call "%venv_path%\Scripts\activate.bat" || (
    echo Error: Failed to activate virtual environment
    exit /b 1
)

:: Install requirements
pip install -r %requirements_path% || (
    echo Error: Failed to install requirements
    exit /b 1
)

:: Create the frad-proc.bat script
(
    echo @echo off
    echo "%venv_path%\Scripts\python.exe" "%cli_path%" %%*
) > "%scripts_path%\frad-proc.bat"


echo frad-proc is ready to be used. Add the scripts directory to your PATH to use the CLI tool globally.
endlocal