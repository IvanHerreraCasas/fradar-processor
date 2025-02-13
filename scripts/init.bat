@echo off
setlocal EnableDelayedExpansion

:: Set up paths
set "rel_project_path=..\.."
set "project_path=%~dp0%rel_project_path%"

set "radproc_path=%project_path%radproc"
set "cli_path=%radproc_path%\cli\main.py"
set "venv_path=%project_path%\.venv"
set "scripts_path=%project_path%\scripts"

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
pip install -r requirements.txt || (
    echo Error: Failed to install requirements
    exit /b 1
)

:: Create the frad-proc.bat script
(
    echo @echo off
    echo "%venv_path%\Scripts\python.exe" "%cli_path%" %%*
) > "%scripts_path%\frad-proc.bat"

:: Check if we have admin rights for permanent PATH modification
net session >nul 2>&1
set "is_admin=%errorlevel%"

echo Would you like to add frad-proc to PATH? This will allow you to run it from anywhere.
echo 1. Yes - Add permanently (requires admin rights)
echo 2. Yes - Add temporarily (for current session only)
echo 3. No - Skip this step
set /p "choice=Enter your choice (1-3): "

if "%choice%"=="1" (
    if "%is_admin%"=="0" (
        :: Add to system PATH permanently
        for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path') do set "current_path=%%b"
        
        :: Check if path is already in PATH
        echo %current_path% | find /i "%scripts_path%" > nul
        if errorlevel 1 (
            setx /M PATH "%current_path%;%scripts_path%" || (
                echo Error: Failed to modify system PATH
                exit /b 1
            )
            echo Successfully added to system PATH. Please restart your command prompt to use frad-proc globally.
        ) else (
            echo Path is already in system PATH.
        )
    ) else (
        echo Error: Administrator rights required for permanent PATH modification.
        echo Please run this script as administrator or choose the temporary option.
        exit /b 1
    )
) else if "%choice%"=="2" (
    :: Add to PATH temporarily
    set "PATH=%PATH%;%scripts_path%"
    echo Successfully added to PATH for current session.
    echo You can now use frad-proc from anywhere in this command prompt window.
) else (
    echo Skipping PATH modification.
)


:: Run the commands with error checking
call "%scripts_path%\frad-proc.bat" enable || (
    echo Error: Failed to run frad-proc enable
    exit /b 1
)

call "%scripts_path%\frad-proc.bat" run || (
    echo Error: Failed to run frad-proc run
    exit /b 1
)

echo Script completed successfully
endlocal