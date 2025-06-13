@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION

ECHO --- Starting RadProc Application Setup (Windows - Conda) ---
ECHO IMPORTANT: This script assumes PostgreSQL database and user have ALREADY been set up manually.
ECHO Ensure your PostgreSQL server is running, a database and user for RadProc exist,
ECHO and you have updated 'config\database_config.yaml' and set the RADPROC_DB_PASSWORD environment variable.
ECHO.
CHOICE /C YN /M "Proceed with RadProc application setup?"
IF ERRORLEVEL 2 (
    ECHO Application setup aborted by user.
    EXIT /B 0
)
IF ERRORLEVEL 1 (
    ECHO.
)

REM --- Determine Paths ---
SET "SCRIPT_DIR=%~dp0"
FOR %%A IN ("%SCRIPT_DIR%..") DO SET "PROJECT_ROOT=%%~fA"

REM Define key paths relative to project root
SET "RADPROC_MODULE_DIR=%PROJECT_ROOT%\radproc"
SET "REQUIREMENTS_PATH=%PROJECT_ROOT%\requirements.txt"
SET "ENVIRONMENT_YAML_PATH=%PROJECT_ROOT%\environment.yml"
SET "LOG_DIR=%PROJECT_ROOT%\log"
SET "DATA_DIR=%PROJECT_ROOT%\data"
SET "CACHE_DIR=%PROJECT_ROOT%\cache"
SET "ANIM_TMP_DIR=%CACHE_DIR%\animation_tmp"
SET "ANIM_OUTPUT_DIR=%PROJECT_ROOT%\animations"
SET "API_JOB_OUTPUT_DIR=%CACHE_DIR%\api_job_outputs"
SET "PYTHON_MODULE=radproc.cli.main"

REM --- Verification ---
ECHO Verifying prerequisites...
IF NOT EXIST "%RADPROC_MODULE_DIR%" (
    ECHO Error: Module directory not found at: "%RADPROC_MODULE_DIR%"
    EXIT /B 1
)
IF NOT EXIST "%ENVIRONMENT_YAML_PATH%" (
    ECHO Error: environment.yml not found at: "%ENVIRONMENT_YAML_PATH%"
    EXIT /B 1
)
WHERE conda >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error: 'conda' command not found. Please install Anaconda or Miniconda and add to PATH.
    EXIT /B 1
)
ECHO Prerequisites verified.

REM --- Setup ---

REM Create essential application directories if they don't exist
ECHO Ensuring required application directories exist...
IF NOT EXIST "%LOG_DIR%" MKDIR "%LOG_DIR%"
IF NOT EXIST "%DATA_DIR%" MKDIR "%DATA_DIR%"
IF NOT EXIST "%CACHE_DIR%" MKDIR "%CACHE_DIR%"
IF NOT EXIST "%ANIM_TMP_DIR%" MKDIR "%ANIM_TMP_DIR%"
IF NOT EXIST "%ANIM_OUTPUT_DIR%" MKDIR "%ANIM_OUTPUT_DIR%"
IF NOT EXIST "%API_JOB_OUTPUT_DIR%" MKDIR "%API_JOB_OUTPUT_DIR%"
ECHO Application directories ensured.

REM Create/Update Conda environment
ECHO Creating/Updating Conda environment from "%ENVIRONMENT_YAML_PATH%"...
REM Extract env name from YAML (simple parsing, might need more robust if comments exist before name)
FOR /F "tokens=2 delims=: " %%G IN ('findstr /B /C:"name:" "%ENVIRONMENT_YAML_PATH%"') DO SET "CONDA_ENV_NAME=%%G"

IF "%CONDA_ENV_NAME%"=="" (
    ECHO Error: Could not determine Conda environment name from environment.yml.
    ECHO Please ensure it has a 'name: your_env_name' line.
    SET CONDA_ENV_NAME=frad-proc
    ECHO Using default name '%CONDA_ENV_NAME%'
)

REM Check if environment exists
CALL conda env list | findstr /B /C:"%CONDA_ENV_NAME% " >nul
IF %ERRORLEVEL% EQU 0 (
    ECHO Environment '%CONDA_ENV_NAME%' already exists. Updating...
    CALL conda env update --name "%CONDA_ENV_NAME%" -f "%ENVIRONMENT_YAML_PATH%" --prune
    IF !ERRORLEVEL! NEQ 0 (
        ECHO Error: Failed to update Conda environment '%CONDA_ENV_NAME%'.
        EXIT /B 1
    )
) ELSE (
    ECHO Creating new Conda environment '%CONDA_ENV_NAME%'...
    CALL conda env create -f "%ENVIRONMENT_YAML_PATH%"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO Error: Failed to create Conda environment '%CONDA_ENV_NAME%'.
        EXIT /B 1
    )
)
ECHO Conda environment '%CONDA_ENV_NAME%' is ready.

REM --- Create Wrapper Script (frad-proc.bat) ---
SET "WRAPPER_SCRIPT_PATH=%SCRIPT_DIR%frad-proc.bat"
ECHO Creating wrapper script: "%WRAPPER_SCRIPT_PATH%"

(
ECHO @ECHO OFF
ECHO SETLOCAL
ECHO REM Wrapper script for the RadProc application (Conda environment^)
ECHO.
ECHO SET "PYTHON_MODULE=%PYTHON_MODULE%"
ECHO SET "CONDA_ENV_NAME=%CONDA_ENV_NAME%"
ECHO SET "PROJECT_ROOT_IN_WRAPPER=%%~dp0.."
ECHO.
ECHO REM Try to activate Conda environment.
ECHO REM This assumes conda is in PATH and the shell is initialized for conda.
ECHO CALL conda activate %CONDA_ENV_NAME%
ECHO IF ERRORLEVEL 1 ^(
ECHO     ECHO Error: Failed to activate Conda environment '%CONDA_ENV_NAME%'.
ECHO     ECHO Please activate it manually: conda activate %CONDA_ENV_NAME%
ECHO     PAUSE
ECHO     EXIT /B 1
ECHO ^)
ECHO.
ECHO REM Execute the Python module
ECHO PUSHD "%%PROJECT_ROOT_IN_WRAPPER%%"
ECHO python -m %PYTHON_MODULE% %%*
ECHO POPD
ECHO.
ECHO ENDLOCAL
) > "%WRAPPER_SCRIPT_PATH%"

ECHO.
ECHO --- RadProc Application Setup Complete ---
ECHO Conda environment '%CONDA_ENV_NAME%' created/updated.
ECHO Application directories created.
ECHO Wrapper script created at: "%WRAPPER_SCRIPT_PATH%"
ECHO.
ECHO To use RadProc:
ECHO 1. Open a new Anaconda Prompt or Command Prompt where Conda is initialized.
ECHO 2. Manually activate the Conda environment:
ECHO    conda activate %CONDA_ENV_NAME%
ECHO 3. Then you can run the 'radproc' command (if setup.py develop succeeded^):
ECHO    radproc --help
ECHO    radproc run
ECHO OR
ECHO 1. Add the '"%SCRIPT_DIR%"' directory to your system PATH.
ECHO 2. Then, from any new command prompt, run RadProc using the wrapper:
ECHO    frad-proc --help
ECHO    frad-proc run
ECHO.
ECHO IMPORTANT: Remember to configure your 'config\*.yaml' files, especially:
ECHO   - config\database_config.yaml (with details of your PostgreSQL setup^)
ECHO   - config\app_config.yaml (paths, etc.^)
ECHO And set the RADPROC_DB_PASSWORD environment variable.

ENDLOCAL
EXIT /B 0