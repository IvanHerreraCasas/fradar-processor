@ECHO OFF
SETLOCAL

ECHO --- RadProc PostgreSQL Schema Application (Windows) ---
ECHO This script will attempt to apply the database schema (tables, indexes)
ECHO to an EXISTING RadProc PostgreSQL database and user.
ECHO.
ECHO Prerequisites:
ECHO   1. PostgreSQL server is installed and running.
ECHO   2. A database for RadProc has been created (e.g., 'radproc_db').
ECHO   3. A user for RadProc has been created and granted privileges on that database.
ECHO   4. 'psql.exe' (PostgreSQL command-line tool) is in your system's PATH.
ECHO   5. You know the connection details (host, port, dbname, username).
ECHO.

CHOICE /C YN /M "Do you want to proceed with applying the schema?"
IF ERRORLEVEL 2 (
    ECHO Schema application aborted by user.
    EXIT /B 0
)
IF ERRORLEVEL 1 (
    ECHO.
)

REM --- Determine Project Root ---
REM Assumes this script is in the 'scripts' directory
SET "SCRIPT_DIR=%~dp0"
FOR %%A IN ("%SCRIPT_DIR%..") DO SET "PROJECT_ROOT=%%~fA"
SET "SCHEMA_FILE=%PROJECT_ROOT%\database\schema.sql"

IF NOT EXIST "%SCHEMA_FILE%" (
    ECHO Error: Database schema file not found at: "%SCHEMA_FILE%"
    EXIT /B 1
)

REM --- Prompt for Connection Details ---
ECHO Please enter the connection details for your RadProc PostgreSQL database:
:PROMPT_HOST
SET "DB_HOST="
SET /P DB_HOST="Enter PostgreSQL host (e.g., localhost, or press Enter for 'localhost'): "
IF "%DB_HOST%"=="" SET "DB_HOST=localhost"

:PROMPT_PORT
SET "DB_PORT="
SET /P DB_PORT="Enter PostgreSQL port (e.g., 5432, or press Enter for '5432'): "
IF "%DB_PORT%"=="" SET "DB_PORT=5432"

:PROMPT_DBNAME
SET "DB_NAME="
SET /P DB_NAME="Enter RadProc Database Name (e.g., radproc_db): "
IF "%DB_NAME%"=="" (
    ECHO Database name cannot be empty.
    GOTO PROMPT_DBNAME
)

:PROMPT_USER
SET "DB_USER="
SET /P DB_USER="Enter RadProc Username for this database: "
IF "%DB_USER%"=="" (
    ECHO Username cannot be empty.
    GOTO PROMPT_USER
)
ECHO You will be prompted for the password for user '%DB_USER%' by psql.
ECHO.

ECHO --- Summary ---
ECHO   Host:          %DB_HOST%
ECHO   Port:          %DB_PORT%
ECHO   Database Name: %DB_NAME%
ECHO   Username:      %DB_USER%
ECHO   Schema File:   "%SCHEMA_FILE%"
ECHO.

CHOICE /C YN /M "Are these details correct and do you want to apply the schema?"
IF ERRORLEVEL 2 (
    ECHO Schema application aborted.
    EXIT /B 0
)
IF ERRORLEVEL 1 (
    ECHO.
)

ECHO Attempting to apply schema using psql...

REM The -W flag will force psql to prompt for a password.
REM The -a flag echoes all input from script.
REM The -v ON_ERROR_STOP=1 will cause psql to exit on an error.
psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d %DB_NAME% -a -v ON_ERROR_STOP=1 -f "%SCHEMA_FILE%"

IF %ERRORLEVEL% EQU 0 (
    ECHO.
    ECHO --- Database Schema Applied Successfully! ---
    ECHO The tables and indexes should now be created in database '%DB_NAME%'.
    ECHO.
    ECHO Next steps:
    ECHO   1. Ensure 'config\database_config.yaml' reflects these connection details.
    ECHO   2. Set the RADPROC_DB_PASSWORD environment variable for the user '%DB_USER%'.
    ECHO   3. If you haven't already, run the RadProc application setup script (e.g., setup_windows.bat).
) ELSE (
    ECHO.
    ECHO --- Failed to Apply Database Schema ---
    ECHO An error occurred. psql exited with error code %ERRORLEVEL%.
    ECHO Please check the output above from psql for details.
    ECHO Common issues:
    ECHO   - 'psql.exe' not found in PATH.
    ECHO   - Incorrect connection details (host, port, dbname, user).
    ECHO   - Incorrect password entered when prompted.
    ECHO   - Database or user does not exist or user lacks privileges.
    ECHO   - Errors within the '%SCHEMA_FILE%' file itself.
)

PAUSE
ENDLOCAL
EXIT /B %ERRORLEVEL%