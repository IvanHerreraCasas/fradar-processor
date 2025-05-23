@echo off
echo --- RadProc PostgreSQL Database Setup (Windows) ---

REM --- Prompt for Configuration ---
set /p DB_NAME="Enter desired PostgreSQL database name (e.g., radproc_db): "
set /p DB_USER="Enter desired PostgreSQL username for RadProc (e.g., radproc_user): "
echo Enter password for new PostgreSQL user '%DB_USER%':
set /p DB_PASSWORD=
echo Confirm password for new PostgreSQL user '%DB_USER%':
set /p DB_PASSWORD_CONFIRM=

if not "%DB_PASSWORD%"=="%DB_PASSWORD_CONFIRM%" (
    echo Passwords do not match. Exiting.
    goto :eof
)

set /p PG_SUPERUSER="Enter PostgreSQL superuser (default: postgres): "
if "%PG_SUPERUSER%"=="" set PG_SUPERUSER=postgres

echo.
echo Summary:
echo   Database Name: %DB_NAME%
echo   Username:      %DB_USER%
echo   Superuser:     %PG_SUPERUSER%
echo.
set /p CONFIRM_SETUP="Proceed with setup? (yes/no): "

if /i not "%CONFIRM_SETUP%"=="yes" (
    echo Setup aborted by user.
    goto :eof
)

echo Attempting to create database and user...

REM --- Execute SQL Commands ---
REM For Windows, psql might require setting PGPASSWORD environment variable
REM or using a .pgpass file for the superuser if password is required.
REM This script assumes psql is in PATH.

REM Double quotes around identifiers, single quotes around string literals (like password)
set PGSQL_COMMANDS="CREATE DATABASE \"%DB_NAME%\"; CREATE USER \"%DB_USER%\" WITH PASSWORD '%DB_PASSWORD%'; GRANT ALL PRIVILEGES ON DATABASE \"%DB_NAME%\" TO \"%DB_USER%\"; ALTER DATABASE \"%DB_NAME%\" OWNER TO \"%DB_USER%\";"

REM The superuser might be prompted for their password by psql.
psql -U "%PG_SUPERUSER%" -d postgres -c %PGSQL_COMMANDS%

if errorlevel 1 (
    echo --- PostgreSQL Setup Failed ---
    echo An error occurred. Please check the output above and your PostgreSQL logs.
    echo Ensure psql is in your PATH.
    echo Ensure the superuser '%PG_SUPERUSER%' exists and you can authenticate as them.
    goto :eof
)

echo --- PostgreSQL Setup Successful ---
echo Database '%DB_NAME%' and user '%DB_USER%' created.
echo.
echo Next steps:
echo 1. Update your application configuration (e.g., app_config.yaml or database_config.yaml)
echo    with these details:
echo    DB_HOST: localhost (or your PostgreSQL server address)
echo    DB_PORT: 5432 (or your PostgreSQL port)
echo    DB_NAME: %DB_NAME%
echo    DB_USER: %DB_USER%
echo 2. Set the DB_PASSWORD environment variable for your application:
echo    set RADPROC_DB_PASSWORD=%DB_PASSWORD%
echo    (Consider setting this persistently in System Environment Variables)

:eof