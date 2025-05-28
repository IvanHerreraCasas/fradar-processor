#!/bin/bash

# RadProc PostgreSQL Database and Schema Setup (Linux)
# This script:
# 1. Prompts for database name, username, and password.
# 2. Creates the specified PostgreSQL database and user.
# 3. Grants privileges to the new user on the new database.
# 4. Applies the schema from ../database/schema.sql to the new database.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- RadProc PostgreSQL Database & Schema Setup (Linux) ---"
echo "This script will create a new PostgreSQL database and user,"
echo "and then apply the RadProc schema."
echo "It assumes PostgreSQL is installed and you have access to the 'postgres' superuser (or equivalent)."
echo ""

# --- Prompt for Configuration ---
read -p "Enter desired PostgreSQL database name (e.g., radproc_db): " DB_NAME
read -p "Enter desired PostgreSQL username for RadProc (e.g., radproc_user): " DB_USER
read -s -p "Enter password for new PostgreSQL user '$DB_USER': " DB_PASSWORD
echo
read -s -p "Confirm password for new PostgreSQL user '$DB_USER': " DB_PASSWORD_CONFIRM
echo

if [ "$DB_PASSWORD" != "$DB_PASSWORD_CONFIRM" ]; then
    echo "Passwords do not match. Exiting."
    exit 1
fi

read -p "Enter PostgreSQL superuser (default: postgres): " PG_SUPERUSER
PG_SUPERUSER=${PG_SUPERUSER:-postgres}

# Determine Project Root (assuming this script is in the 'scripts' directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
SCHEMA_FILE="${PROJECT_ROOT}/database/schema.sql"

if [ ! -f "$SCHEMA_FILE" ]; then
    echo "Error: Database schema file not found at: $SCHEMA_FILE" >&2
    exit 1
fi

echo ""
echo "--- Summary ---"
echo "  Database Name:   $DB_NAME"
echo "  Username:        $DB_USER"
echo "  Password:        (hidden)"
echo "  PostgreSQL User: $PG_SUPERUSER (will be used to create DB/User)"
echo "  Schema File:     $SCHEMA_FILE"
echo ""
read -p "Proceed with setup? (yes/no): " CONFIRM_SETUP

if [ "$CONFIRM_SETUP" != "yes" ]; then
    echo "Setup aborted by user."
    exit 0
fi

echo ""
echo "--- Creating Database and User ---"
echo "Attempting as OS user '$PG_SUPERUSER'..."

# Peer authentication should allow the OS user PG_SUPERUSER to connect as the PG_SUPERUSER.
sudo -u "$PG_SUPERUSER" psql -v ON_ERROR_STOP=1 --command "CREATE DATABASE \"$DB_NAME\";"
sudo -u "$PG_SUPERUSER" psql -v ON_ERROR_STOP=1 --command "CREATE USER \"$DB_USER\" WITH PASSWORD '$DB_PASSWORD';"
sudo -u "$PG_SUPERUSER" psql -v ON_ERROR_STOP=1 --command "GRANT ALL PRIVILEGES ON DATABASE \"$DB_NAME\" TO \"$DB_USER\";"
sudo -u "$PG_SUPERUSER" psql -v ON_ERROR_STOP=1 --command "ALTER DATABASE \"$DB_NAME\" OWNER TO \"$DB_USER\";"

echo "Database '$DB_NAME' and user '$DB_USER' created (or an error occurred)."
echo ""

echo "--- Applying Database Schema ---"
echo "Attempting to apply schema from '$SCHEMA_FILE' to database '$DB_NAME' as user '$DB_USER'..."

# Use PGPASSWORD for non-interactive password supply for the new user
export PGPASSWORD="$DB_PASSWORD"
psql -v ON_ERROR_STOP=1 -h localhost -U "$DB_USER" -d "$DB_NAME" -a -f "$SCHEMA_FILE"

# Check if psql command was successful (basic check)
if [ $? -eq 0 ]; then
    echo "--- PostgreSQL Database and Schema Setup Successful ---"
    echo "Database '$DB_NAME' created and schema applied."
    echo "User '$DB_USER' created and granted privileges."
    echo ""
    echo "Next steps:"
    echo "1. Ensure your 'config/database_config.yaml' is updated with:"
    echo "   host: localhost (or your PostgreSQL server address)"
    echo "   port: 5432 (or your PostgreSQL port)"
    echo "   dbname: $DB_NAME"
    echo "   user: $DB_USER"
    echo "2. Set the RADPROC_DB_PASSWORD environment variable for your application:"
    echo "   export RADPROC_DB_PASSWORD=\"$DB_PASSWORD\""
    echo "   (Consider adding this to your ~/.bashrc or ~/.profile for persistence)"
    echo "3. Proceed with RadProc application setup (e.g., running the main setup script)."
else
    echo "--- PostgreSQL Database and/or Schema Setup Failed ---"
    echo "An error occurred. Please check the output above and your PostgreSQL logs."
    echo "Ensure the superuser '$PG_SUPERUSER' exists and you have its password if prompted by sudo."
    echo "Ensure user '$DB_USER' was created correctly and psql can connect."
    # No unset PGPASSWORD here to allow user to retry manually if needed immediately
    exit 1
fi

unset PGPASSWORD
echo "Setup script finished."
exit 0