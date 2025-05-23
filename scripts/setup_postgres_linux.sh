#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- RadProc PostgreSQL Database Setup (Linux) ---"

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

echo ""
echo "Summary:"
echo "  Database Name: $DB_NAME"
echo "  Username:      $DB_USER"
echo "  Superuser:     $PG_SUPERUSER"
echo ""
read -p "Proceed with setup? (yes/no): " CONFIRM_SETUP

if [ "$CONFIRM_SETUP" != "yes" ]; then
    echo "Setup aborted by user."
    exit 0
fi

echo "Attempting to create database and user as OS user '$PG_SUPERUSER'..."

# --- Execute SQL Commands using sudo -u <PG_SUPERUSER> ---
# The OS user PG_SUPERUSER (defaulting to 'postgres') executes psql.
# Peer authentication should allow this OS user to connect as the PG_SUPERUSER.
# No password for PG_SUPERUSER should be prompted by psql itself here
# if peer auth is correctly set for local connections for the 'postgres' PG user.

sudo -u "$PG_SUPERUSER" psql -d postgres -v ON_ERROR_STOP=1 <<EOF
CREATE DATABASE "$DB_NAME";
CREATE USER "$DB_USER" WITH PASSWORD '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE "$DB_NAME" TO "$DB_USER";
ALTER DATABASE "$DB_NAME" OWNER TO "$DB_USER";
-- Optional: Connect to the new database and grant schema permissions
-- \\c "$DB_NAME"
-- GRANT ALL ON SCHEMA public TO "$DB_USER";
EOF

# Check if psql command was successful (basic check)
if [ $? -eq 0 ]; then
    echo "--- PostgreSQL Setup Successful ---"
    echo "Database '$DB_NAME' and user '$DB_USER' created."
    echo ""
    echo "Next steps:"
    echo "1. Update your application configuration (e.g., app_config.yaml or database_config.yaml)"
    echo "   with these details:"
    echo "   DB_HOST: localhost (or your PostgreSQL server address)"
    echo "   DB_PORT: 5432 (or your PostgreSQL port)"
    echo "   DB_NAME: $DB_NAME"
    echo "   DB_USER: $DB_USER"
    echo "2. Set the DB_PASSWORD environment variable for your application:"
    echo "   export RADPROC_DB_PASSWORD=\"$DB_PASSWORD\""
    echo "   (Add this to your ~/.bashrc or ~/.profile for persistence)"
else
    echo "--- PostgreSQL Setup Failed ---"
    echo "An error occurred. Please check the output above and your PostgreSQL logs."
    echo "Ensure the superuser '$PG_SUPERUSER' exists and you have its password if prompted."
    exit 1
fi

exit 0