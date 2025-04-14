#!/bin/bash

# Setup script for the radproc application on Ubuntu/Linux.
# Creates a virtual environment, installs dependencies, and sets up a wrapper script.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting radproc Setup ---"

# --- Determine Paths ---
# Get the directory where this script is located (robustly handling symlinks etc.)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Assume the script is in the 'scripts' directory, project root is one level up.
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
echo "Project root detected: $PROJECT_ROOT"

# Define key paths relative to project root
RADPROC_MODULE_DIR="${PROJECT_ROOT}/radproc"
VENV_PATH="${PROJECT_ROOT}/.venv"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts" # This script's directory
REQUIREMENTS_PATH="${PROJECT_ROOT}/requirements.txt"
LOG_DIR="${PROJECT_ROOT}/log"
DATA_DIR="${PROJECT_ROOT}/data" # For queue DB
CACHE_DIR="${PROJECT_ROOT}/cache" # Define cache base dir
ANIM_TMP_DIR="${CACHE_DIR}/animation_tmp" # Define anim tmp dir
ANIM_OUTPUT_DIR="${PROJECT_ROOT}/animations" # Define default anim output dir
API_JOB_OUTPUT_DIR="${PROJECT_ROOT}/cache/api_job_outputs" # Default path

# Python module path to run
PYTHON_MODULE="radproc.cli.main"
# Python interpreter command (use python3, standard on modern Ubuntu)
PYTHON_CMD="python3"

# --- Verification ---
if [ ! -d "$RADPROC_MODULE_DIR" ]; then
    echo "Error: Module directory not found at: $RADPROC_MODULE_DIR" >&2
    exit 1
fi
if [ ! -f "$REQUIREMENTS_PATH" ]; then
    echo "Error: requirements.txt not found at: $REQUIREMENTS_PATH" >&2
    exit 1
fi
# Verify python3 command exists
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: '$PYTHON_CMD' command not found. Please install Python 3." >&2
    exit 1
fi
# Verify venv module is available
if ! $PYTHON_CMD -m venv -h &> /dev/null; then
     echo "Error: Python 3 'venv' module not found. Please install python3-venv package (e.g., sudo apt install python3-venv)." >&2
     exit 1
fi

echo "Project structure verified."

# --- Setup ---

# Create essential directories if they don't exist (-p creates parents and ignores existing)
echo "Ensuring required directories exist..."
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$CACHE_DIR"           # Ensure base cache dir exists
mkdir -p "$ANIM_TMP_DIR"        # Create animation temp dir
mkdir -p "$ANIM_OUTPUT_DIR"
mkdir -p "$API_JOB_OUTPUT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    $PYTHON_CMD -m venv "$VENV_PATH" || {
        echo "Error: Failed to create virtual environment." >&2
        exit 1
    }
else
    echo "Virtual environment already exists at $VENV_PATH."
fi

# Activate virtual environment (for the pip install step *in this script*)
# Note: The wrapper script will need to activate it separately.
echo "Activating virtual environment for setup..."
source "$VENV_PATH/bin/activate" || {
    echo "Error: Failed to activate virtual environment." >&2
    # Try to deactivate manually in case of partial activation? Unlikely needed with set -e.
    exit 1
}

# Install/Upgrade requirements
# Using pip from the activated venv
echo "Installing/Updating requirements from $REQUIREMENTS_PATH..."
pip install --upgrade -r "$REQUIREMENTS_PATH" || {
    echo "Error: Failed to install requirements." >&2
    deactivate # Attempt to clean up
    exit 1
}

# Deactivate environment for this setup script (wrapper will activate)
echo "Deactivating virtual environment for setup script."
deactivate

# --- Create Wrapper Script ---
WRAPPER_SCRIPT_PATH="${SCRIPTS_DIR}/frad-proc" # Use 'frad-proc' as the command name
echo "Creating wrapper script: $WRAPPER_SCRIPT_PATH"

# Use cat with a HEREDOC to create the script content
cat << EOF > "$WRAPPER_SCRIPT_PATH"
#!/bin/bash
# Wrapper script for the radproc application

# Exit on error
set -e

# Find paths relative to this wrapper script
WRAPPER_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="\$(dirname "\$WRAPPER_DIR")" # Project root is parent of 'scripts' dir
VENV_PATH="\${PROJECT_ROOT}/.venv"
PYTHON_MODULE="$PYTHON_MODULE" # Embed module name

# Activate virtual environment
source "\${VENV_PATH}/bin/activate" || { echo "Error: Failed to activate venv from wrapper." >&2; exit 1; }

# Execute the python module from the project root directory
# Pass all command-line arguments ($@) to the python script
cd "\${PROJECT_ROOT}"
# Use exec to replace the shell process with python, optional but slightly cleaner
exec python -m \$PYTHON_MODULE "\$@"
# If exec fails or isn't used, the script simply ends here.

EOF
# Make the wrapper script executable
chmod +x "$WRAPPER_SCRIPT_PATH" || {
    echo "Error: Failed to make wrapper script executable." >&2
    exit 1
}

# --- Completion ---
echo ""
echo "--- Setup Complete ---"
echo "Wrapper script created at: $WRAPPER_SCRIPT_PATH"
echo ""
echo "To use:"
echo "1. Set required environment variables for FTP passwords:"
echo "   export FTP_PASSWORD_PRIMARY_SERVER=\"your_password\""
echo "   (Add similar lines for other servers to your ~/.bashrc or ~/.profile)"
echo "2. Add the scripts directory to your system PATH:"
echo "   export PATH=\"\$PATH:$SCRIPTS_DIR\""
echo "   (Add this line to your ~/.bashrc or ~/.profile and run 'source ~/.bashrc')"
echo "3. Open a new terminal or run 'source ~/.bashrc' / 'source ~/.profile'."
echo "4. Run 'frad-proc run' or 'frad-proc reprocess ...' from any directory."
echo ""
echo "For scheduled tasks (e.g., cron):"
echo "- Ensure the environment variables (passwords, PATH) are available to the cron job."
echo "- Use the full path to the wrapper: '$WRAPPER_SCRIPT_PATH run'"

exit 0