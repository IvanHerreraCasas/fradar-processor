#!/bin/bash

# RadProc Application Setup Script (Linux - Conda)
# This script:
# 1. Checks for Conda installation.
# 2. Creates/updates the Conda environment using environment.yml.
# 3. Creates necessary application directories (log, data, cache, etc.).
# 4. Creates a wrapper script to run the RadProc CLI.
#
# IMPORTANT: This script assumes PostgreSQL database and user have ALREADY been set up.
# Use 'setup_postgres_linux.sh' or manual setup for the database first.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting RadProc Application Setup (Linux - Conda) ---"
echo "Ensure your PostgreSQL database is set up before running this."
read -p "Proceed with RadProc application setup? (yes/no): " CONFIRM_APP_SETUP
if [ "$CONFIRM_APP_SETUP" != "yes" ]; then
    echo "Application setup aborted by user."
    exit 0
fi

# --- Determine Paths ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )" # Project root is one level up from 'scripts'

# Define key paths relative to project root
RADPROC_MODULE_DIR="${PROJECT_ROOT}/radproc"
REQUIREMENTS_PATH="${PROJECT_ROOT}/requirements.txt" # Still used by setup.py, may be installed by conda via environment.yml
ENVIRONMENT_YAML_PATH="${PROJECT_ROOT}/environment.yml"
LOG_DIR="${PROJECT_ROOT}/log"
DATA_DIR="${PROJECT_ROOT}/data" # For queue DBs (FTP upload, Huey)
CACHE_DIR="${PROJECT_ROOT}/cache"
ANIM_TMP_DIR="${CACHE_DIR}/animation_tmp"
ANIM_OUTPUT_DIR="${PROJECT_ROOT}/animations"
API_JOB_OUTPUT_DIR="${PROJECT_ROOT}/cache/api_job_outputs"

PYTHON_MODULE="radproc.cli.main" # Python module to run for the CLI

# --- Verification ---
echo "Verifying prerequisites..."
if [ ! -d "$RADPROC_MODULE_DIR" ]; then
    echo "Error: Module directory not found at: $RADPROC_MODULE_DIR" >&2
    exit 1
fi
if [ ! -f "$ENVIRONMENT_YAML_PATH" ]; then
    echo "Error: environment.yml not found at: $ENVIRONMENT_YAML_PATH" >&2
    exit 1
fi
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please install Anaconda or Miniconda and ensure it's in your PATH." >&2
    exit 1
fi
echo "Prerequisites verified."

# --- Setup ---

# Create essential application directories if they don't exist
echo "Ensuring required application directories exist..."
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$ANIM_TMP_DIR"
mkdir -p "$ANIM_OUTPUT_DIR"
mkdir -p "$API_JOB_OUTPUT_DIR"
echo "Application directories ensured."

# Create/Update Conda environment
echo "Creating/Updating Conda environment from $ENVIRONMENT_YAML_PATH..."
CONDA_ENV_NAME=$(grep 'name:' "$ENVIRONMENT_YAML_PATH" | awk '{print $2}') # Get name from YAML
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Environment '$CONDA_ENV_NAME' already exists. Updating..."
    conda env update -f "$ENVIRONMENT_YAML_PATH" --prune || {
        echo "Error: Failed to update Conda environment '$CONDA_ENV_NAME'." >&2
        exit 1
    }
else
    echo "Creating new Conda environment '$CONDA_ENV_NAME'..."
    conda env create -f "$ENVIRONMENT_YAML_PATH" || {
        echo "Error: Failed to create Conda environment '$CONDA_ENV_NAME'." >&2
        exit 1
    }
fi
echo "Conda environment '$CONDA_ENV_NAME' is ready."

# If setup.py still needs to be run (e.g., for 'editable' install or console_scripts)
# This is often handled by including pip in environment.yml and then:
# dependencies:
#   - pip:
#     - -e .
# If not, this step might be skippable or modified.
# The current setup.py creates a 'radproc' console script.
# Conda can also create entry points if defined in meta.yaml (for conda build),
# but for direct env setup, setup.py is common.
echo "Running setup.py develop to make 'radproc' command available in Conda env..."
conda run -n "$CONDA_ENV_NAME" python "${PROJECT_ROOT}/setup.py" develop || {
    echo "Error: Failed to run 'python setup.py develop' in $CONDA_ENV_NAME." >&2
    echo "The 'radproc' command might not be directly available after activating the environment."
    # exit 1 # Decide if this is fatal
}


# --- Create Wrapper Script (frad-proc) ---
# This provides an alternative way to call the CLI without manually activating the env.
WRAPPER_SCRIPT_PATH="${SCRIPTS_DIR}/frad-proc"
echo "Creating wrapper script: $WRAPPER_SCRIPT_PATH"

cat << EOF > "$WRAPPER_SCRIPT_PATH"
#!/bin/bash
# Wrapper script for the RadProc application (Conda environment)

# Exit on error
set -e

# Determine the directory of this wrapper script
WRAPPER_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="\$(dirname "\$WRAPPER_DIR")" # Project root is parent of 'scripts' dir
PYTHON_MODULE="$PYTHON_MODULE"
CONDA_ENV_NAME="$CONDA_ENV_NAME" # Embed Conda environment name

# Attempt to find CONDA_EXE, otherwise assume conda is in PATH
_CONDA_EXE="\${CONDA_EXE:-\$(command -v conda)}"
if [ -z "\$_CONDA_EXE" ]; then
    echo "Error: Conda executable not found. Cannot run RadProc via wrapper." >&2
    exit 1
fi

# Source Conda's shell functions if needed (best effort)
_CONDA_PROFILE_PATH_1="\$HOME/miniconda3/etc/profile.d/conda.sh"
_CONDA_PROFILE_PATH_2="\$HOME/anaconda3/etc/profile.d/conda.sh"
_CONDA_PROFILE_PATH_3="/opt/conda/etc/profile.d/conda.sh" # Common in Docker

if [ -f "\$_CONDA_PROFILE_PATH_1" ]; then
    source "\$_CONDA_PROFILE_PATH_1"
elif [ -f "\$_CONDA_PROFILE_PATH_2" ]; then
    source "\$_CONDA_PROFILE_PATH_2"
elif [ -f "\$_CONDA_PROFILE_PATH_3" ]; then
    source "\$_CONDA_PROFILE_PATH_3"
else
    echo "Info: Could not automatically source conda.sh. 'conda activate' will be attempted directly."
fi

# Activate Conda environment
# Using a subshell to keep activation local to this script execution if possible,
# though 'conda activate' often modifies the current shell.
# More robust might be 'conda run -n <env> python ...' if activation side-effects are an issue.
(
    conda activate "\$CONDA_ENV_NAME" || {
        echo "Error: Failed to activate Conda environment '\$CONDA_ENV_NAME' from wrapper." >&2;
        echo "Try activating manually: conda activate \$CONDA_ENV_NAME" >&2;
        exit 1;
    }
    # Execute the python module from the project root directory
    # Pass all command-line arguments (\$@) to the python script
    cd "\${PROJECT_ROOT}"
    exec python -m \$PYTHON_MODULE "\$@"
)
# If exec fails or isn't used, the script simply ends here.
EOF

chmod +x "$WRAPPER_SCRIPT_PATH" || {
    echo "Error: Failed to make wrapper script executable." >&2
    exit 1
}

# --- Completion ---
echo ""
echo "--- RadProc Application Setup Complete ---"
echo "Conda environment '$CONDA_ENV_NAME' created/updated."
echo "Application directories created."
echo "Wrapper script created at: $WRAPPER_SCRIPT_PATH"
echo ""
echo "To use RadProc:"
echo "1. Manually activate the Conda environment:"
echo "   conda activate $CONDA_ENV_NAME"
echo "2. Then you can run the 'radproc' command (if setup.py develop succeeded):"
echo "   radproc --help"
echo "   radproc run"
echo "OR"
echo "1. Add the '$SCRIPTS_DIR' to your system PATH (e.g., in ~/.bashrc):"
echo "   export PATH=\"\$PATH:$SCRIPTS_DIR\""
echo "   Then, after sourcing your .bashrc or opening a new terminal:"
echo "2. Run RadProc using the wrapper from any directory:"
echo "   frad-proc --help"
echo "   frad-proc run"
echo ""
echo "IMPORTANT: Remember to configure your 'config/*.yaml' files, especially:"
echo "  - config/database_config.yaml (with details of your PostgreSQL setup)"
echo "  - config/app_config.yaml (paths, etc.)"
echo "And set the RADPROC_DB_PASSWORD environment variable."

exit 0