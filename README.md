# RadProc: Radar Data Processor

**RadProc** is a Python application designed for processing, visualizing, and managing radar scan data, specifically targeting Furuno radar `.scnx.gz` files.

It automates the workflow of monitoring new scans, generating standard data products and visualizations (Plan Position Indicator - PPI plots), and optionally uploading raw data to FTP servers with robust error handling.

## Key Features

*   **Automated Monitoring:** Watches a specified input directory (and subdirectories) for new `.scnx.gz` scan files.
*   **Visualization:** Generates configurable PPI plot images for specified radar variables (e.g., RATE).
*   **FTP Upload (Optional):** Uploads original scan files to one or more configured FTP servers. Features a persistent SQLite-based queue with automatic retries for robust transfers.
*   **Historical Reprocessing:** Regenerates plots for scan data within a specified date/time range from previously processed files.
*   **Configuration Driven:** Application behavior (paths, FTP settings, plot styles, variables to process) is controlled via external YAML configuration files (`config/`).
*   **Modular Design:** Built with a functional, modular approach, separating concerns into distinct components within the `radproc.core` library.
*   **CLI Interface:** Operated via a Command Line Interface (`radproc.cli.main`).

## Project Structure Highlights

*   `radproc/core/`: Contains the core backend logic (data I/O, processing, plotting, FTP, queue management, configuration loading). Designed to be reusable.
*   `radproc/cli/`: Provides the command-line interface entry points.
*   `config/`: Holds YAML configuration files (`app_config.yaml`, `ftp_config.yaml`, `plot_styles.yaml`, `radar_params.yaml`).
*   `scripts/`: Contains setup scripts (`setup_linux.sh`, `setup_windows.bat`) and runtime wrapper scripts (`frad-proc`, `frad-proc.bat`).
*   `data/`: Default location for the persistent FTP upload queue database (`upload_queue.db`).
*   `log/`: Default location for application logs (`radproc_activity.log`).

## Configuration

RadProc relies heavily on YAML configuration files located in the `config/` directory:

*   `app_config.yaml`: General settings like input/output paths, file patterns, logging configuration, and local file handling rules.
*   `ftp_config.yaml`: FTP server connection details (host, user - **NO PASSWORDS HERE**), upload modes (`disabled`, `standard`, `ftp_only`), queue settings, and retry logic.
*   `radar_params.yaml`: Radar-specific details like latitude, longitude, etc.
*   `plot_styles.yaml`: Defines plotting parameters per variable (colormaps, value ranges, map tile settings, watermark usage).

**Important: Secure Password Handling**

*   **FTP passwords are NEVER stored in configuration files or the database.**
*   For each FTP server defined in `ftp_config.yaml` with an `alias` (e.g., `primary_server`), RadProc expects a corresponding environment variable named `FTP_PASSWORD_<ALIAS_UPPERCASE>` (e.g., `FTP_PASSWORD_PRIMARY_SERVER`).
*   These variables must be set in the environment where RadProc runs. See Setup section.

## Setup

1.  **Prerequisites:**
    *   Python 3 (check `requirements.txt` or setup scripts for specific version compatibility).
    *   `python3-venv` package (recommended, especially on Linux).
    *   Git (to clone the repository).

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd radproc-project-directory
    ```

3.  **Run Setup Script:**
    *   **Linux:**
        ```bash
        cd scripts
        ./setup_linux.sh
        cd ..
        ```
    *   **Windows:**
        ```bat
        cd scripts
        setup_windows.bat
        cd ..
        ```
    *   These scripts will:
        *   Create a Python virtual environment (`.venv`).
        *   Install required packages from `requirements.txt`.
        *   Create necessary directories (`log`, `data`, `cache`).
        *   Generate wrapper scripts (`scripts/frad-proc`, `scripts/frad-proc.bat`) for easier execution.

4.  **Configure:**
    *   Edit the `.yaml` files in the `config/` directory. Pay close attention to paths (`input_dir`, `output_dir`, `images_dir`) and ensure they exist or the application has permissions to create them.
    *   Configure FTP servers and modes in `ftp_config.yaml`.
    *   Adjust plot styles in `plot_styles.yaml`.

5.  **Set Environment Variables (FTP Passwords):**

Set these environment variables directly in the service unit file (e.g., `systemd` `EnvironmentFile`) or system/user environment variables (Windows Task Scheduler user context). **Do not store passwords in scripts.**

## Usage (CLI)

It's recommended to add the `scripts` directory to your system `PATH` or activate the virtual environment first (`source .venv/bin/activate` or `.venv\Scripts\activate.bat`). The wrapper scripts handle environment activation.

*   **Show Help:**
    ```bash
    frad-proc --help
    ```

*   **Run Continuous Monitoring:**
    *   Starts monitoring the `input_dir` for new files.
    *   Processes new files according to the configuration (`app_config.yaml`, `ftp_config.yaml`).
    *   Starts the background FTP upload worker thread if FTP is enabled.
    ```bash
    frad-proc run
    ```
    *(Press Ctrl+C to stop)*

*   **Reprocess Historical Data (Generate Plots):**
    *   Finds processed scan files (`.scnx.gz`) within the `output_dir` structure matching the date range.
    *   Regenerates plots based on the *current* configuration and code.
    *   Requires start and end timestamps in `YYYYMMDD_HHMM` format.
    ```bash
    frad-proc reprocess 20231026_0000 20231026_1200
    ```

## Running as a Service

For continuous operation, RadProc can be run as a system service:

*   **Linux:** Use `systemd`. Create a service unit file defining the user, working directory, environment variables (using `EnvironmentFile=` for secrets), and the command to execute (`<path_to_project>/.venv/bin/python -m radproc.cli.main run`). Manage with `systemctl`.
*   **Windows:** Use Task Scheduler or a tool like NSSM (Non-Sucking Service Manager). Configure the task/service to run the `scripts\frad-proc.bat run` command. Ensure environment variables for FTP passwords are set for the user account the task runs as.

Refer to the detailed project description (Section 8) or system documentation for specific service setup instructions.

## Logging

*   Logging behaviour is configured in `app_config.yaml` (file path, rotation, levels).
*   Logs are typically written to `log/radproc_activity.log`.
*   Console output level is usually less verbose than the file log level.
*   When run as a `systemd` service, logs can also be accessed via `journalctl`.

## Dependencies

All Python dependencies are listed in `requirements.txt`. Key dependencies include:

*   `xarray` & `numpy`
*   `matplotlib` & `cartopy` & `shapely`
*   `pyyaml`
*   `watchdog`