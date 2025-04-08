# RadProc: Radar Data Processor

**RadProc** is a Python application designed for processing, visualizing, and managing radar scan data, specifically targeting Furuno radar `.scnx.gz` files.

It automates the workflow of monitoring new scans, generating standard data products (PPI plots, point time series), performing calculations (precipitation accumulation), and optionally uploading data to FTP servers.

## Key Features

*   **Automated Monitoring:** Watches a specified input directory for new `.scnx.gz` scan files.
*   **Visualization:** Generates configurable PPI plot images for specified radar variables (e.g., RATE).
*   **Time Series Extraction:** Extracts time series data for specific variables at user-defined points (Lat/Lon coordinates, target elevation) and saves to CSV files. Can be run historically or automatically during monitoring.
*   **Precipitation Accumulation:** Calculates accumulated precipitation over user-defined intervals (e.g., '1H', '15min') from the generated rate time series CSVs.
*   **FTP Upload (Optional):** Uploads original scan files and/or generated images to one or more configured FTP servers via a persistent queue with retries.
*   **Historical Reprocessing:** Regenerates plots for scan data within a specified date/time range.
*   **Configuration Driven:** Behavior (paths, FTP, plot styles, points) is controlled via external YAML files (`config/`).
*   **Modular Design:** Built with a functional, modular approach.
*   **CLI Interface:** Operated via a Command Line Interface.

## Project Structure Highlights

*   `radproc/core/`: Core backend logic (I/O, processing, plotting, FTP, analysis).
*   `radproc/cli/`: Command-line interface.
*   `config/`: Holds YAML configuration files:
    *   `app_config.yaml` (Paths, logging, main behaviour)
    *   `ftp_config.yaml` (FTP servers, modes)
    *   `plot_styles.yaml` (Plotting styles)
    *   `radar_params.yaml` (Radar hardware info)
    *   `points.yaml` (Points for time series extraction)
*   `scripts/`: Setup and runtime wrapper scripts.
*   `data/`: Default location for FTP queue (`upload_queue.db`).
*   `log/`: Default location for logs.
*   *(User Configured)* `timeseries_dir`: Location for generated time series CSV files (defined in `app_config.yaml`).

## Configuration

RadProc relies heavily on YAML configuration files in `config/`:

*   `app_config.yaml`: General settings (input/output/image paths), logging, **`timeseries_dir`** (path for CSV output), **`enable_timeseries_updates`** (boolean for automatic updates during `run`).
*   `ftp_config.yaml`: FTP server details (host, user - **NO PASSWORDS HERE**), upload modes (scans, images), queue settings.
*   `points.yaml`: Define points of interest using `name`, `latitude`, `longitude`, `target_elevation`, and default `variable`.
*   `radar_params.yaml`: Radar location, etc.
*   `plot_styles.yaml`: Plotting appearance.

**Important: Secure Password Handling**

*   FTP passwords are NEVER stored in config files.
*   Set environment variables `FTP_PASSWORD_<ALIAS_UPPERCASE>` (e.g., `FTP_PASSWORD_PRIMARY_SERVER`). See Setup section.

## Setup

1.  **Prerequisites:** Python 3, `python3-venv` (Linux recommended), Git.
2.  **Clone:** `git clone <your-repository-url>`
3.  **Run Setup Script:**
    *   Linux: `cd scripts && ./setup_linux.sh`
    *   Windows: `cd scripts && setup_windows.bat`
    *   (Creates `.venv`, installs requirements, creates wrappers)
4.  **Configure:** Edit `config/*.yaml` files. **Crucially, set paths in `app_config.yaml` including `timeseries_dir` if using timeseries features.** Configure points in `points.yaml`.
5.  **Set Environment Variables:** Set `FTP_PASSWORD_...` variables (e.g., in `.env` file for development, or system/service environment for production).

## Usage (CLI)

Activate virtual environment (`source .venv/bin/activate` or `.venv\Scripts\activate.bat`) or use the wrapper scripts (`frad-proc`, `frad-proc.bat`) after adding `scripts` directory to PATH.

*   **Show Help:**
    ```bash
    frad-proc --help
    ```

*   **Run Continuous Monitoring:**
    *   Processes new scans, generates plots, optionally uploads via FTP.
    *   *Also updates timeseries CSVs automatically if `enable_timeseries_updates: true`*.
    ```bash
    frad-proc run
    ```
    *(Press Ctrl+C to stop)*

*   **Reprocess Historical Plots:**
    ```bash
    frad-proc reprocess YYYYMMDD_HHMM YYYYMMDD_HHMM
    ```

*   **Generate/Update Point Time Series (Historical):**
    *   Generates/appends data to `<timeseries_dir>/<point_name>_<variable>.csv`.
    ```bash
    frad-proc timeseries <point_name> <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> [--variable VAR]
    ```

*   **Calculate Accumulated Precipitation:**
    *   Reads rate CSV, calculates accumulation, saves to new CSV.
    ```bash
    frad-proc accumulate <point_name> <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> <interval> [--variable RATE_VAR] [--output-file PATH]
    ```
    *   *(Example interval: '1H', '15min', '1D')*

## Running as a Service

*   **Linux:** Use `systemd` (see `scripts/radproc.service` template). Configure User, WorkingDirectory, EnvironmentFile (for passwords), ExecStart.
*   **Windows:** Use NSSM or Task Scheduler. Run `scripts\frad-proc.bat run`. Set environment variables for the service user.

## Logging

*   Configured in `app_config.yaml`. Logs written to file (default `log/radproc_activity.log`) and console.

## Dependencies

See `requirements.txt`. Key libraries include `xarray`, `pandas`, `matplotlib`, `cartopy`, `pyyaml`, `watchdog`, `pyproj`.