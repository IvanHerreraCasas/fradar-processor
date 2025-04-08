# RadProc: Radar Data Processor

**RadProc** is a Python application designed for processing, visualizing, and managing radar scan data, specifically targeting Furuno radar `.scnx.gz` files.

It automates the workflow of monitoring new scans, generating standard data products (PPI plots, point time series), performing calculations (precipitation accumulation), creating animations, and optionally uploading data to FTP servers.

## Key Features

*   **Automated Monitoring:** Watches a specified input directory for new `.scnx.gz` scan files.
*   **Visualization:** Generates configurable PPI plot images for specified radar variables (e.g., RATE).
*   **Time Series Extraction:** Extracts time series data for specific variables at user-defined points (Lat/Lon coordinates, target elevation) and saves to CSV files. Can be run historically or automatically.
*   **Precipitation Accumulation:** Calculates accumulated precipitation over user-defined intervals (e.g., '1H', '15min') from the generated rate time series CSVs.
*   **Animation:** Creates animations (GIF/MP4) from generated plot images for a specified variable, elevation, and time range. Supports custom extents and watermark toggling.
*   **FTP Upload (Optional):** Uploads original scan files and/or generated images to one or more configured FTP servers via a persistent queue with retries.
*   **Historical Reprocessing:** Regenerates plots for scan data within a specified date/time range.
*   **Configuration Driven:** Behavior (paths, FTP, plot styles, points, animation defaults) is controlled via external YAML files (`config/`).
*   **Modular Design:** Built with a functional, modular approach.
*   **CLI Interface:** Operated via a Command Line Interface.

## Project Structure Highlights

*   `radproc/core/`: Core backend logic (I/O, processing, plotting, FTP, analysis, animation).
*   `radproc/cli/`: Command-line interface.
*   `config/`: Holds YAML configuration files:
    *   `app_config.yaml` (Paths, logging, main behaviour, animation defaults)
    *   `ftp_config.yaml` (FTP servers, modes)
    *   `plot_styles.yaml` (Plotting styles)
    *   `radar_params.yaml` (Radar hardware info)
    *   `points.yaml` (Points for time series extraction)
*   `scripts/`: Setup and runtime wrapper scripts.
*   `data/`: Default location for FTP queue (`upload_queue.db`).
*   `log/`: Default location for logs.
*   `cache/`: Default location for map tiles and temporary animation frames.
*   *(User Configured)* `timeseries_dir`: Location for generated time series CSV files (defined in `app_config.yaml`).

## Configuration

RadProc relies heavily on YAML configuration files in `config/`:

*   `app_config.yaml`: General settings (input/output/image paths), logging, `timeseries_dir`, `enable_timeseries_updates`, **`animation_fps`**, **`animation_default_format`**, **`animation_tmp_dir`**.
*   `ftp_config.yaml`: FTP server details (host, user - **NO PASSWORDS HERE**), upload modes, queue settings.
*   `points.yaml`: Define points of interest (`name`, `latitude`, `longitude`, `target_elevation`, `variable`).
*   `radar_params.yaml`: Radar location, etc.
*   `plot_styles.yaml`: Plotting appearance.

**Important: Secure Password Handling**

*   FTP passwords are NEVER stored in config files.
*   Set environment variables `FTP_PASSWORD_<ALIAS_UPPERCASE>`. See Setup section.

## Setup

1.  **Prerequisites:** Python 3, `python3-venv` (Linux recommended), Git. For animations, `ffmpeg` might need to be installed system-wide depending on your OS if `imageio-ffmpeg` doesn't bundle it successfully.
2.  **Clone:** `git clone <your-repository-url>`
3.  **Run Setup Script:**
    *   Linux: `cd scripts && ./setup_linux.sh`
    *   Windows: `cd scripts && setup_windows.bat`
    *   (Creates `.venv`, installs requirements including `imageio` and `imageio-ffmpeg`, creates wrappers and default directories like `cache/animation_tmp`)
4.  **Configure:** Edit `config/*.yaml` files. Set required paths in `app_config.yaml` (`input_dir`, `output_dir`, `images_dir`, `timeseries_dir`). Configure points in `points.yaml`.
5.  **Set Environment Variables:** Set `FTP_PASSWORD_...` variables.

## Usage (CLI)

Activate virtual environment or use the wrapper scripts (`frad-proc`, `frad-proc.bat`) after adding `scripts` directory to PATH.

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

*   **Create Animation:**
    *   Finds/regenerates frames and creates an animation file.
    ```bash
    frad-proc animate <variable> <elevation> <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> <output_file.[gif|mp4|...]> [--extent LONMIN LONMAX LATMIN LATMAX] [--no-watermark] [--fps FPS]
    ```
    *   *(Example: `frad-proc animate RATE 0.5 20231027_1000 20231027_1200 output/rate_anim.mp4 --fps 10`)*

## Running as a Service

*   **Linux:** Use `systemd` (see `scripts/radproc.service` template).
*   **Windows:** Use NSSM or Task Scheduler. Run `scripts\frad-proc.bat run`.

## Logging

*   Configured in `app_config.yaml`. Logs written to file (default `log/radproc_activity.log`) and console.

## Dependencies

See `requirements.txt`. Key libraries include `xarray`, `xradar`, `pandas`, `matplotlib`, `cartopy`, `pyyaml`, `watchdog`, `pyproj`, `imageio`, `imageio-ffmpeg`.