# RadProc: Radar Data Processor & API (PostgreSQL Edition)

**RadProc** is a Python application designed for processing, visualizing, and managing radar scan data, specifically targeting Furuno radar `.scnx.gz` files. It offers both a Command Line Interface (CLI) for batch processing and monitoring, and a FastAPI web API for real-time data access and background job management.

This version features a robust **PostgreSQL backend** for storing scan metadata, points of interest, and extracted time series data, moving away from previous file-based or YAML-based configurations for these core entities.

## Key Features

* **Automated Monitoring (CLI):** Watches a specified input directory for new `.scnx.gz` scan files via `radproc run`.
* **Core Processing:** Reads (`xarray`, `xradar`), preprocesses, and georeferences (`pyproj`) radar data.
* **PostgreSQL Database Backend:**
    * Logs all processed scans, including their metadata (filepath, precise timestamp, elevation, sequence number), into the `radproc_scan_log` table.
    * Stores definitions for points of interest (`radproc_points`) and radar variables (`radproc_variables`).
    * Manages extracted time series data in the `timeseries_data` table.
* **Volume Grouping:** The `radproc group-volumes` command analyzes the `radproc_scan_log` to assign a common `volume_identifier` to scans belonging to the same volumetric sweep.
* **Visualization:** Generates configurable PPI plot images (`matplotlib`, `cartopy`) for specified radar variables. Saves plots historically and updates a "realtime" image for API access.
* **Time Series Extraction & Management:**
    * Extracts time series data for specified variables at user-defined points (now managed in the database).
    * Stores time series directly in the PostgreSQL database.
    * `radproc timeseries` CLI command for historical generation, supporting specific or all database-defined points.
    * Optimized for memory efficiency during batch processing.
* **Precipitation Accumulation:** Calculates accumulated precipitation over user-defined intervals from rate time series (sourced from the database) via `radproc accumulate` or an API job, outputting to CSV.
* **Animation:** Creates animations (GIF/MP4) from generated plot images (`radproc animate` or via API job). Uses `imageio` and sources scan information from the database.
* **FTP Upload (Optional):** Uploads original scan files and/or generated images to FTP servers via a persistent SQLite-based queue with retries. Managed by `radproc run`.
* **FastAPI Web API:**
    * Provides HTTP endpoints to access status, configured points (from DB), latest ("realtime") plots, and specific historical plots.
    * Offers a Server-Sent Events (SSE) stream (`/plots/stream/updates`) for real-time notification of plot updates.
    * Allows submitting background jobs (timeseries generation, accumulation, animation) via a task queue (`Huey` with SQLite backend).
    * Provides endpoints to check job status and retrieve results/data (timeseries data from DB, other files from disk).
* **Background Task Queue:** Uses `Huey` with an SQLite backend (`huey_queue.db`) for handling asynchronous API tasks. Requires a separate `huey_consumer` process.
* **Configuration Driven:** Core behavior (paths, FTP, plot styles, API settings, queue paths) is controlled via external YAML files (`config/`). Point definitions are now primarily managed in the database.
* **Setup Scripts:** Provides scripts for setting up the PostgreSQL database/user/schema on Linux and separate scripts for setting up the Conda application environment on Linux and Windows.

## Project Structure Highlights

* `radproc/core/`: Core backend logic (I/O, processing, plotting, DB interaction, analysis, animation).
* `radproc/cli/`: Command-line interface (`radproc` command).
* **`radproc/api/`**: FastAPI application code.
* **`radproc/huey_config.py` & `radproc/tasks.py`**: Huey task queue setup and task definitions.
* `database/schema.sql`: PostgreSQL database schema definition.
* `config/`: Holds YAML configuration files.
    * `app_config.yaml`: Paths, logging, general behavior (includes `volume_grouping.max_inter_scan_gap_minutes`).
    * `database_config.yaml`: PostgreSQL connection parameters (excluding password).
    * `ftp_config.yaml`: FTP servers, modes, SQLite queue settings.
    * `radar_params.yaml`: Radar hardware info.
    * `plot_styles.yaml`: Plotting appearance.
    * `points.yaml`: **Deprecated for runtime use.** May be used for initial migration to the database. Points are now managed in the `radproc_points` table.
* `scripts/`:
    * `setup_postgres_linux.sh`: Sets up PostgreSQL database, user, and applies `schema.sql` on Linux.
    * `apply_schema_windows.bat`: Applies `schema.sql` to an existing PostgreSQL database on Windows.
    * `setup_linux.sh` / `setup_windows.bat`: Set up the Conda Python environment for RadProc.
* `data/`: Default location for persistent SQLite queues (FTP `upload_queue.db`, Huey `huey_queue.db`).
* `environment.yml`: Conda environment definition file.

## Configuration

RadProc relies on YAML configuration files in `config/`.

* **`app_config.yaml`**: Paths, logging, `realtime_image_dir`, `timeseries_dir`, `enable_timeseries_updates`, `volume_grouping.max_inter_scan_gap_minutes`, etc.
* **`database_config.yaml`**: PostgreSQL connection details (host, port, dbname, user). **Password must be set via `RADPROC_DB_PASSWORD` environment variable.**
* **`ftp_config.yaml`**: FTP server details. **Passwords must be set via `FTP_PASSWORD_<ALIAS_UPPERCASE>` environment variables.**
* **`points.yaml`**: Now primarily for potential one-time migration into the database. Points of interest are stored in the `radproc_points` PostgreSQL table.

**Important: Secure Password Handling**

* Database and FTP passwords are NEVER stored in config files.
* Set `RADPROC_DB_PASSWORD` for PostgreSQL access.
* Set `FTP_PASSWORD_<ALIAS_UPPERCASE>` for FTP servers.

**Important: Huey SQLite Configuration**

* The Huey task queue uses an SQLite database (`data/huey_queue.db` by default).
* The API (`radproc/api/main.py`) attempts to apply `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;` on startup. Verify this in API logs.

## Setup

The setup process is now two-staged: Database Setup, then Application Environment Setup.

**Stage 1: Database Setup**

* **Linux:**
    1.  Ensure PostgreSQL server is installed and running.
    2.  Run `cd scripts && ./setup_postgres_linux.sh`. This script will prompt you for DB name, user, password, and apply the schema from `database/schema.sql`.
* **Windows:**
    1.  Install PostgreSQL (e.g., via EDB installer), including Command Line Tools. Add PostgreSQL `bin` to your PATH.
    2.  Manually create a database (e.g., `radproc_db`) and a user (e.g., `radproc_user`) with a password, granting appropriate privileges using `pgAdmin` or `psql`.
    3.  Run `cd scripts && apply_schema_windows.bat`. This script will prompt for connection details and apply the `database/schema.sql`.
* **Post-DB Setup:**
    1.  Update `config/database_config.yaml` with your DB connection details (host, port, dbname, user).
    2.  Set the `RADPROC_DB_PASSWORD` environment variable to the password of your RadProc database user.

**Stage 2: RadProc Application Environment Setup (Conda)**

1.  **Prerequisites:** Anaconda or Miniconda installed, Git. For animations, `ffmpeg` might need to be installed system-wide if `imageio-ffmpeg` doesn't bundle it.
2.  **Clone:** `git clone <your-repository-url>` (if not already done)
3.  **Run Application Setup Script:**
    * Linux: `cd scripts && ./setup_linux.sh`
    * Windows: `cd scripts && setup_windows.bat`
    * (This creates/updates the Conda environment from `environment.yml`, creates app directories, and sets up CLI wrappers).
4.  **Configure:** Review and edit other `config/*.yaml` files as needed (e.g., `app_config.yaml`, `ftp_config.yaml`).
5.  **Set FTP Passwords:** Set `FTP_PASSWORD_...` environment variables if using FTP.

## Usage

### Command Line Interface (CLI)

Activate the Conda environment (`conda activate frad-proc` or the name in your `environment.yml`) or use the wrapper scripts (`frad-proc`, `frad-proc.bat`) potentially added to your PATH.

* **Show Help:**
    ```bash
    radproc --help
    ```
* **Run Continuous Monitoring:**
    ```bash
    radproc run
    ```
* **Index Existing Scans into DB:**
    ```bash
    radproc index-scans [--output-dir /path/to/processed_scans] [--dry-run]
    ```
* **Group Scans into Volumes:**
    ```bash
    radproc group-volumes [--lookback-hours N] [--limit N] [--dry-run]
    ```
* **Generate/Update Point Time Series in DB (Historical):**
    ```bash
    radproc timeseries <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> [--points P1 P2] [--variable VAR]
    ```
    *(If `--points` is omitted, processes all points from the database)*
* **Calculate Accumulated Precipitation (from DB data, outputs CSV):**
    ```bash
    radproc accumulate <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> --points <point_name> --interval <interval> [--variable RATE_VAR] [--output-file PATH]
    ```
* **Create Animation File (uses DB for scan info):**
    ```bash
    radproc animate <variable> <elevation> <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> <output_file.[gif|mp4|...]> [--extent LONMIN LONMAX LATMIN LATMAX] [--no-watermark] [--fps FPS]
    ```

### Web API

1.  **Run the API Server:**
    ```bash
    # Activate Conda environment
    uvicorn radproc.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    * Access docs: `http://localhost:8000/docs` (Swagger), `http://localhost:8000/redoc` (ReDoc).

2.  **Run the Huey Worker (for API Background Jobs):**
    ```bash
    # Activate Conda environment
    huey_consumer radproc.huey_config.huey -k thread -w 1 -v
    ```

3.  **API Endpoint Highlights:**
    * `GET /status`: Basic API status.
    * `GET /points`: List configured points of interest (from DB).
    * `GET /points/{point_name}`: Get details for a specific point (from DB).
    * `GET /plots/realtime/{variable}/{elevation}`: Get the latest plot image.
    * `GET /plots/historical/{variable}/{elevation}/{datetime_str}`: Get a specific past plot image.
    * `GET /plots/frames`: Get list of available frame identifiers for a sequence.
    * `GET /plots/stream/updates`: Server-Sent Events stream for real-time plot update notifications.
    * `POST /jobs/{timeseries|accumulation|animation}`: Submit background jobs. Timeseries jobs now operate fully against the PostgreSQL database.
    * `GET /jobs/{task_id}/status`: Check job status.
    * `GET /jobs/{timeseries|accumulation|animation}/{task_id}/data`: Retrieve job results/data files.

## Logging

* Configured in `app_config.yaml` (separate `log_file`, `api_log_file`, `huey_log_file`).
* Huey consumer output can be directed to a file using its own flags or system service management.

## Dependencies

See `environment.yml`. Key libraries include: `fastapi`, `uvicorn`, `huey`, `psycopg2` (for PostgreSQL), `watchfiles`, `xarray`, `xradar`, `pandas`, `matplotlib`, `cartopy`, `pyyaml`, `tqdm`, `imageio`, `imageio-ffmpeg`.
