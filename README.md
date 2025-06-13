# RadProc: Radar Data Processor & API

**RadProc** is a Python application designed for processing, visualizing, and managing weather radar data, specifically targeting Furuno `.scnx.gz` files. It offers a Command Line Interface (CLI) for a complete, multi-stage batch processing pipeline and a FastAPI web API for real-time data access.

The core of RadProc is a robust workflow that ingests raw radar scans, applies a chain of versioned scientific corrections, and produces high-quality, analysis-ready volumetric data products in CfRadial2 format, all orchestrated through a **PostgreSQL database** and YAML configuration files.

## Key Features

* **Three-Stage Batch Processing Pipeline:**
    1.  **Ingest (`radproc run`):** Automated monitoring of an input directory for new raw `.scnx.gz` scan files.
    2.  **Group (`radproc group-volumes`):** Analyzes the database to logically group individual scans into complete volumetric sweeps.
    3.  **Process (`radproc process-volumes`):** Creates corrected, volumetric CfRadial2 NetCDF files from the grouped scans.
* **Configurable Scientific Correction Engine:**
    *  `radproc/core/corrections` package applies a chain of scientific algorithms to the volumetric data.
    * The entire correction process is version-controlled and defined in `config/corrections.yaml`.
    * **Noise & Clutter Filtering:** Removes non-meteorological echoes using configurable methods like Correlation Coefficient (`RhoHV`), texture of reflectivity, and area-based or radial-based despeckling.
    * **Attenuation Correction:** Corrects reflectivity data for path-integrated attenuation using KDP-based methods.
    * **Quantitative Precipitation Estimation (QPE):** Derives rainfall rate from radar variables using configurable algorithms like composite Z-R and KDP-R relationships.
* **Volumetric Data Creation:** Combines multiple single-elevation scans into a single, multi-sweep `pyart.Radar` object for holistic processing.
* **Data Subsetting & Quantization:**
    * Reduces final file size by allowing the user to specify which variables and elevation angles to keep.
    * Uses quantization to store floating-point data as scaled integers, dramatically improving compression without losing significant scientific precision.
* **PostgreSQL Database Backend:**
    * Logs all processed scans and their metadata in the `radproc_scan_log` table.
    * Tracks the creation of corrected volumetric files in the `radproc_processed_volumes` table.
    * Manages definitions for points of interest, variables, and extracted time series data.
* **Advanced Visualization (`radproc plot`):**
    * Generates configurable PPI plot images (`matplotlib`, `cartopy`) from either the **raw** source data or the final **corrected** data, allowing for direct comparison.
* **Advanced Time Series & Accumulation (`radproc timeseries`, `radproc accumulate`):**
    * Extracts and analyzes time series data from either the raw or final corrected data products.
* **FastAPI Web API:** Provides HTTP endpoints for status, data access, and background job management.
* **Configuration Driven:** Core behavior is controlled via external YAML files, including the new `corrections.yaml` for defining scientific processing chains.

## Project Structure Highlights

* `radproc/core/`: Core backend logic (I/O, processing, plotting, DB interaction, analysis).
* **`radproc/core/corrections/`**: The new scientific processing package.
    * `__init__.py`: The main `apply_corrections` orchestrator.
    * `filtering.py`, `despeckle.py`, `attenuation.py`, `qpe.py`: Modules containing specific algorithms.
* `radproc/cli/`: Command-line interface (`radproc` command).
* `radproc/api/`: FastAPI application code.
* `config/`: Holds YAML configuration files.
    * `app_config.yaml`: General behavior, paths, logging.
    * `database_config.yaml`: PostgreSQL connection details.
    * **`corrections.yaml`**: **New!** Defines versioned scientific processing chains.
    * `plot_styles.yaml`: Plotting appearance.
    * `points.yaml`: Deprecated. Point definitions are now in the `radproc_points` database table.
* `database/schema.sql`: PostgreSQL database schema definition.
* `environment.yml`: Conda environment definition.
## Configuration

RadProc relies on YAML configuration files in `config/`.

* **`app_config.yaml`**: Paths, logging, `realtime_image_dir`, `timeseries_dir`, `enable_timeseries_updates`, `volume_grouping.max_inter_scan_gap_minutes`, etc.
* **`database_config.yaml`**: PostgreSQL connection details (host, port, dbname, user). **Password must be set via `RADPROC_DB_PASSWORD` environment variable.**
* **`ftp_config.yaml`**: FTP server details. **Passwords must be set via `FTP_PASSWORD_<ALIAS_UPPERCASE>` environment variables.**
* **`points.yaml`**: Now primarily for potential one-time migration into the database. Points of interest are stored in the `radproc_points` PostgreSQL table.
* * **`corrections.yaml`**: It defines versioned workflows for scientific processing. Each version specifies the methods and parameters for `noise_filter`, `despeckle`, `attenuation`, `rate_estimation`, `subsetting`, and `quantization`.

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

### Standard Operating Workflow (CLI)

The application is designed to be run as a sequence of commands, which can be automated with a scheduler like `cron`.

1.  **Run Continuously:** The `run` command should be active 24/7 to ingest new raw files.
    ```bash
    radproc run
    ```
2.  **Run Periodically (e.g., every 2 minutes):** The `group-volumes` command groups the newly ingested scans.
    ```bash
    radproc group-volumes
    ```
3.  **Run Periodically (e.g., every 5 minutes):** The `process-volumes` command finds grouped volumes and creates the final corrected data products, applying a specific version of the correction algorithms.
    ```bash
    radproc process-volumes --version v1_0_standard
    ```

### All CLI Commands

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
    radproc index-scans
    ```
* **Group Scans into Volumes:**
    ```bash
    radproc group-volumes [--lookback-hours N] [--limit N] [--dry-run]
    ```
* **Process Grouped Volumes into Corrected Files:**
    ```bash
    radproc process-volumes --version <version_name> [--limit N]
    ```
* **Generate Historical Plots:**
    ```bash
    radproc plot <variable> <elevation> <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> [--source <raw|corrected>] [--version <version_name>]
    ```
* **Generate Historical Time Series:**
    ```bash
    radproc timeseries <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> [--points P1 P2] [--source <raw|corrected>] [--version <version_name>]
    ```
* **Calculate Accumulated Precipitation:**
    ```bash
    radproc accumulate <point_name> <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> <interval> [--source <raw|corrected>] [--version <version_name>]
    ```
* **Create Animation File:**
    ```bash
    radproc animate <variable> <elevation> <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> <output_file>
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

See `environment.yml`. Key libraries include: `fastapi`, `uvicorn`, `huey`, `psycopg2` (for PostgreSQL), `watchfiles`, `xarray`, `xradar`, `pandas`, `matplotlib`, `scipy`, `pyart`, `cartopy`, `pyyaml`, `tqdm`, `imageio`, `imageio-ffmpeg`.
