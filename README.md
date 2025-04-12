# RadProc: Radar Data Processor & API

**RadProc** is a Python application designed for processing, visualizing, and managing radar scan data, specifically targeting Furuno radar `.scnx.gz` files. It offers both a Command Line Interface (CLI) for batch processing and monitoring, and a FastAPI web API for real-time data access and background job management.

It automates the workflow of monitoring new scans, generating standard data products (PPI plots, point time series), performing calculations (precipitation accumulation), creating animations, and optionally uploading data to FTP servers. The API provides endpoints to view plots, monitor updates, and manage asynchronous processing tasks.

## Key Features

* **Automated Monitoring (CLI):** Watches a specified input directory for new `.scnx.gz` scan files via `radproc run`.
* **Core Processing:** Reads (`xarray`, `xradar`), preprocesses, and georeferences (`pyproj`) radar data.
* **Visualization:** Generates configurable PPI plot images (`matplotlib`, `cartopy`) for specified radar variables (e.g., RATE). Saves plots historically and updates a "realtime" image for API access.
* **Time Series Extraction:** Extracts time series data for specified variables at user-defined points (Lat/Lon/Elevation) and saves to CSV files. Can be run historically (`radproc timeseries ...`) or automatically during `radproc run`.
* **Precipitation Accumulation:** Calculates accumulated precipitation over user-defined intervals from rate time series CSVs (`radproc accumulate ...` or via API job).
* **Animation:** Creates animations (GIF/MP4) from generated plot images (`radproc animate ...` or via API job). Uses `imageio`.
* **FTP Upload (Optional):** Uploads original scan files and/or generated images to FTP servers via a persistent SQLite-based queue (`core/utils/upload_queue.py`) with retries. Managed by `radproc run`.
* **FastAPI Web API:**
    * Provides HTTP endpoints to access status, configured points, latest ("realtime") plots, and specific historical plots.
    * Offers a Server-Sent Events (SSE) stream (`/plots/stream/updates`) for real-time notification of plot updates (using `watchfiles`).
    * Allows submitting background jobs (timeseries generation, accumulation, animation) via a task queue (`Huey` with SQLite backend).
    * Provides endpoints to check job status and retrieve results/data (CSV, JSON, files).
    * Includes download endpoints for generated data products.
* **Background Task Queue:** Uses `Huey` with an SQLite backend (`huey.db`) for handling asynchronous API tasks (e.g., historical timeseries generation). Requires a separate `huey_consumer` process.
* **Configuration Driven:** Behavior (paths, FTP, plot styles, points, API settings, queue paths) is controlled via external YAML files (`config/`).
* **Cross-Platform:** Designed to run on both Linux and Windows (with considerations for Huey/SQLite worker configuration).

## Project Structure Highlights

* `radproc/core/`: Core backend logic (I/O, processing, plotting, FTP, analysis, animation).
* `radproc/cli/`: Command-line interface (`radproc` command).
* **`radproc/api/`**: FastAPI application code.
    * `main.py`: FastAPI app instance, lifespan manager (starts file watcher).
    * `routers/`: Endpoint definitions (status, points, plots, jobs, downloads).
    * `schemas/`: Pydantic models for request/response validation.
    * `dependencies.py`: FastAPI dependencies (e.g., config access).
    * `state.py`: Shared state (e.g., SSE update queue).
* **`radproc/huey_config.py`**: Huey instance configuration (uses SQLite backend).
* **`radproc/tasks.py`**: Huey background task definitions (wrapping core logic).
* `config/`: Holds YAML configuration files:
    * `app_config.yaml`: Paths (input, output, images, *realtime_image_dir*, timeseries, logs, cache, animation), logging settings (*api_log_file*), main behaviour, animation defaults.
    * `ftp_config.yaml`: FTP servers, modes, queue settings (`queue_db_path`).
    * `plot_styles.yaml`: Plotting appearance.
    * `radar_params.yaml`: Radar hardware info (e.g., `azimuth_step`).
    * `points.yaml`: Points for time series extraction.
    * `huey_config.yaml` (*Implied/Optional*): Could hold Huey DB path instead of `app_config.yaml` if preferred.
* `scripts/`: Setup (`setup_*.sh`/`.bat`), runtime wrappers (`frad-proc*`), service templates (`radproc.service`).
* `data/`: Default location for persistent queues (`upload_queue.db`, `huey.db`).
* `log/`: Default location for logs (`radproc_activity.log`, `radproc_api.log`).
* `cache/`: Default location for map tiles and temporary animation frames.
* *(User Configured)* `timeseries_dir`, `images_dir`, `animations`, etc.: Locations for generated data (defined in `app_config.yaml`).

## Configuration

RadProc relies heavily on YAML configuration files in `config/`. Create actual `.yaml` files from the `.template` files provided.

* `app_config.yaml`: General settings (input/output/image paths), logging levels/files (separate `log_file` and `api_log_file` recommended), `realtime_image_dir` (for API), `timeseries_dir`, `enable_timeseries_updates`, animation defaults, etc.
* `ftp_config.yaml`: FTP server details (host, user), upload modes, `queue_db_path`, retry settings. **NO PASSWORDS HERE**.
* `points.yaml`: Define points of interest (`name`, `latitude`, `longitude`, `elevation`, `variable`, `description`).
* `radar_params.yaml`: Radar info like `azimuth_step`.
* `plot_styles.yaml`: Plotting appearance (colors, norms, map tiles).
* `huey_config.yaml` (*Optional*): Can define `huey.db_path` here, otherwise uses default or path from `app_config.yaml` if implemented that way.

**Important: Secure Password Handling**

* FTP passwords are NEVER stored in config files.
* Set environment variables: `FTP_PASSWORD_<ALIAS_UPPERCASE>=your_password`. See Setup.

**Important: Huey SQLite Configuration**

* The Huey task queue uses an SQLite database (`data/huey.db` by default).
* For reliable operation, especially with multiple API requests or potential worker concurrency, **it is crucial to configure SQLite's Write-Ahead Logging (WAL) mode and a busy timeout.**
* The included API (`radproc/api/main.py`) attempts to apply these settings (`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;`) automatically on startup via the `lifespan` manager and `radproc.huey_config.apply_huey_pragmas()`. Verify this occurs in the API startup logs. Alternatively, run a setup script or manual command to configure the `huey.db` file once after creation.

## Setup

1.  **Prerequisites:** Python 3.8+, `python3-venv` (Linux recommended), Git. For animations, `ffmpeg` might need to be installed system-wide if `imageio-ffmpeg` doesn't bundle it.
2.  **Clone:** `git clone <your-repository-url>`
3.  **Run Setup Script:**
    * Linux: `cd scripts && ./setup_linux.sh`
    * Windows: `cd scripts && setup_windows.bat`
    * (Creates `.venv`, installs requirements from `requirements.txt` including `fastapi`, `uvicorn`, `huey`, `peewee`, `watchfiles`, etc., creates wrappers, sets up default directories).
4.  **Configure:** Copy `config/*.yaml.template` files to `config/*.yaml` and edit them. Set required paths in `app_config.yaml` (`input_dir`, `output_dir`, `images_dir`, `realtime_image_dir`, `timeseries_dir`, log paths). Configure points, FTP, etc.
5.  **Set Environment Variables:** Set `FTP_PASSWORD_...` variables for any configured FTP servers.

## Usage

### Command Line Interface (CLI)

Activate virtual environment (`source .venv/bin/activate` or `.venv\Scripts\activate.bat`) or use the wrapper scripts (`frad-proc`, `frad-proc.bat`) potentially added to your PATH by the setup scripts.

* **Show Help:**
    ```bash
    radproc --help
    ```

* **Run Continuous Monitoring:**
    * Processes new scans, generates plots (including realtime), optionally updates timeseries, starts FTP upload worker.
    * This is the primary mode for data ingestion and basic product generation.
    ```bash
    radproc run
    ```
    *(Press Ctrl+C to stop)*

* **Reprocess Historical Plots:**
    ```bash
    radproc reprocess <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM>
    ```

* **Generate/Update Point Time Series (Historical):**
    ```bash
    radproc timeseries <point_name> <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> [--variable VAR]
    ```

* **Calculate Accumulated Precipitation:**
    * Ensures source rate timeseries is up-to-date for the range first.
    ```bash
    radproc accumulate <point_name> <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> <interval> [--variable RATE_VAR] [--output-file PATH]
    ```
    *(Example interval: '1H', '15min', '1D')*

* **Create Animation File:**
    ```bash
    radproc animate <variable> <elevation> <start_YYYYMMDD_HHMM> <end_YYYYMMDD_HHMM> <output_file.[gif|mp4|...]> [--extent LONMIN LONMAX LATMIN LATMAX] [--no-watermark] [--fps FPS]
    ```

### Web API

The API provides endpoints for accessing data and managing background jobs.

1.  **Run the API Server:**
    ```bash
    # Activate virtual environment
    # Needs to run separately from 'radproc run' and 'huey_consumer'
    uvicorn radproc.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    * Access the interactive documentation (Swagger UI) at `http://localhost:8000/docs`.
    * Access the ReDoc documentation at `http://localhost:8000/redoc`.

2.  **Run the Huey Worker (for API Background Jobs):**
    * This process executes tasks queued by the API (e.g., historical timeseries). It needs to run separately from the API server and the `radproc run` monitor.
    ```bash
    # Activate virtual environment
    # CRITICAL for SQLite: Use -k thread and low worker count (-w 1 or -w 2)
    # Ensure PRAGMAs (WAL mode, busy_timeout) are set on huey.db
    huey_consumer radproc.huey_config.huey -k thread -w 1 -v
    ```
    * See "Running as a Service" for production setups.

3.  **API Endpoint Highlights:**
    * `GET /api/v1/status`: Basic API status.
    * `GET /api/v1/points`: List configured points of interest.
    * `GET /api/v1/plots/realtime/{variable}/{elevation}`: Get the latest plot image.
    * `GET /api/v1/plots/historical/{variable}/{elevation}/{datetime_str}`: Get a specific past plot image.
    * `GET /api/v1/plots/frames`: Get list of available frame identifiers for a sequence (client-side animation).
    * `GET /api/v1/plots/stream/updates`: Server-Sent Events stream for real-time plot update notifications.
    * `POST /api/v1/jobs/{timeseries|accumulation|animation}`: Submit background jobs.
    * `GET /api/v1/jobs/{task_id}/status`: Check job status (PENDING, SUCCESS, FAILURE, REVOKED).
    * `GET /api/v1/jobs/{timeseries|accumulation|animation}/{task_id}/data`: Retrieve job results/data files.
    * `GET /api/v1/downloads/{timeseries|accumulation|animation}/{filename}`: Download generated files.

## Running as a Service

To run continuously in production, you need to manage multiple processes:

1.  **RadProc Monitor (`radproc run`)**: For ingesting new scans and core processing/plotting/FTP. (Optional if only using API jobs for processing).
2.  **FastAPI Server (`uvicorn ...`)**: To serve the web API.
3.  **Huey Worker (`huey_consumer ...`)**: To process background jobs submitted via the API.

* **Linux:** Use `systemd`. Create separate service files for Uvicorn and `huey_consumer`. You might adapt `scripts/radproc.service` or create new ones. Ensure correct user, working directory, environment variables (including FTP passwords!), and the recommended Huey consumer flags (`-k thread -w 1`). Ensure SQLite PRAGMAs are set.
* **Windows:** Use Task Scheduler or a tool like NSSM (Non-Sucking Service Manager) to run the Uvicorn command and the `huey_consumer` command reliably as services. Use the `.bat` wrapper scripts or direct Python calls within the virtual environment.

## Logging

* Configured in `app_config.yaml`.
* Supports separate log files for the CLI (`log_file`) and API (`api_log_file`).
* Includes console logging with configurable levels.
* Uses rotating file handlers to manage log size.
* Huey consumer output can be directed to a file using the `-l` flag or captured by systemd/supervisor.

## Dependencies

See `requirements.txt`. Key libraries include:
`fastapi`, `uvicorn`, `huey`, `peewee` (for Huey/SQLite), `watchfiles` (for API SSE), `xarray`, `xradar`, `pandas`, `matplotlib`, `cartopy`, `pyyaml`, `watchdog` (for CLI monitor), `pyproj`, `imageio`, `imageio-ffmpeg`.