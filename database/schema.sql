-- RadProc Database Schema
-- Version: 1.1
-- Description: Updated schema with scan log and modifications to points/variables.

-- Ensure connection to the correct database before running this, e.g.,
-- \c radproc_db

BEGIN;

-- Drop existing tables in reverse order of dependency (if recreating)
-- NOTE: Use with caution, especially in production!
-- DROP TABLE IF EXISTS timeseries_data CASCADE;
-- DROP TABLE IF EXISTS radproc_scan_log CASCADE;
-- DROP TABLE IF EXISTS radproc_points CASCADE;
-- DROP TABLE IF EXISTS radproc_variables CASCADE;

-- Table for storing point definitions (Updated: removed default_variable_name)
CREATE TABLE IF NOT EXISTS radproc_points (
    point_id SERIAL PRIMARY KEY,
    point_name VARCHAR(255) UNIQUE NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    target_elevation DOUBLE PRECISION NOT NULL,
    description TEXT,
    cached_azimuth_index INTEGER,
    cached_range_index INTEGER
);

COMMENT ON TABLE radproc_points IS 'Stores definitions of points of interest for time series extraction.';
COMMENT ON COLUMN radproc_points.point_name IS 'Unique user-defined name for the point.';

-- Table for storing variable definitions (Updated: units/description nullable)
CREATE TABLE IF NOT EXISTS radproc_variables (
    variable_id SERIAL PRIMARY KEY,
    variable_name VARCHAR(50) UNIQUE NOT NULL,
    units VARCHAR(50) NULL, -- Made NULLABLE
    description TEXT NULL    -- Made NULLABLE
);

COMMENT ON TABLE radproc_variables IS 'Stores definitions of radar variables (e.g., RATE, DBZH).';
COMMENT ON COLUMN radproc_variables.variable_name IS 'Short name of the variable (e.g., RATE, DBZH).';
COMMENT ON COLUMN radproc_variables.units IS 'Units of the variable (Optional).';
COMMENT ON COLUMN radproc_variables.description IS 'Longer description of the variable (Optional).';

-- Table for storing time series data
CREATE TABLE IF NOT EXISTS timeseries_data (
    timestamp TIMESTAMPTZ NOT NULL, -- Stores UTC
    point_id INTEGER NOT NULL,
    variable_id INTEGER NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    source_version VARCHAR(50) NOT NULL DEFAULT 'raw', -- Tracks the source (e.g., 'raw', 'v1.0')
    PRIMARY KEY (timestamp, point_id, variable_id, source_version), -- Updated Primary Key
    CONSTRAINT fk_point
        FOREIGN KEY(point_id)
        REFERENCES radproc_points(point_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_variable
        FOREIGN KEY(variable_id)
        REFERENCES radproc_variables(variable_id)
        ON DELETE CASCADE
);

COMMENT ON TABLE timeseries_data IS 'Stores time series data values for specific points and variables.';
COMMENT ON COLUMN timeseries_data.timestamp IS 'Timestamp of the data reading, stored in UTC.';

-- Table for storing scan log data
CREATE TABLE IF NOT EXISTS radproc_scan_log (
    scan_log_id SERIAL PRIMARY KEY,
    filepath VARCHAR(1024) UNIQUE NOT NULL,
    precise_timestamp TIMESTAMPTZ NOT NULL,
    elevation DOUBLE PRECISION NOT NULL,
    scan_sequence_number INTEGER NOT NULL,
    volume_identifier TIMESTAMPTZ NULL,
    nominal_filename_timestamp TIMESTAMPTZ NULL,
    processed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_volume_scan UNIQUE (volume_identifier, scan_sequence_number, elevation)
);


COMMENT ON TABLE radproc_scan_log IS 'Logs metadata for each processed radar scan file.';
COMMENT ON COLUMN radproc_scan_log.filepath IS 'Full path to the scan file after processing/moving.';
COMMENT ON COLUMN radproc_scan_log.precise_timestamp IS 'Canonical time extracted from within the scan data.';
COMMENT ON COLUMN radproc_scan_log.elevation IS 'Elevation angle of the scan.';
COMMENT ON COLUMN radproc_scan_log.scan_sequence_number IS 'The _N sequence number from the original filename.';
COMMENT ON COLUMN radproc_scan_log.volume_identifier IS 'Timestamp of the _0 scan in the volume; set by group-volumes job.';
COMMENT ON COLUMN radproc_scan_log.nominal_filename_timestamp IS 'Timestamp parsed from the original filename (for reference).';
COMMENT ON COLUMN radproc_scan_log.processed_at IS 'When this scan was added to the database.';


-- Create the new table for tracking processed volumes with the correct data type
CREATE TABLE IF NOT EXISTS radproc_processed_volumes (
    processed_volume_id SERIAL PRIMARY KEY,
    -- This column MUST match the data type of volume_identifier in radproc_scan_log
    volume_identifier TIMESTAMPTZ NOT NULL UNIQUE,
    filepath VARCHAR(1024) UNIQUE NOT NULL,
    processing_version VARCHAR(50) NOT NULL,
    processed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE radproc_processed_volumes IS 'Tracks processed data volumes (e.g., CfRadial2) generated from multiple raw scans.';
COMMENT ON COLUMN radproc_processed_volumes.volume_identifier IS 'The unique volume identifier (a timestamp), shared by all raw scans in the volume.';


-- Create/Recreate indexes for performance
-- Indexes for timeseries_data
CREATE INDEX IF NOT EXISTS idx_timeseries_data_point_var_time ON timeseries_data (point_id, variable_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_timeseries_data_timestamp ON timeseries_data (timestamp);

-- Index for radproc_points
CREATE INDEX IF NOT EXISTS idx_radproc_points_point_name ON radproc_points (point_name);

-- Index for radproc_variables
CREATE INDEX IF NOT EXISTS idx_radproc_variables_variable_name ON radproc_variables (variable_name);

-- Indexes for radproc_scan_log (NEW)
CREATE INDEX IF NOT EXISTS idx_radproc_scan_log_filepath ON radproc_scan_log (filepath);
CREATE INDEX IF NOT EXISTS idx_radproc_scan_log_precise_timestamp ON radproc_scan_log (precise_timestamp);
CREATE INDEX IF NOT EXISTS idx_radproc_scan_log_elevation ON radproc_scan_log (elevation);
CREATE INDEX IF NOT EXISTS idx_radproc_scan_log_scan_sequence_number ON radproc_scan_log (scan_sequence_number);
CREATE INDEX IF NOT EXISTS idx_radproc_scan_log_volume_identifier ON radproc_scan_log (volume_identifier NULLS FIRST);

-- Index for faster lookups on processed volumes by their volume_identifier
CREATE INDEX IF NOT EXISTS idx_radproc_processed_volumes_volume_identifier ON radproc_processed_volumes (volume_identifier);

COMMIT;