-- RadProc Database Schema
-- Version: 1.0
-- Description: Initial schema for storing points, variables, and time series data.

-- Ensure connection to the correct database before running this, e.g.,
-- \c radproc_db

BEGIN;

-- Table for storing point definitions
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
COMMENT ON COLUMN radproc_points.point_name IS 'Unique user-defined name for the point (e.g., from points.yaml).';

-- Table for storing variable definitions
CREATE TABLE IF NOT EXISTS radproc_variables (
    variable_id SERIAL PRIMARY KEY,
    variable_name VARCHAR(50) UNIQUE NOT NULL,
    units VARCHAR(50),
    description TEXT
);

COMMENT ON TABLE radproc_variables IS 'Stores definitions of radar variables (e.g., RATE, DBZH).';
COMMENT ON COLUMN radproc_variables.variable_name IS 'Short name of the variable (e.g., RATE, DBZH).';

-- Table for storing time series data
CREATE TABLE IF NOT EXISTS timeseries_data (
    timestamp TIMESTAMPTZ NOT NULL, -- Stores UTC
    point_id INTEGER NOT NULL,
    variable_id INTEGER NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (timestamp, point_id, variable_id),
    CONSTRAINT fk_point
        FOREIGN KEY(point_id)
        REFERENCES radproc_points(point_id)
        ON DELETE CASCADE, -- Or ON DELETE RESTRICT based on desired behavior
    CONSTRAINT fk_variable
        FOREIGN KEY(variable_id)
        REFERENCES radproc_variables(variable_id)
        ON DELETE CASCADE -- Or ON DELETE RESTRICT
);

COMMENT ON TABLE timeseries_data IS 'Stores time series data values for specific points and variables.';
COMMENT ON COLUMN timeseries_data.timestamp IS 'Timestamp of the data reading, stored in UTC.';

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_timeseries_data_point_var_time ON timeseries_data (point_id, variable_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_timeseries_data_timestamp ON timeseries_data (timestamp);
-- Index on point_name in radproc_points is implicitly created by UNIQUE constraint, but explicit one is fine
CREATE INDEX IF NOT EXISTS idx_radproc_points_point_name ON radproc_points (point_name);
CREATE INDEX IF NOT EXISTS idx_radproc_variables_variable_name ON radproc_variables (variable_name);


-- Optional: Insert some default variables if known beforehand
-- INSERT INTO radproc_variables (variable_name, units, description) VALUES
-- ('RATE', 'mm/hr', 'Rainfall Rate'),
-- ('DBZH', 'dBZ', 'Reflectivity (Horizontal Polarization)')
-- ON CONFLICT (variable_name) DO NOTHING;

COMMIT;