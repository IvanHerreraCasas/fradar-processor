# core/visualization/plotter.py

import logging
import io
from typing import Optional, Union, Tuple, Dict, Any
import warnings
import os
from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import shapely.geometry
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from ..config import get_config, get_setting
from ..db_manager import get_connection, release_connection
from ..processor import logger, process_new_scan

try:
    import pyproj  # Needed for type checking and conversion
except ImportError:
    pyproj = None  # Handle gracefully if pyproj is not installed

from datetime import datetime

# Import the style data class
from .style import PlotStyle

# xradar is used for getting CRS, ensure it's importable if not implicitly via xarray
try:
    import xradar.georeference
except ImportError:
    pass  # Assume georeference was done earlier and CRS is in attrs

from ..utils.geo import get_dataset_crs 

logger = logging.getLogger(__name__)

# --- Plotting Function ---


def create_ppi_image(
    ds: xr.Dataset,
    variable: str,
    style: PlotStyle,
    watermark_path: Optional[str] = None,
    watermark_zoom: float = 0.05,
    coverage_radius_km: Optional[float] = 70.0,
    plot_extent: Optional[Tuple[float, float, float, float]] = None
) -> Optional[bytes]:
    """
    Generates a PPI (Plan Position Indicator) plot for a given variable in a dataset.

    Args:
        ds: Georeferenced xarray.Dataset containing the variable to plot (single scan).
            Must include 'time', 'elevation', 'latitude', 'longitude', 'x', 'y' coordinates.
        variable: The name of the data variable within the dataset to plot (e.g., "RATE").
        style: A PlotStyle object containing the cmap, norm, map_tile, and variable_dname.
        watermark_path: Optional path to a watermark image file.
        watermark_zoom: Zoom factor for the watermark image.
        coverage_radius_km: Radius in km for the radar coverage circle. Set to None to disable.
        plot_extent: Optional tuple (LonMin, LonMax, LatMin, LatMax) to override the default plot extent. Coordinates are geographic (PlateCarree).

    Returns:
        Bytes representing the generated PNG image, or None if plotting fails.
    """
    fig = None  # Initialize fig to None for proper closing in finally block
    cartopy_crs_instance = None # Define higher up for use in extent setting
    try:
        # --- Input Validation ---
        if variable not in ds.data_vars:
            logger.error(f"Variable '{variable}' not found in the provided dataset.")
            return None
        required_coords = {"time", "elevation", "latitude", "longitude", "x", "y"}
        if not required_coords.issubset(ds.coords):
            missing_coords = required_coords - set(ds.coords)
            logger.error(
                f"Dataset is missing required coordinates for plotting: {missing_coords}"
            )
            return None

        # --- Extract Metadata ---
        try:
            dt_np = np.datetime64(ds['time'].values.item())
            dt_utc = dt_np.astype(datetime) # Python datetime object in UTC
            elevation = float(ds['elevation'].values.item()) # Default float formatting
            radar_lat = float(ds['latitude'].values.item())
            radar_lon = float(ds['longitude'].values.item())

            datetime_utc_str = dt_utc.strftime("%Y-%m-%d %H:%M") # Original format without "UTC" text
            # Calculate Local Time (assuming fixed UTC-5 offset, same as original)
            # NOTE: This fixed offset might not be correct for all situations/locations.
            # Consider making the offset configurable in the future.
            dt_lt = dt_utc - timedelta(hours=5)
            datetime_lt_str = dt_lt.strftime("%Y-%m-%d %H:%M")
        except Exception as meta_err:
            logger.error(
                f"Could not extract metadata (time, elevation, lat, lon) from dataset: {meta_err}"
            )
            return None  # Cannot proceed without metadata

        # --- Determine and Convert CRS ---
        try:
            proj_crs = get_dataset_crs(ds)  
            if proj_crs is None: raise ValueError("CRS not found")

            # --- Explicit Conversion to Cartopy Projection ---
            cartopy_crs_instance = None
            if pyproj and isinstance(proj_crs, pyproj.crs.CRS):
                #logger.debug(
                #    f"Detected pyproj.CRS. Attempting conversion to Cartopy CRS."
                #)
                # Use CF conventions for parameter names
                cf_params = proj_crs.to_cf()
                grid_mapping_name = cf_params.get("grid_mapping_name")

                if grid_mapping_name == "azimuthal_equidistant":
                    clon = cf_params.get("longitude_of_projection_origin")
                    clat = cf_params.get("latitude_of_projection_origin")
                    globe = None
                    # Try to create globe from CF params if available
                    if "earth_radius" in cf_params:
                        globe = ccrs.Globe(
                            semimajor_axis=cf_params["earth_radius"],
                            semiminor_axis=cf_params["earth_radius"],
                            ellipse=None,
                        )
                    elif (
                        "semi_major_axis" in cf_params
                        and "inverse_flattening" in cf_params
                    ):
                        globe = ccrs.Globe(
                            semimajor_axis=cf_params["semi_major_axis"],
                            inverse_flattening=cf_params["inverse_flattening"],
                            ellipse=None,
                        )
                    elif (
                        "semi_major_axis" in cf_params
                        and "semi_minor_axis" in cf_params
                    ):
                        globe = ccrs.Globe(
                            semimajor_axis=cf_params["semi_major_axis"],
                            semiminor_axis=cf_params["semi_minor_axis"],
                            ellipse=None,
                        )
                    else:
                        # Default to WGS84 if globe info is missing, which is common
                        #logger.debug(
                        #    "Globe parameters not found in CF dict, assuming WGS84 globe for AzimuthalEquidistant."
                        #)
                        globe = ccrs.Globe(ellipse="WGS84")

                    if clon is not None and clat is not None:
                        #logger.info(
                        #    f"Converting AzimuthalEquidistant from pyproj (Lon={clon}, Lat={clat})"
                        #)
                        cartopy_crs_instance = ccrs.AzimuthalEquidistant(
                            central_longitude=clon, central_latitude=clat, globe=globe
                        )
                    else:
                        raise ValueError(
                            "Could not extract central lon/lat from pyproj AzimuthalEquidistant CRS."
                        )

                # Add elif for other common radar projections if needed (e.g., LambertConformal)
                # elif grid_mapping_name == 'lambert_conformal_conic':
                #     ... extract params for ccrs.LambertConformal ...

                else:
                    raise ValueError(
                        f"Unsupported pyproj grid mapping for Cartopy conversion: {grid_mapping_name}"
                    )

            elif isinstance(proj_crs, ccrs.Projection):
                #logger.debug("CRS is already a Cartopy Projection.")
                cartopy_crs_instance = proj_crs  # Use directly
            else:
                # Handle case where proj_crs might be a string (less likely now) or other type
                if isinstance(proj_crs, str):
                    #logger.warning(
                    #    "CRS found as string. Attempting direct use with Cartopy (may fail)."
                    #)
                    # This is risky, Cartopy often needs projection parameters.
                    # Consider parsing the string if possible (e.g., EPSG code)
                    try:
                        cartopy_crs_instance = ccrs.Projection(
                            proj_crs
                        )  # Attempt direct creation
                    except Exception as e:
                        raise ValueError(
                            f"Failed to create Cartopy Projection from CRS string '{proj_crs}': {e}"
                        )
                else:
                    raise TypeError(
                        f"Unsupported CRS type received: {type(proj_crs)}. Expected Cartopy Projection or convertible pyproj CRS."
                    )

            # --- CRS conversion complete ---

        except ImportError:
            logger.error("pyproj is required for CRS handling. Please install it.")
            return None
        except Exception as crs_err:
            logger.error(
                f"Could not determine/convert Cartopy CRS from dataset: {crs_err}",
                exc_info=True,
            )
            return None

        # --- Plot Setup ---
        # Use Agg backend for non-interactive plotting, good for saving bytes
        with plt.ioff():  # Turn off interactive mode temporarily
            fig = plt.figure(figsize=(10, 8))
            # Use PlateCarree for extent/gridlines, but data transform is cartopy_crs
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

            # --- Add Map Background (Optional) ---
            if style.map_tile:
                try:
                    # Determine appropriate zoom level (can be adaptive later)
                    zoom_level = 9
                    ax.add_image(
                        style.map_tile,
                        zoom_level,
                        alpha=0.5,
                        cmap="gray",
                    )  # Removed alpha, cmap - let tile handle its look
                except Exception as tile_err:
                    logger.warning(f"Could not add map tile background: {tile_err}")
            else:
                # Add basic coastlines/borders if no tile is available/desired
                try:
                    import cartopy.feature as cfeature

                    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                    # ax.stock_img() # Alternative very basic background
                except ImportError:
                    logger.warning(
                        "cartopy.feature not available. Adding basic background failed."
                    )

            # +++ Set Extent Logic +++
            extent_set_successfully = False
            if plot_extent and len(plot_extent) == 4:
                try:
                    logger.debug(f"Setting user-defined plot extent (PlateCarree): {plot_extent}")
                    ax.set_extent(plot_extent, crs=ccrs.PlateCarree()) # Extent is defined in PlateCarree
                    extent_set_successfully = True
                except Exception as extent_err:
                     logger.warning(f"Failed to apply user-defined extent {plot_extent}: {extent_err}. Using default.")
                     # Fallback to default logic

            if not extent_set_successfully: # Run default if no user extent or if user extent failed
                 try:
                     # Ensure ds has 'x' and 'y' coords before accessing
                     if 'x' in ds.coords and 'y' in ds.coords and cartopy_crs_instance:
                         x_min, x_max = ds["x"].min().item(), ds["x"].max().item()
                         y_min, y_max = ds["y"].min().item(), ds["y"].max().item()
                         pad_x = 0
                         pad_y = 0
                         logger.debug("Setting plot extent based on data x/y coordinates.")
                         ax.set_extent(
                             [x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y],
                             crs=cartopy_crs_instance, # Set extent in the DATA's CRS
                         )
                     else:
                          logger.warning("Cannot set default extent: Missing x/y coords or data CRS.")
                          # Let Cartopy determine extent automatically
                 except Exception as extent_err:
                     logger.warning(f"Could not determine data extent from x/y coords: {extent_err}. Using automatic extent.")
            # +++ End Extent Logic +++

            logger.info(f"Attempting to plot variable '{variable}'...")
            #logger.info(
            #    f"Using Cartopy CRS for transform: {cartopy_crs_instance}"
            #)  # Log the actual Cartopy object
            #logger.info(f"Type of Cartopy CRS object: {type(cartopy_crs_instance)}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                quadmesh = ds[variable].plot(
                    ax=ax,
                    x="x",
                    y="y",
                    cmap=style.cmap,
                    norm=style.norm,
                    transform=cartopy_crs_instance,  # <<< USE THE CONVERTED CARTOPY OBJECT
                    #add_colorbar=True,
                    cbar_kwargs=dict(
                        pad=0.075, shrink=0.75, #label=style.variable_dname
                    ),
                )

            # --- Add Map Features ---
            # Gridlines (labels use PlateCarree)
            try:
                grid_lines = ax.gridlines(
                    draw_labels=True,
                    crs=ccrs.PlateCarree(),
                    # --- REMOVED dms=True, x_inline, y_inline ---
                )
                grid_lines.top_labels = False
                grid_lines.right_labels = False
            except Exception as grid_err:
                logger.warning(f"Could not draw gridlines: {grid_err}")

            # Coverage Circle
            if coverage_radius_km is not None and coverage_radius_km > 0:
                try:
                    geodesic = Geodesic()
                    circle_points = geodesic.circle(
                        lon=radar_lon,
                        lat=radar_lat,
                        radius=coverage_radius_km * 1000,  # Convert km to meters
                        n_samples=100,
                        endpoint=False,
                    )
                    coverage_area = shapely.geometry.Polygon(circle_points)
                    ax.add_geometries(
                        [coverage_area],
                        crs=ccrs.PlateCarree(),  # Circle defined in lat/lon
                        facecolor="none",
                        edgecolor="black",
                        linewidth=0.5,
                        linestyle="--",
                    )
                except Exception as circle_err:
                    logger.warning(f"Could not draw coverage circle: {circle_err}")

            # --- Title ---
            # Include variable display name, elevation, time
            ax.set_title(
                f"{style.variable_dname} a EL: {elevation}°\n{datetime_utc_str} UTC ({datetime_lt_str} LT)"
            )
            
            fig.tight_layout()  # Adjust layout for better spacing

            logger.info(f"Watermark: {watermark_path}")
            # --- Watermark ---
            if watermark_path and os.path.exists(watermark_path):
                logger.info(f"Watermark exists: {watermark_path}")
                try:
                    watermark_img = plt.imread(watermark_path)
                    imagebox = OffsetImage(watermark_img, zoom=watermark_zoom)

                    # Position below colorbar (relative to figure fraction)
                    cbar = quadmesh.colorbar
                    if cbar:
                        cbar_ax = cbar.ax
                        cbar_pos = (
                            cbar_ax.get_position()
                        )  # Get position after layout adjustments

                        # Anchor point: Bottom-right of the colorbar axes in figure coordinates
                        anchor_x = cbar_pos.x1
                        anchor_y = cbar_pos.y0 - 0.02  # Slightly below the colorbar

                        ab = AnnotationBbox(
                            imagebox,
                            (anchor_x, anchor_y),
                            xycoords="figure fraction",
                            box_alignment=(
                                1.0,
                                1.0,
                            ),  # Align top-right of image slightly above anchor
                            frameon=False,
                            #zorder=10,
                        )  # Ensure watermark is on top
                        ax.add_artist(ab)
                    else:
                        logger.warning(
                            "Colorbar not found, cannot position watermark relative to it."
                        )
                        # Alternative positioning could be added here (e.g., bottom-right of figure)

                except Exception as wm_err:
                    logger.warning(f"Could not add watermark: {wm_err}")
            elif watermark_path:
                logger.info(f"Watermark file not found: {watermark_path}")

            # --- Final Touches & Saving ---
            # fig.tight_layout(pad=0.5) # Apply tight layout carefully, can interfere with annotations

            buffer = io.BytesIO()
            fig.savefig(
                buffer, format="png", dpi=300, 
            )  # Adjust dpi as needed
            buffer.seek(0)
            image_bytes = buffer.getvalue()
            logger.info(
                f"Successfully created PPI image for {variable} at {datetime_utc_str} EL {elevation}"
            )
            return image_bytes

    except Exception as e:
        logger.error(f"Failed to create PPI plot for {variable}: {e}", exc_info=True)
        return None
    finally:
        # Ensure the figure is closed to release memory
        if fig is not None:
            plt.close(fig)


def generate_historical_plots(start_dt: datetime, end_dt: datetime, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Finds processed radar scan files (from radproc_scan_log) within a date range
    and regenerates plots for them.
    """
    # This function will need to be updated to query radproc_scan_log
    # via db_manager to get filepaths, instead of get_filepaths_in_range.
    logger.warning("generate_historical_plots: Needs update to use radproc_scan_log via db_manager.")

    # --- TEMPORARY: Keep old logic using get_filepaths_in_range for now ---
    # --- THIS WILL BE REPLACED ---
    if config is None: config = get_config()
    processed_data_dir = get_setting('app.output_dir')
    if not processed_data_dir:
        logger.error("Config missing 'app.output_dir'. Cannot find historical files for plotting.")
        return

    conn_temp = None
    try:
        conn_temp = get_connection()
        # Query radproc_scan_log for filepaths based on elevation and time
        # This requires knowing which elevations to reprocess, or reprocessing all.
        # For simplicity, let's assume we reprocess for all elevations found in the log for now.
        # A more targeted approach might take elevation as an argument.

        # This is a placeholder for what query_scan_log_for_timeseries_processing does,
        # but for plotting we might not need a specific elevation filter initially, or we might
        # iterate through distinct elevations in the log for the time range.
        # For now, this function remains largely non-functional until fully adapted.

        # TEMPORARY: Using a dummy list to avoid breaking structure
        # In a real scenario, you'd query radproc_scan_log
        logger.info(f"Attempting to find scans in log for reprocessing plots: {start_dt} to {end_dt}")
        all_scans_in_range = []  # This should be populated by a db_manager query

        # Example of how you might get distinct elevations and then query per elevation
        # distinct_elevs_sql = "SELECT DISTINCT elevation FROM radproc_scan_log WHERE precise_timestamp >= %s AND precise_timestamp <= %s;"
        # cur = conn_temp.cursor()
        # cur.execute(distinct_elevs_sql, (start_dt, end_dt))
        # for row in cur.fetchall():
        #    target_elev = row[0]
        #    scans_for_elev = query_scan_log_for_timeseries_processing(conn_temp, target_elev, start_dt, end_dt) # Reuses existing func
        #    all_scans_in_range.extend(scans_for_elev) # Collects (filepath, precise_ts, scan_log_id)

        # For now, let's assume `all_scans_in_range` is populated with (filepath, precise_datetime_from_log)
        # The `process_new_scan` function is the main processing unit.
        # We are calling it in a 'reprocessing' context so it shouldn't move files
        # or re-log to scan_log (ideally).

        if not all_scans_in_range:  # Placeholder check
            logger.info("No historical scan files found in radproc_scan_log for the specified range for plotting.")
            return

        logger.info(f"Found {len(all_scans_in_range)} scan entries in log to reprocess for plots.")
        processed_count = 0;
        failed_count = 0
        for scan_filepath, _scan_dt_from_log, _scan_log_id in all_scans_in_range:  # Use filepath from log
            logger.info(f"Reprocessing plots for historical file: {scan_filepath}")

            # Create a temporary config to disable file moving and timeseries updates for this plot-only run
            temp_config_dict = get_config().copy()  # Get a mutable copy
            temp_config_dict['app'] = temp_config_dict.get('app', {}).copy()  # Ensure 'app' key exists and is mutable
            temp_config_dict['app']['move_file_after_processing'] = False
            temp_config_dict['app']['enable_timeseries_updates'] = False
            # Also prevent re-logging to scan_log - process_new_scan would need a flag for this.
            # For now, it will try to re-log but ON CONFLICT should handle it.

            try:
                # process_new_scan will read the file again.
                success = process_new_scan(scan_filepath, config_override=temp_config_dict)
                if success:
                    processed_count += 1
                else:
                    failed_count += 1; logger.warning(f"Failed to reprocess plots for: {scan_filepath}")
            except Exception as e:
                failed_count += 1;
                logger.error(f"Critical error reprocessing plots for {scan_filepath}: {e}", exc_info=True)
        logger.info(f"Historical plot reprocessing finished. Processed: {processed_count}, Failed: {failed_count}")

    except ConnectionError as ce:
        logger.error(f"DB Connection error in generate_historical_plots: {ce}")
    except Exception as e:
        logger.error(f"Error in generate_historical_plots: {e}", exc_info=True)
    finally:
        if conn_temp: release_connection(conn_temp)
