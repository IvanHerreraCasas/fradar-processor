# core/visualization/plotter.py

# ... (imports, logger) ...
import cartopy.crs as ccrs # Ensure cartopy CRS is imported

# --- Plotting Function ---
def create_ppi_image(
    ds: xr.Dataset,
    variable: str,
    style: PlotStyle,
    watermark_path: Optional[str] = None,
    watermark_zoom: float = 0.05,
    coverage_radius_km: Optional[float] = 70.0,
    plot_extent: Optional[Tuple[float, float, float, float]] = None # <<< ADDED ARGUMENT
) -> Optional[bytes]:
    """
    Generates a PPI plot...

    Args:
        # ... (existing args) ...
        plot_extent: Optional tuple (LonMin, LonMax, LatMin, LatMax) to override
                     the default plot extent. Coordinates are geographic (PlateCarree).
    Returns:
        Bytes representing the generated PNG image, or None if plotting fails.
    """
    fig = None
    cartopy_crs_instance = None # Define higher up for use in extent setting
    try:
        # --- Input Validation ---
        # ... (existing validation) ...

        # --- Extract Metadata ---
        # ... (existing metadata extraction) ...

        # --- Determine and Convert CRS ---
        try:
            # ... (existing CRS determination logic, resulting in cartopy_crs_instance) ...
            # Make sure cartopy_crs_instance is assigned here
            proj_crs = get_dataset_crs(ds) # Assume helper function exists
            if proj_crs is None: raise ValueError("CRS not found")
            # ... (conversion logic from proj_crs to cartopy_crs_instance) ...
            # Example placeholder if conversion was complex:
            if isinstance(proj_crs, pyproj.CRS):
                 cf = proj_crs.to_cf()
                 if cf.get("grid_mapping_name") == "azimuthal_equidistant":
                      # ... logic to create ccrs.AzimuthalEquidistant ...
                      cartopy_crs_instance = ccrs.AzimuthalEquidistant(...) # Assign here
                 else: raise ValueError("Unsupported projection")
            elif isinstance(proj_crs, ccrs.Projection):
                 cartopy_crs_instance = proj_crs
            else: raise TypeError("Unsupported CRS type")

        except Exception as crs_err:
            logger.error(f"Could not determine/convert Cartopy CRS: {crs_err}", exc_info=True)
            return None

        # --- Plot Setup ---
        with plt.ioff():
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

            # --- Add Map Background ---
            # ... (existing map background logic) ...

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


            # --- Plot Data ---
            logger.debug(f"Attempting to plot variable '{variable}'...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                # Check if cartopy_crs_instance was successfully created before using it
                if cartopy_crs_instance is None:
                     logger.error("Cannot plot data: Data transform CRS is unknown.")
                     raise ValueError("Data transform CRS could not be determined.") # Stop plotting

                quadmesh = ds[variable].plot(
                    ax=ax,
                    x="x", y="y", # Plot using projected coords
                    cmap=style.cmap,
                    norm=style.norm,
                    transform=cartopy_crs_instance, # Tell plot the CRS of the x,y coords
                    add_colorbar=True, # Ensure colorbar is added
                    cbar_kwargs=dict(pad=0.075, shrink=0.75, label=style.variable_dname),
                )

            # --- Add Map Features (Gridlines, Coverage Circle) ---
            # ... (existing logic, ensure CRS=ccrs.PlateCarree() for gridlines and circle geometry) ...

            # --- Title ---
            # ... (existing title logic) ...

            # --- Watermark ---
            # Check the passed watermark_path argument here
            if watermark_path and os.path.exists(watermark_path):
                # ... (existing watermark logic using the passed watermark_path) ...
                pass
            elif watermark_path:
                 logger.warning(f"Watermark path provided but not found: {watermark_path}")

            # --- Final Touches & Saving ---
            # ... (existing tight_layout, buffer saving) ...

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=150) # Adjust dpi if needed
            buffer.seek(0)
            image_bytes = buffer.getvalue()
            return image_bytes

    # ... (existing except/finally block, ensuring plt.close(fig)) ...
    except Exception as e:
        logger.error(f"Failed to create PPI plot for {variable}: {e}", exc_info=True)
        return None
    finally:
        if fig is not None: plt.close(fig)
        if cartopy_crs_instance: del cartopy_crs_instance # Help GC