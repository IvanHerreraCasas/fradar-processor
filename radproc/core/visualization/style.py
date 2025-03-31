# core/visualization/style.py

import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt # For getting standard cmaps
import cartopy.io.img_tiles as cimgt

# Import config access function
from core.config import get_setting

logger = logging.getLogger(__name__)

# --- Style Data Class ---

@dataclass
class PlotStyle:
    """Holds plotting style elements for a radar variable."""
    cmap: Union[mcolors.Colormap, str]
    norm: mcolors.Normalize
    map_tile: Optional[cimgt.GoogleWTS] # Base class for OSM, Stamen, etc.
    variable_dname: str # Display name added here for convenience

# --- Style Loading Logic ---

_style_cache: Dict[str, PlotStyle] = {} # Cache loaded styles

def get_plot_style(variable: str) -> Optional[PlotStyle]:
    """
    Loads or retrieves the PlotStyle for a given variable from configuration.

    Parses settings from plot_styles.yaml, constructs matplotlib/cartopy objects,
    and caches the result.

    Args:
        variable: The name of the radar variable (e.g., "RATE", "DBZH").

    Returns:
        A PlotStyle object containing cmap, norm, map_tile, and display name,
        or None if the variable or its configuration is invalid.
    """
    if variable in _style_cache:
        return _style_cache[variable]

    logger.debug(f"Loading plot style for variable: {variable}")

    # Get style definitions and defaults from config
    styles_config = get_setting('styles.styles', {})
    defaults_config = get_setting('styles.defaults', {})
    variable_config = styles_config.get(variable)
    variable_dname = get_setting(f'app.variables_to_process.{variable}', variable) # Get display name

    if not variable_config:
        logger.warning(f"No style configuration found for variable '{variable}' in plot_styles.yaml. Cannot generate style.")
        # Optionally, create and return a default grayscale style here
        # return _create_default_style(variable)
        return None

    # --- Parse Components ---
    try:
        cmap = _parse_cmap(variable_config.get('cmap'))
        # Norm parsing might need the number of colors from the cmap
        norm = _parse_norm(variable_config.get('norm'), cmap)
        # Map tile parsing needs cache dir from app config
        map_tile = _parse_map_tile(variable_config.get('map_tile'), defaults_config.get('map_tile'))

        if cmap is None or norm is None:
             logger.error(f"Failed to parse essential style components (cmap or norm) for variable '{variable}'.")
             return None

        style = PlotStyle(cmap=cmap, norm=norm, map_tile=map_tile, variable_dname=variable_dname)
        _style_cache[variable] = style
        logger.info(f"Successfully loaded style for variable: {variable}")
        return style

    except Exception as e:
        logger.error(f"Error creating PlotStyle for variable '{variable}': {e}", exc_info=True)
        return None


def _parse_cmap(cmap_config: Optional[Union[str, Dict[str, Any]]]) -> Optional[mcolors.Colormap]:
    """Parses cmap configuration from YAML into a Colormap object."""
    if cmap_config is None:
        logger.warning("Cmap configuration missing. Using default 'viridis'.")
        return plt.get_cmap('viridis') # Return a default

    if isinstance(cmap_config, str):
        # Simple case: cmap name
        try:
            return plt.get_cmap(cmap_config)
        except ValueError:
            logger.error(f"Colormap name '{cmap_config}' not found. Using default 'viridis'.")
            return plt.get_cmap('viridis')

    if isinstance(cmap_config, dict):
        cmap_type = cmap_config.get('type')
        if cmap_type == 'ListedColormap':
            colors = cmap_config.get('colors')
            name = cmap_config.get('name', 'custom_listed') # Optional name
            if colors and isinstance(colors, list):
                try:
                    return mcolors.ListedColormap(colors, name=name)
                except Exception as e:
                    logger.error(f"Failed to create ListedColormap: {e}")
            else:
                logger.error("Invalid 'colors' definition for ListedColormap.")
        # Add parsers for other cmap types if needed (e.g., LinearSegmentedColormap)
        else:
             logger.error(f"Unsupported cmap type '{cmap_type}' in configuration.")

    logger.error("Failed to parse cmap configuration. Using default 'viridis'.")
    return plt.get_cmap('viridis')


def _parse_norm(norm_config: Optional[Dict[str, Any]], cmap: Optional[mcolors.Colormap]) -> Optional[mcolors.Normalize]:
    """Parses norm configuration from YAML into a Normalize object."""
    if norm_config is None:
        logger.warning("Norm configuration missing. Using default Normalize.")
        return mcolors.Normalize() # Return a default linear norm

    if isinstance(norm_config, dict):
        norm_type = norm_config.get('type')
        vmin = norm_config.get('vmin')
        vmax = norm_config.get('vmax')
        clip = norm_config.get('clip', False) # Default clip=False for Normalize

        if norm_type == 'Normalize':
             # Standard linear normalization
             return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=clip)
        elif norm_type == 'BoundaryNorm':
            bounds = norm_config.get('bounds')
            if bounds and isinstance(bounds, list) and cmap:
                 # BoundaryNorm needs N colors for N+1 bounds.
                 # The number of colors should ideally match len(bounds) - 1
                 ncolors = cmap.N if isinstance(cmap, mcolors.ListedColormap) else 256 # Use cmap.N if available
                 extend = norm_config.get('extend', 'neither') # 'min', 'max', 'both', 'neither'

                 # Adjust ncolors based on bounds if cmap looks like standard gradient
                 # This part can be tricky. BoundaryNorm links intervals to colors.
                 # If len(bounds) = N+1, we ideally need N colors.
                 expected_ncolors = len(bounds) - 1
                 if isinstance(cmap, mcolors.ListedColormap) and cmap.N != expected_ncolors:
                      logger.warning(f"BoundaryNorm: Number of colors in ListedColormap ({cmap.N}) "
                                     f"does not match expected ({expected_ncolors}) based on bounds. Results may vary.")
                      # Decide policy: truncate cmap? error? proceed? Let's proceed.
                      ncolors = cmap.N # Trust the cmap provided
                 elif not isinstance(cmap, mcolors.ListedColormap):
                      ncolors = expected_ncolors # Use bounds to define segments in continuous cmap


                 return mcolors.BoundaryNorm(boundaries=bounds, ncolors=ncolors, clip=clip, extend=extend)

            else:
                logger.error("Invalid 'bounds' definition or missing cmap for BoundaryNorm.")
        # Add parsers for other norm types like LogNorm, PowerNorm etc. if needed
        # elif norm_type == 'LogNorm':
        #     return mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=clip)
        else:
             logger.error(f"Unsupported norm type '{norm_type}' in configuration.")

    logger.error("Failed to parse norm configuration. Using default Normalize.")
    return mcolors.Normalize()


def _parse_map_tile(tile_config: Optional[Dict[str, Any]], default_tile_config: Optional[Dict[str, Any]]) -> Optional[cimgt.GoogleWTS]:
    """Parses map tile configuration into a Cartopy Tile object."""
    config_to_use = tile_config if tile_config is not None else default_tile_config

    if config_to_use is None:
        logger.warning("No map tile configuration found (specific or default). No map background will be added.")
        return None

    if isinstance(config_to_use, dict):
        tile_type = config_to_use.get('type')
        params = config_to_use.get('params', {})

        # Get cache directory from app config
        cache_dir = get_setting('app.tile_cache_dir', './cache/OSM') # Use default if not set
        os.makedirs(cache_dir, exist_ok=True) # Ensure cache dir exists

        try:
            if tile_type == 'OSM':
                # Inject the cache directory into params
                params['cache'] = cache_dir
                # Handle desired_tile_form safely
                if 'desired_tile_form' not in params:
                     params['desired_tile_form'] = 'RGB' # Default to RGB if not specified
                return cimgt.OSM(**params)

            elif tile_type == 'Stamen':
                 params['cache'] = cache_dir
                 style = params.get('style', 'terrain-background') # Default style
                 return cimgt.Stamen(style=style, cache=cache_dir) # Stamen takes style differently

            elif tile_type == 'QuadtreeTiles': # Example for another type
                 params['cache'] = cache_dir
                 return cimgt.QuadtreeTiles(**params)

            # Add other tile types (e.g., custom URL via GoogleTiles) if needed
            # elif tile_type == 'GoogleTiles':
            #      params['cache'] = cache_dir
            #      return cimgt.GoogleTiles(**params) # Requires url usually

            else:
                logger.error(f"Unsupported map tile type '{tile_type}'. No map background will be added.")

        except ImportError:
             logger.error("Cartopy tile prerequisites (e.g., Pillow, requests) might be missing for the selected tile type.")
        except Exception as e:
             logger.error(f"Failed to create map tile object: {e}")

    else:
        logger.error("Invalid map tile configuration format. Expected a dictionary.")

    return None


# --- Helper to ensure necessary directories exist ---
import os
def _ensure_cache_dir():
     cache_dir = get_setting('app.tile_cache_dir', './cache/OSM')
     os.makedirs(cache_dir, exist_ok=True)

_ensure_cache_dir() # Ensure cache dir exists when module is imported