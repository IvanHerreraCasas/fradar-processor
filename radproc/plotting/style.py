from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.io.img_tiles as cimgt

class PlotStyle:
    def __init__(self, cmap, norm, map_tile):
        self.cmap = cmap
        self.norm = norm
        self.map_tile = map_tile
        

class RatePlotStyle(PlotStyle):
    def __init__(self):
        PRECIP_BOUNDS = [0, 0.1, 1, 5, 10, 20, 50, 100, 200]  # Precipitation thresholds
        PRECIP_COLORS = [
            (0, 0, 0, 0),        # Transparent (0-0.1)
            (0.53, 0.81, 0.98, 1),  # Skyblue (0.1-1)
            (0, 0, 1, 1),        # Blue (1-5)
            (0, 1, 0, 1),        # Green (5-10) 
            (1, 1, 0, 1),        # Yellow (10-20)
            (1, 0.65, 0, 1),     # Orange (20-50)
            (1, 0, 0, 1),        # Red (50-100)
            (0.5, 0, 0.5, 1),    # Purple (>100)
        ]

        # Create colormap and normalization
        precip_cmap = ListedColormap(PRECIP_COLORS)
        precip_norm = BoundaryNorm(PRECIP_BOUNDS, precip_cmap.N)
        
        map_tile = cimgt.OSM(cache='./cache')
        super().__init__(precip_cmap, precip_norm, map_tile)
        
