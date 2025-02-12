import xarray as xr
import xradar as xd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import numpy as np
import shapely
import cv2
from cartopy.geodesic import Geodesic

from core.data import FRadarData
from plotting.style import PlotStyle, RatePlotStyle

from datetime import datetime, timedelta

import os.path

class FRadarPlotter():
    
    def __init__(self, variable, variable_dname, output_dir):
        self.data = FRadarData
        self.output_dir = output_dir
        
        match variable:
            case "RATE":
                self.style = RatePlotStyle()
            case "DBZH":
                pass
            case "VRADH":
                pass
            case "ZDR":
                pass
            case "KDP":
                pass
            case "PHIDP":
                pass
            case "RHOHV":
                pass
            case "WRADH":
                pass
            case "QUAL":
                pass
         
        self.variable = variable   
        self.variable_dname = variable_dname
        
    def plot(self, data: FRadarData, filename=None):
            
        ds = data.ds
        
        dt: datetime = data.get_datetime()
        datetime_file = dt.strftime("%Y%m%d_%H%M")
        datetime_utc = dt.strftime("%Y-%m-%d %H:%M")
        datetime_lt = (dt - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
        elevation: float = ds.elevation.values[0]        
        if not filename:
            filename = f"{self.variable}_{int(elevation*100)}_{datetime_file}.png"
        
        filepath = os.path.join(self.output_dir, filename) 
        
        # ==============================
        # Data Preparation
        # ==============================
        proj_crs = xd.georeference.get_crs(ds)
        cartopy_crs = ccrs.Projection(proj_crs)
        
        
        # ==============================
        # Plot Setup
        # ==============================
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        map_tile = self.style.map_tile
        cmap = self.style.cmap
        norm = self.style.norm

        # Add base map
        ax.add_image(
            map_tile,
            9,
            alpha=0.5,
            cmap="gray",
        )  
        
        # Plot data
        
        ds[self.variable].plot(
        x="x",
        y="y",
        cmap=cmap,
        norm=norm,
        transform=cartopy_crs,
        cbar_kwargs=dict(pad=0.075, shrink=0.75),
        ) 
        

        # Add gridlines
        grid_lines = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
        grid_lines.top_labels = False
        grid_lines.right_labels = False

        # Add radar coverage circle
        geodesic = Geodesic()
        circle_points = geodesic.circle(
            lon=ds[self.variable]["longitude"],
            lat=ds[self.variable]["latitude"],
            radius=70000,  # 70km radius
            n_samples=100,
            endpoint=False,
        )
        coverage_area = shapely.geometry.Polygon(circle_points)
        ax.add_geometries(
            [coverage_area],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=0.5
        )
        
        ax.set_title(f"{self.variable_dname} a EL: {elevation}° \n {datetime_utc} UTC ({datetime_lt} LT)")
        
        fig.savefig(filepath, dpi=300)
        
        plt.close(fig)
    
    def animate(self, img_paths, filename, fps=2, codec='mp4v'):
        """
        Creates a video from a list of image filepaths.
        
        Parameters:
            img_paths (list): List of paths to input images.
            output_path (str): Path to save the output video.'.
            fps (int): Frame rate of the output video. Default: 2.
            frame_size (tuple): Desired frame size (width, height). If None, uses the size of the first image.
            codec (str): FourCC codec code (e.g., 'mp4v' for MP4). Default: 'mp4v'.
        
        Raises:
            ValueError: If img_paths is empty or the first image cannot be read.
            RuntimeError: If the video writer cannot be initialized.
        """
        if not img_paths:
            raise ValueError("The list of image paths is empty.")
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Determine frame size from the first image if not provided
        if frame_size is None:
            first_image = cv2.imread(img_paths[0])
            if first_image is None:
                raise ValueError(f"Could not read the first image: {img_paths[0]}")
            frame_size = (first_image.shape[1], first_image.shape[0])
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
        
        if not video_writer.isOpened():
            raise RuntimeError("Could not open video writer.")
        
        # Process each image
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping.")
                continue
            
            
            video_writer.write(img)
        
        video_writer.release()