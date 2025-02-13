import xarray as xr
import numpy as np
import numpy as np
    

class FRadarData:
    def __init__(self, ds):
        """
        Initialize the Furuno radar data object with an xarray dataset.

        Parameters:
        - ds: xarray dataset containing radar data
        """
        
        self.ds = ds
    
    @classmethod
    def from_filepaths(cls, filepaths, variables):
        
        
        elevation_var = 'sweep_fixed_angle'
        
        if elevation_var not in variables:
            variables.append(elevation_var)
        
        ds = xr.open_mfdataset(
            filepaths,
            engine="furuno",
            concat_dim="time",
            combine="nested",
            parallel=True,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            preprocess=lambda ds: FRadarData._preprocess(ds, variables),
        )
        
        return cls(ds)
    
    @staticmethod    
    def _preprocess(ds, variables):
        ds = ds[variables]
        azimuth_angles = np.arange(0, 360, 0.5)
    
        ds = ds.reindex(azimuth=azimuth_angles, method="nearest")
    
        time = ds.time.dt.floor("1min").values[0]
        ds = ds.drop_vars("time")
        ds = ds.expand_dims({"time": [time]})
        
        ds = ds.drop_vars("elevation")
        ds = ds.expand_dims({"elevation": [ds.sweep_fixed_angle.values[0]]})
        return ds
        
    def georeference(self):
        """
        Georeference the radar data.

        Returns:
        - Georeferenced radar data
        """
        ds = self.ds.xradar.georeference()

        return FRadarData(ds)
    
    def subset(self, variables=None, time_range=None, elevations=None):
        """
        Subset the radar data based on time and elevation ranges.

        Parameters:
        - time_range: Tuple containing start and end times
        - elevations: List of elevation angles

        Returns:
        - Subsetted radar data
        """
        
        ds = self.ds
        
        if variables:
            ds = ds[variables]

        if time_range:
            ds = ds.sel(time=slice(*time_range))
        
        if elevations:
            ds = ds.sel(elevation=elevations)

        return FRadarData(ds)
 
    def aggregate(self, method, dim="time"):
        """
        Aggregate radar data using a specified method.

        Parameters:
        - method: Aggregation method (e.g., 'sum', 'mean')

        Returns:
        - Aggregated radar data
        """
        
        match method:
            case "sum":
                ds = self.ds.sum(dim=dim)
            case "mean":
                ds = self.ds.mean(dim=dim)
            case _:
                pass
        
        return FRadarData(ds)     
       
    def get_datetime(self, format=None):
        date_time = self.ds.time.values[0].astype('datetime64[s]').item()
        
        if format:
            date_time = date_time.strftime(format)
            
        return date_time

    def close(self):
        self.ds.close()