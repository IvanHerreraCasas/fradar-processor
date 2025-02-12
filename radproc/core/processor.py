from core.data import FRadarData
from plotting.plotter import FRadarPlotter
                
import os
import re
from datetime import datetime, date
from typing import List, Optional

class FRadarProcessor:
    
    def __init__(self, images_dir: str, variables: List[str]):
        self.variables = variables
        
        self.images_dir = images_dir
        
        os.makedirs(self.images_dir, exist_ok=True)
        
    def create_plots(self, filepath: str) -> None:
        """Generate plots for each variable from radar data file."""
        variable_keys = list(self.variables.keys())
        
        data = FRadarData.from_filepaths([filepath], variable_keys)
        date_str = data.get_datetime(format="%Y%m%d")
        elevation = int(data.ds['elevation'].values[0] * 100)
        
        for variable, variable_dname in self.variables.items():
            output_dir = os.path.join(self.images_dir, variable, date_str, str(elevation))
            os.makedirs(output_dir, exist_ok=True)
            
            plotter = FRadarPlotter(variable, variable_dname, output_dir)
            var_data = data.subset(variables=[variable])
            var_data = var_data.georeference()
            plotter.plot(var_data)
            
    def _parse_dir_date(self, dir_name: str) -> Optional[date]:
        """Parse directory name (YYYYMMDD) into a date object."""
        try:
            return datetime.strptime(dir_name, "%Y%m%d").date()
        except ValueError:
            return None

    def _parse_file_datetime(self, dir_date: date, filename: str) -> Optional[datetime]:
        """Extract datetime from filename, ensuring it matches directory date."""
        match = re.search(r'_(\d{8})_(\d{6})\.png$', filename)
        if not match:
            return None
        
        file_date_str, time_str = match.groups()
        if file_date_str != dir_date.strftime("%Y%m%d"):
            return None
        
        try:
            time_obj = datetime.strptime(time_str, "%H%M%S").time()
            return datetime.combine(dir_date, time_obj)
        except ValueError:
            return None
    
    #def create_animations(self, start_dt_str, end_dt_str, folder, filename=None) -> None:
    #    """Produce animations for each variable/elevation within the given timeframe."""
    #    
    #    start_dt = datetime.strptime(start_dt_str, "%Y%m%d%H%M")
    #    end_dt = datetime.strptime(end_dt_str, "%Y%m%d%H%M")
    #    
    #    start_date = start_dt.date()
    #    end_date = end_dt.date(), elevation
    #    
    #    for variable, variable_dname in self.variables.items():
    #        images_dir = os.path.join(self.images_dir, variable)
    #        
    #        if not os.path.exists(images_dir):
    #            continue
    #            
    #        dirs = {}
    #        for date_dir_name in os.listdir(images_dir):
    #            date_dir_path = os.path.join(images_dir, date_dir_name)
    #            if not os.path.isdir(date_dir_path):
    #                continue
    #            
    #            dir_date = self._parse_dir_date(date_dir_name)
    #            if not dir_date:
    #                continue
    #            
    #            for el_dir_name in os.listdir(date_dir_path):
    #                el_dir_path = os.path.join(date_dir_path, el_dir_name)
    #                if os.path.isdir(el_dir_path):
    #                    if el_dir_name not in dirs:
    #                        dirs[el_dir_name] = []
    #                    dirs[el_dir_name].append((dir_date, el_dir_path))
    #        
    #        for elevation, dates_dirs in dirs.items():
    #            
    #            output_dir = os.path.join(self.animations_dir, folder, variable, str(elevation))
    #            plotter = FRadarPlotter(variable, variable_dname, output_dir)
    #            
    #            images_files = []
    #            for dir_date, el_dir_path in sorted(dates_dirs, key=lambda x: x[0]):
    #                if not (start_date <= dir_date <= end_date):
    #                    continue
    #                
    #                for filename in os.listdir(el_dir_path):
    #                    if filename.endswith('.png'):
    #                        file_dt = self._parse_file_datetime(dir_date, filename)
    #                        if file_dt and start_dt <= file_dt <= end_dt:
    #                            images_files.append((os.path.join(el_dir_path, filename), file_dt))
    #            
    #            if images_files:
    #                images_files.sort(key=lambda x: x[1])
    #                
    #                if not filename:
    #                    filename = f"{start_dt_str}_{end_dt_str}_{variable}_{elevation}.mp4"
    #                    
    #                plotter.animate([fp for fp, _ in images_files], filename)        