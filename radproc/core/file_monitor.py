import schedule
import time
import os
import datetime
import glob
import shutil
import re

from core.processor import FRadarProcessor

class FileMonitor:
    def __init__(self, input_dir, output_dir, images_dir, processor: FRadarProcessor, after_process):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.images_dir = images_dir
        self.after_process = after_process
            
        self.processor = processor
                
    def run(self, interval=60):
        while True:
            new_files = self.get_new_files()[:50]
            if new_files:
                self.process_files(new_files)
  
            time.sleep(interval)
             
    def _parse_dir_date(self, dir_name):
        try:
            return datetime.datetime.strptime(dir_name, "%Y%m%d").date()
        except ValueError:
            return None

    def _parse_file_datetime(self, filename):
        try:
            parts = filename.split('_')
            if len(parts) < 4:
                return None
            
            date_str = parts[1]
            time_str = parts[2]
            
            return datetime.datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        
        except (ValueError, IndexError):
            return None

    def get_new_files(self):
        if not os.path.exists(self.input_dir):
            return []

        return glob.glob(os.path.join(self.input_dir, '*.scnx.gz'))
  
    def get_files(self, start_dt, end_dt):
        
        files = []
        
        start_date = start_dt.date()
        end_date = end_dt.date()
        
        start_time = start_dt.time()
        end_time = end_dt.time()
        
        if not os.path.exists(self.output_dir):
            return []
        
        # List and sort directories by date
        dirs = []
        for dir_name in os.listdir(self.output_dir):
            dir_path = os.path.join(self.output_dir, dir_name)
            if os.path.isdir(dir_path):
                dir_date = self._parse_dir_date(dir_name)
                if dir_date:
                    dirs.append((dir_date, dir_path))

        # Process directories in chronological order
        for dir_date, dir_path in sorted(dirs, key=lambda x: x[0]):
            # Skip directories older than last known date
            if start_date > dir_date or end_date < dir_date:
                continue

            # Determine if we need full directory processing
            process_full = (start_date < dir_date) and (dir_date < end_date)
            

            # Process directory files
            for filename in os.listdir(dir_path):
                if not filename.endswith('.scnx.gz'):
                    continue

                file_dt = self._parse_file_datetime(filename)
                if not file_dt:
                    continue

                # Add file if: processing full directory or time is newer
                if process_full or ((file_dt.time() >= start_dt) and (file_dt.time() <= end_dt.time())):
                    files.append((os.path.join(dir_path, filename), file_dt))

        # Sort all files by datetime
        files.sort(key=lambda x: x[1])
        return files

    def move_file(self, file_path):
        """
        Move files from a list of paths to a specified output directory.

        Args:
            file_paths (list): List of strings representing paths to files/directories to move
            output_dir (str): Path to the destination directory

        Raises:
            OSError: If file operations fail (e.g., permission issues, missing files)
            shutil.Error: If file moving operations fail
        """
        
        try:
            filename = re.split(r'[\\/]', file_path)[-1]
            
            date = self._parse_file_datetime(filename).strftime("%Y%m%d")
            output_path = os.path.join(self.output_dir, date)
            output_filepath = os.path.join(output_path, filename)
                            
            os.makedirs(output_path, exist_ok=True)
            
            # Check if the output file exists and remove it
            if os.path.isfile(output_filepath):
                os.remove(output_filepath)
                
            # Move file/directory to the target directory
            shutil.move(file_path, output_path)
        except Exception as e:
            print(f"Error moving {file_path}: {str(e)}")
            raise

    def reprocess_files(self, start: str, end: str):
        start_dt = datetime.datetime.strptime(start, "%Y%m%d_%H%M")
        end_dt = datetime.datetime.strptime(end, "%Y%m%d_%H%M")
        
        file_list = self.get_files(start_dt, end_dt)
        
        for filepath, dt in file_list:
            print(f"Reprocessing {filepath}")
            self.processor.create_plots(filepath)
     
    def process_files(self, file_list):
        if not file_list:
            return

        # Add your custom processing logic here
        for file_path in file_list:
            print(f"Processing {file_path}")   
            self.processor.create_plots(file_path)
            self.move_file(file_path)
            