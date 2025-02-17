#!/usr/bin/env python3

import argparse
import yaml
import matplotlib
import os

matplotlib.use("Agg")  # Set backend to Agg

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Add the project root to Python's module search path
root_dir = Path(__file__).parent.parent  # Path to the project root
sys.path.append(str(root_dir))


from service.service import FRadarProcessorService
from core.file_monitor import FileMonitor
from core.processor import FRadarProcessor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def _setup_logger(log_file: str):
        # Create a logger object
        logger = logging.getLogger('frad-proc')
        logger.setLevel(logging.DEBUG)
        
        # Create file handler that logs all messages (DEBUG and above)
        file_handler = RotatingFileHandler(
            log_file,              # Log file name
            maxBytes=1024*1024,     # 1MB per file
            backupCount=5           # Keep 5 backup files
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler that logs WARNING and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        # Create formatters and add them to the handlers
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

class FRadarProcCLI:
    def __init__(self, config_file, logger):

        self.config_file = config_file

        with open(config_file) as file:
            config = yaml.safe_load(file)

        self.config = config

        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        images_dir = config["images_dir"]

        variables = config["variables"]

        self.processor = FRadarProcessor(images_dir, variables, logger)
        self.monitor = FileMonitor(
            input_dir=input_dir,
            output_dir=output_dir,
            images_dir=images_dir,
            processor=self.processor,
            after_process=lambda x: self._save_last_processed(x),
            logger=logger,
        )

    def _save_last_processed(self, last_processed):

        with open(self.config_file, "w") as file:

            self.config["last_processed"] = last_processed

            yaml.dump(self.config, file, default_flow_style=False)
            
    

    def run(self, args):
        self.monitor.run()
        
    def reprocess(self, args):
        start_dt_str = args.start
        end_dt_str = args.end
        
        self.monitor.reprocess_files(start_dt_str, end_dt_str)


def main():
    
    # Relative path to the file from the current script's location
    relative_config_path = "../../config/config.yaml"
    relative_log_path = "../../log/frad-proc.log"
    relative_frad_proc_bat_path = "../../scripts/frad-proc.bat"

    # Get the absolute path of the current script
    script_path = os.path.abspath(__file__)

    # Extract the directory of the script
    script_dir = os.path.dirname(script_path)

    # Combine the script directory with the relative path and resolve to absolute
    CONFIG_FILE = os.path.abspath(os.path.join(script_dir, relative_config_path))
    LOG_FILE = os.path.abspath(os.path.join(script_dir, relative_log_path))
    FRAD_PROC_FILE = os.path.abspath(os.path.join(script_dir, relative_frad_proc_bat_path))

    logger = _setup_logger(LOG_FILE)

    cli = FRadarProcCLI(CONFIG_FILE, logger)
    service = FRadarProcessorService("frad-proc", FRAD_PROC_FILE + " run", logger)

    parser = argparse.ArgumentParser(description="Radar Furuno CLI")
    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand")

    start_parser = subparsers.add_parser("start")
    start_parser.set_defaults(func=service.start)

    stop_parser = subparsers.add_parser("stop")
    stop_parser.set_defaults(func=service.stop)

    restart_parser = subparsers.add_parser("restart")
    restart_parser.set_defaults(func=service.restart)

    enable_parser = subparsers.add_parser("enable")
    enable_parser.set_defaults(func=service.enable)

    disable_parser = subparsers.add_parser("disable")
    disable_parser.set_defaults(func=service.disable)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=service.status)

    run_parser = subparsers.add_parser("run")
    run_parser.set_defaults(func=cli.run)
    
    reprocess_parser = subparsers.add_parser("reprocess")
    reprocess_parser.add_argument("start", help="Start datetime to reprocess (YYYYMMDD_HHMM)")
    reprocess_parser.add_argument("end", help="End datetime to reprocess (YYYYMMDD_HHMM)")
    reprocess_parser.set_defaults(func=cli.reprocess)

    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
