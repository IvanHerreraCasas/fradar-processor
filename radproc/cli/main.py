import argparse

from radproc.service.service import FRadarProcessorService

def main():
    CONFIG_FILE = "./config.ini"
    
    service = FRadarProcessorService()
        
    variables = [
        "RATE",
        "DBZH",
        "VRADH",
        "ZDR",
        "KDP",
        "PHIDP",
        "RHOHV",
        "WRADH",
        "QUAL",
    ]
    
    parser = argparse.ArgumentParser(description="Radar Furuno CLI")
    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand")
    
    
    start_parser = subparsers.add_parser("start")
    start_parser.set_defaults(service.start)
    
    stop_parser = subparsers.add_parser("stop")
    stop_parser.set_defaults(service.stop)
    
    restart_parser = subparsers.add_parser("restart")
    restart_parser.set_defaults(service.restart)
    
    enable_parser = subparsers.add_parser("enable")
    restart_parser.set_defaults(service.enable)
    
    disable_parser = subparsers.add_parser("disable")
    restart_parser.set_defaults(service.disable)
    
    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(service.status)
    
    
    
    