import subprocess

class FRadarProcessorService:
    def __init__(self, name: str = "fradar-processor") -> None:
        """
        Initialize the service with a given name.
        
        :param name: Name of the Windows service to manage
        """
        self.name = name

    def start(self, args) -> None:
        """Start the Windows service"""
        subprocess.run(["sc", "start", self.name], check=True)

    def stop(self, args) -> None:
        """Stop the Windows service"""
        subprocess.run(["sc", "stop", self.name], check=True)

    def restart(self, args) -> None:
        """Restart the Windows service"""
        self.stop()
        self.start()

    def status(self, args) -> str:
        """Get service status"""
        result = subprocess.run(
            ["sc", "query", self.name], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout

    def enable(self, args) -> None:
        """Set service to start automatically at boot"""
        subprocess.run(["sc", "config", self.name, "start=", "auto"], check=True)

    def disable(self, args) -> None:
        """Disable automatic service startup"""
        subprocess.run(["sc", "config", self.name, "start=", "disabled"], check=True)

    def delete(self, args) -> None:
        """Delete/Uninstall the service"""
        subprocess.run(["sc", "delete", self.name], check=True)

    # Bonus: Add error handling wrapper
    def _safe_run(self, command: list) -> None:
        """Wrapper for subprocess.run with error handling"""
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr}")