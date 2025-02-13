import subprocess

class FRadarProcessorService:
    def __init__(self, name: str, command: str, logger) -> None:
        """
        Initialize the scheduled task manager.
        
        :param name: Name of the scheduled task.
        :param command: Full command to execute (e.g., `"C:\\path\\to\python.exe C:\\path\\to\\main.py"`).
        """
        self.name = name
        self.command = command
        self.logger = logger

    def start(self, args) -> None:
        """Start the task immediately."""
        self.logger.info("Starting...")
        subprocess.run(["schtasks", "/Run", "/TN", self.name], check=True)

    def stop(self, args) -> None:
        """Stop the running task (if applicable)."""
        self.logger.info("Service stopped")
        subprocess.run(["schtasks", "/End", "/TN", self.name], check=True)

    def restart(self, args) -> None:
        """Restart the task (stop → start)."""
        self.stop(args)
        self.start(args)

    def status(self, args) -> str:
        """Get detailed task status."""
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", self.name, "/V"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout

    def enable(self, args) -> None:
        """Create the task to run at system startup."""
        subprocess.run(
            [
                "schtasks", "/Create", "/TN", self.name,
                "/SC", "ONSTART", "/TR", self.command,
                "/RU", "SYSTEM", "/RL", "HIGHEST"
            ],
            check=True
        )
        self.logger.info("Task created")

    def disable(self, args) -> None:
        """Delete the task (disable auto-start)."""
        subprocess.run(["schtasks", "/Delete", "/TN", self.name, "/F"], check=True)
        self.logger.info("Task deleted")