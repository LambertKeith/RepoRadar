"""
Development Server for RepoRadar

This script runs the application in development mode, monitoring for file changes
and automatically restarting when changes are detected.
"""

import time
import os
import sys
import subprocess
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RepoRadar.DevServer")

# Directories to watch
WATCH_DIRS = ["reporadar", "."]
# Files to watch
WATCH_FILES = ["run.py", "config.yaml"]
# File patterns to ignore
IGNORE_PATTERNS = ["*.pyc", "*.log", "*.git*", "__pycache__", "*.md"]

class ChangeHandler(FileSystemEventHandler):
    """File system change handler for auto-reloading"""
    
    def __init__(self, restart_func):
        self.restart_func = restart_func
        self.last_modified = time.time()
        self.debounce_time = 2  # seconds
    
    def on_any_event(self, event):
        """Handle file system events"""
        # Skip directories and ignored patterns
        if event.is_directory:
            return
            
        # Check if the path contains any ignore patterns
        path = event.src_path
        if any(p in path for p in IGNORE_PATTERNS):
            return
            
        # Only watch Python files, config files and specified files
        if not (path.endswith('.py') or path.endswith('.yaml') or path.endswith('.yml') or
                any(f in path for f in WATCH_FILES)):
            return
            
        # Debounce to avoid multiple restarts
        current_time = time.time()
        if current_time - self.last_modified < self.debounce_time:
            return
            
        self.last_modified = current_time
        logger.info(f"Change detected in {path}")
        self.restart_func()

def start_reporadar():
    """Start the RepoRadar process"""
    cmd = [sys.executable, "run.py"]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    logger.info(f"Starting RepoRadar with command: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def main():
    """Main development server entry point"""
    logger.info("Starting RepoRadar development server")
    
    # Initial process start
    process = start_reporadar()
    
    # Define restart function
    def restart():
        nonlocal process
        logger.info("Restarting RepoRadar...")
        if process.poll() is None:  # Process is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate gracefully, killing...")
                process.kill()
        
        # Start a new process
        process = start_reporadar()
    
    # Set up file watching
    handler = ChangeHandler(restart)
    observer = Observer()
    
    # Add watchers for directories
    for dir_path in WATCH_DIRS:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            logger.info(f"Watching directory: {path}")
            observer.schedule(handler, path=path, recursive=True)
    
    # Start observer
    observer.start()
    
    try:
        logger.info("Development server running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
            # Check if the process has died unexpectedly
            if process.poll() is not None:
                logger.error(f"Process exited with code {process.returncode}, restarting...")
                process = start_reporadar()
                
    except KeyboardInterrupt:
        logger.info("Stopping development server...")
        if process.poll() is None:
            process.terminate()
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main() 