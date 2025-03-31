"""
RepoRadar Backup and Restore Tools

Provides command-line utilities for backing up and restoring the vector database.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import shutil
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RepoRadar.Backup")

def list_backups(backup_dir: str = "./backups"):
    """
    List all available backups in the backup directory
    
    Args:
        backup_dir: Directory containing backups
    """
    backup_path = Path(backup_dir)
    
    if not backup_path.exists() or not backup_path.is_dir():
        logger.error(f"Backup directory {backup_dir} does not exist")
        return
    
    # Find backup files
    backup_files = list(backup_path.glob("github_repos_backup_*.json"))
    if not backup_files:
        logger.info(f"No backups found in {backup_dir}")
        return
    
    # Sort by timestamp
    backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"\n{'='*50}")
    print(f"RepoRadar Backups in {backup_dir}")
    print(f"{'='*50}")
    
    for i, backup_file in enumerate(backup_files):
        # Get backup info
        size_mb = backup_file.stat().st_size / (1024 * 1024)
        timestamp = datetime.fromtimestamp(backup_file.stat().st_mtime)
        
        # Try to get item count
        try:
            with open(backup_file, "r") as f:
                data = json.load(f)
                count = len(data.get("ids", []))
        except:
            count = "Unknown"
        
        print(f"{i+1}. {backup_file.name}")
        print(f"   - Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   - Size: {size_mb:.2f} MB")
        print(f"   - Items: {count}")
        print()

def create_backup(backup_dir: str = "./backups"):
    """
    Create a new backup of the vector database
    
    Args:
        backup_dir: Directory to store backups
    """
    from reporadar.config import ConfigLoader
    from reporadar.storage import VectorStorage
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    
    # Create a device config with minimal settings
    device_config = {
        "device": "cpu",
        "precision": "fp32",
        "model_size": "small"
    }
    
    # Get storage settings
    storage_config = config_loader.get_storage_config()
    data_dir = storage_config.get("data_dir", "./data/vector_db")
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize storage
    storage = VectorStorage(device_config, data_dir=data_dir)
    
    # Get backup timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create backup
    backup_file = backup_path / f"github_repos_backup_{timestamp}.json"
    logger.info(f"Creating backup to {backup_file}")
    
    try:
        # Use storage module to create backup
        storage.backup(backup_dir)
        logger.info(f"Backup completed successfully")
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return

def restore_backup(backup_file: str):
    """
    Restore from a backup file
    
    Args:
        backup_file: Path to the backup file
    """
    backup_path = Path(backup_file)
    
    if not backup_path.exists() or not backup_path.is_file():
        logger.error(f"Backup file {backup_file} does not exist")
        return
    
    from reporadar.config import ConfigLoader
    from reporadar.storage import VectorStorage
    
    # Load configuration
    config_loader = ConfigLoader()
    
    # Create a device config with minimal settings
    device_config = {
        "device": "cpu",
        "precision": "fp32",
        "model_size": "small"
    }
    
    # Get storage settings
    storage_config = config_loader.get_storage_config()
    data_dir = storage_config.get("data_dir", "./data/vector_db")
    
    # Confirm with user
    print(f"\nWARNING: Restoring will replace your existing database with backup data.")
    confirm = input("Do you want to continue? (y/n): ")
    
    if confirm.lower() != 'y':
        logger.info("Restore operation cancelled")
        return
    
    # Initialize storage
    storage = VectorStorage(device_config, data_dir=data_dir)
    
    # Restore from backup
    logger.info(f"Restoring from backup {backup_file}")
    
    try:
        # Use storage module to restore
        storage.restore(str(backup_path))
        logger.info(f"Restore completed successfully")
    except Exception as e:
        logger.error(f"Error restoring from backup: {e}")
        return

def main():
    """Main entry point for backup tools"""
    parser = argparse.ArgumentParser(description="RepoRadar Backup Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List backups command
    list_parser = subparsers.add_parser("list", help="List all backups")
    list_parser.add_argument("--dir", default="./backups", help="Backup directory")
    
    # Create backup command
    backup_parser = subparsers.add_parser("create", help="Create a new backup")
    backup_parser.add_argument("--dir", default="./backups", help="Backup directory")
    
    # Restore backup command
    restore_parser = subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument("file", help="Backup file to restore from")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_backups(args.dir)
    elif args.command == "create":
        create_backup(args.dir)
    elif args.command == "restore":
        restore_backup(args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 