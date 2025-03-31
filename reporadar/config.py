"""
Configuration Module

This module handles loading and validating configuration from config.yaml.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("RepoRadar.config")

DEFAULT_CONFIG_PATH = "config.yaml"

class ConfigLoader:
    """
    Loads and validates configuration from YAML file.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader with the path to the config file.
        
        Args:
            config_path: Path to the config file, or None to use default
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            
            if not config_file.exists():
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                self._set_defaults()
                return
                
            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f)
                
            logger.info(f"Loaded configuration from {self.config_path}")
            
            # Validate the loaded config
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values"""
        self.config = {
            "collector": {
                "update_interval": 6,
                "languages": ["", "python", "javascript", "typescript", "go", "rust"],
                "time_periods": ["daily", "weekly", "monthly"],
                "max_repos_per_language": 25
            },
            "vectorizer": {
                "models": {
                    "small": {
                        "name": "all-MiniLM-L6-v2",
                        "dim": 384,
                        "onnx_path": "./models/all-MiniLM-L6-v2-onnx/"
                    },
                    "medium": {
                        "name": "paraphrase-multilingual-MiniLM-L12-v2",
                        "dim": 384,
                        "onnx_path": None
                    },
                    "large": {
                        "name": "all-mpnet-base-v2",
                        "dim": 768,
                        "onnx_path": None
                    }
                }
            },
            "storage": {
                "data_dir": "./data/vector_db",
                "memory_threshold": 10000,
                "backup_dir": "./backups"
            },
            "interface": {
                "host": "127.0.0.1",
                "port": 7860,
                "open_browser": True,
                "theme": "soft"
            }
        }
    
    def _validate_config(self):
        """Validate configuration values and ensure required fields exist"""
        # Ensure top-level sections exist
        required_sections = ["collector", "vectorizer", "storage", "interface"]
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing config section {section}, using defaults")
                
                # Add default section if missing
                if section == "collector":
                    self.config["collector"] = self._set_defaults()["collector"]
                elif section == "vectorizer":
                    self.config["vectorizer"] = self._set_defaults()["vectorizer"]
                elif section == "storage":
                    self.config["storage"] = self._set_defaults()["storage"]
                elif section == "interface":
                    self.config["interface"] = self._set_defaults()["interface"]
        
        # Validate collector section
        collector = self.config.get("collector", {})
        if not collector.get("languages"):
            collector["languages"] = ["", "python", "javascript"]
            logger.warning("No languages specified in config, using defaults")
        
        if not collector.get("time_periods"):
            collector["time_periods"] = ["daily", "weekly"]
            logger.warning("No time periods specified in config, using defaults")
        
        # Validate vectorizer section
        vectorizer = self.config.get("vectorizer", {})
        if not vectorizer.get("models"):
            vectorizer["models"] = self._set_defaults()["vectorizer"]["models"]
            logger.warning("No models specified in config, using defaults")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration.
        
        Returns:
            Dictionary with configuration values
        """
        return self.config
    
    def get_collector_config(self) -> Dict[str, Any]:
        """
        Get collector-specific configuration.
        
        Returns:
            Dictionary with collector configuration
        """
        return self.config.get("collector", {})
    
    def get_vectorizer_config(self) -> Dict[str, Any]:
        """
        Get vectorizer-specific configuration.
        
        Returns:
            Dictionary with vectorizer configuration
        """
        return self.config.get("vectorizer", {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """
        Get storage-specific configuration.
        
        Returns:
            Dictionary with storage configuration
        """
        return self.config.get("storage", {})
    
    def get_interface_config(self) -> Dict[str, Any]:
        """
        Get interface-specific configuration.
        
        Returns:
            Dictionary with interface configuration
        """
        return self.config.get("interface", {}) 