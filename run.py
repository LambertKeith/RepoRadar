import argparse
import os
import torch
import logging
from pathlib import Path

from reporadar.collector import GithubTrendsCollector
from reporadar.config import ConfigLoader
from reporadar.interface import launch_interface
from reporadar.storage import VectorStorage
from reporadar.vectorizer import TextVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reporadar.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RepoRadar")

# Ensure required directories exist
Path("./data").mkdir(exist_ok=True)
Path("./models").mkdir(exist_ok=True)
Path("./backups").mkdir(exist_ok=True)

def detect_hardware():
    """Detect available hardware and return optimal configuration"""
    device_info = {
        "device": "cpu",
        "precision": "fp32",
        "model_size": "small"
    }
    
    if torch.cuda.is_available():
        device_info["device"] = "cuda"
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        logger.info(f"GPU detected with {gpu_mem:.2f}GB memory")
        
        if gpu_mem >= 4:
            device_info["precision"] = "fp16"
            device_info["model_size"] = "medium"
        
        if gpu_mem >= 8:
            device_info["model_size"] = "large"
    else:
        logger.info("No GPU detected, using CPU mode")
    
    return device_info

def get_device_config(args):
    """Detect hardware or use specified device and return device configuration."""
    if args.device == "auto":
        return detect_hardware()
    else:
        return {
            "device": "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu",
            "precision": "fp16" if args.device == "gpu" and torch.cuda.is_available() else "fp32",
            "model_size": "medium" if args.device == "gpu" else "small"
        }

def main():
    parser = argparse.ArgumentParser(description="RepoRadar - GitHub Trends Analysis System")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                        help="Computing device to use (default: auto-detect)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--backup", action="store_true",
                        help="Perform database backup")
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()
    
    # 获取设备配置
    device_config = get_device_config(args)
    
    logger.info(f"Using device: {device_config['device']}, "
                f"precision: {device_config['precision']}, "
                f"model size: {device_config['model_size']}")
    
    # Initialize storage first to check if we need to collect data
    storage_config = config_loader.get_storage_config()
    storage = VectorStorage(
        device_config,
        data_dir=storage_config.get("data_dir", "./data/vector_db"),
        memory_threshold=storage_config.get("memory_threshold", 10000)
    )
    
    # Handle backup request if specified
    if args.backup:
        backup_dir = storage_config.get("backup_dir", "./backups")
        logger.info(f"Performing database backup to {backup_dir}")
        storage.backup(backup_dir)
        return
    
    # Initialize modules with configuration
    collector_config = config_loader.get_collector_config()
    collector = GithubTrendsCollector(
        cache_dir="./data/cache",
        update_interval=collector_config.get("update_interval", 6),
        languages=collector_config.get("languages", ["", "python", "javascript"]),
        time_periods=collector_config.get("time_periods", ["daily", "weekly"])
    )
    
    # Initialize vectorizer with model configuration from config
    vectorizer_config = config_loader.get_vectorizer_config()
    if "models" in vectorizer_config:
        device_config["model_configs"] = vectorizer_config["models"]
    
    vectorizer = TextVectorizer(device_config)
    
    # Run data collection if needed
    if collector.should_run():
        logger.info("Collecting GitHub trends data...")
        repos_data = collector.collect_trends()
        vectors = vectorizer.vectorize_batch(repos_data)
        storage.store_vectors(vectors, repos_data)
    else:
        logger.info("当前不在允许的爬取时间内，爬虫功能被禁用。")
    
    # 启动 Web 界面
    interface_config = config_loader.get_interface_config()
    launch_interface(
        storage, 
        vectorizer,
        host=interface_config.get("host", "127.0.0.1"),
        port=interface_config.get("port", 7860),
        open_browser=interface_config.get("open_browser", True)
    )

if __name__ == "__main__":
    main() 