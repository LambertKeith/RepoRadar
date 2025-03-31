# 🔭 RepoRadar - GitHub Trends Intelligent Analysis System

RepoRadar is a hardware-adaptive system for analyzing GitHub trending repositories, designed to run with minimal resources on personal computers.

## Features

- 🔄 **Automatic GitHub Trends Collection** - Fetches and caches trending repositories
- 🔍 **Semantic Search** - Find projects similar to your query using advanced vector embeddings
- 🤖 **Hardware Adaptation** - Automatically uses your GPU if available, gracefully falls back to CPU
- 💾 **Efficient Storage** - Switches between in-memory and disk storage based on dataset size
- 🏷️ **Personal Tagging** - Add your own tags to repositories for better organization
- 🌐 **Easy Web Interface** - Simple Gradio-based UI for searching and managing repositories

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Pip or Conda for package management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RepoRadar.git
   cd RepoRadar
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python run.py
   ```
   
The system will automatically:
- Detect your hardware (CPU/GPU)
- Choose the appropriate model size
- Start collecting GitHub trending data
- Launch a web interface at http://127.0.0.1:7860

## Hardware Adaptation

RepoRadar automatically detects and utilizes your hardware:

| Hardware | Model Used | Memory Usage | Startup Time |
|----------|------------|--------------|--------------|
| CPU Only | MiniLM-L6 (ONNX) | ~500MB RAM | 5-10s |
| GPU (2-4GB VRAM) | MiniLM-L12 | ~1.2GB VRAM | 3-5s |
| GPU (>4GB VRAM) | MPNet-base | ~2.5GB VRAM | 4-6s |

## Command Line Options

```bash
python run.py --device auto|cpu|gpu
```

## Configuration

You can customize RepoRadar by editing the `config.yaml` file:

### Data Collection
```yaml
collector:
  update_interval: 6  # Hours between updates
  languages:
    - ""              # All languages
    - "python"
    - "javascript"
  time_periods:
    - "daily"
    - "weekly"
  max_repos_per_language: 25  # Max repos to collect per language
```

### Model Configuration
```yaml
vectorizer:
  models:
    small:  # Used on CPU or low-end GPU
      name: "all-MiniLM-L6-v2"
      dim: 384
      onnx_path: "./models/all-MiniLM-L6-v2-onnx/"
    medium:  # Used on mid-range GPU (2-4GB VRAM)
      name: "paraphrase-multilingual-MiniLM-L12-v2"
      dim: 384
    large:  # Used on high-end GPU (>4GB VRAM)
      name: "all-mpnet-base-v2"
      dim: 768
```

### Storage Settings
```yaml
storage:
  data_dir: "./data/vector_db"
  memory_threshold: 10000  # Items above which to use persistent storage
  backup_dir: "./backups"
```

### Web Interface
```yaml
interface:
  host: "127.0.0.1"  # Use 0.0.0.0 to allow external access
  port: 7860
  open_browser: true
  theme: "soft"  # Gradio theme
```

## Custom Models

If you have fine-tuned models for specific domains:

1. Place your model in the `./models/` directory
2. Update the `config.yaml` to reference your model:

```yaml
vectorizer:
  models:
    small:
      name: "your-custom-model-name"
      dim: 384  # Vector dimension of your model
      onnx_path: "./models/your-custom-model-onnx/"
```

For models hosted on HuggingFace Hub, you can directly use the HuggingFace model ID:

```yaml
vectorizer:
  models:
    small:
      name: "your-username/your-model-name"
      dim: 384
```

## Data Management

- Data is stored in `./data/` directory
- Backups are automatically created in `./backups/`
- Run periodic backups with:
  ```bash
  python run.py --backup
  ```

## Troubleshooting

### Common Issues

- **"CUDA not available"**: Install PyTorch with CUDA support
- **High memory usage**: Reduce `memory_threshold` in config.yaml
- **Slow startup**: Switch to a smaller model in config.yaml

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- HuggingFace for Sentence Transformers
- ChromaDB for vector storage
- Gradio for the web interface
