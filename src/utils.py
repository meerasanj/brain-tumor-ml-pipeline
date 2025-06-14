import logging
from pathlib import Path
import json
from src.config import Config  # Changed from 'config' to 'src.config'

def setup_logging():
    """Configure logging for the project"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.OUTPUT_DIR / 'pipeline.log'),
            logging.StreamHandler()
        ]
    )

def save_results(results: dict, filename: str = "results.json") -> Path:
    """Save evaluation results to JSON"""
    output_path = Config.OUTPUT_DIR / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return output_path
