import logging
from datetime import datetime
from pathlib import Path
from config import Config

def setup_logging():
    """Configure logging for the project"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def save_results(results: dict, filename: str = "results.json"):
    """Save evaluation results to JSON"""
    import json
    output_path = Config.DATA_DIR / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return output_path