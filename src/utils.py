import json
import logging
from pathlib import Path
import numpy as np
from src.config import Config

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
    """Save evaluation results to JSON with numpy compatibility"""
    def numpy_encoder(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_path = Config.OUTPUT_DIR / filename
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=numpy_encoder)
        return output_path
    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")
        raise