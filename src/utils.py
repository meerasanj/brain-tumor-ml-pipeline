import json
import logging
from pathlib import Path
import numpy as np
from src.config import Config

# Method to configure logging to output to both a file and the console
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # Define handlers to direct logs to file or console
        handlers=[
            logging.FileHandler(Config.OUTPUT_DIR / 'pipeline.log'),
            logging.StreamHandler()
        ]
    )

# Method to save evaluation results to JSON with numpy compatibilit
def save_results(results: dict, filename: str = "results.json") -> Path:
    # Nnsted helper function to convert non-standard types for JSON serialization
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

    # Construct  full output path using configured output directory
    output_path = Config.OUTPUT_DIR / filename
    try:
        with open(output_path, 'w') as f: # Write mode 
            json.dump(results, f, indent=2, default=numpy_encoder)
        return output_path
    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")
        raise
