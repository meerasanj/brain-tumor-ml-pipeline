from pathlib import Path
import os

class Config:
    # Absolute paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # Model settings
    MODEL_NAME = "google/medgemma-4b-it"
    IMAGE_SIZE = (224, 224)
    PROMPT = "Classify this brain MRI scan."

    @classmethod
    def setup(cls):
        """Create required directories"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        print(f"Configuration initialized. DATA_DIR: {cls.DATA_DIR}")

# Initialize when module loads
Config.setup()