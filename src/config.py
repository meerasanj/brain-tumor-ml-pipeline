from pathlib import Path
import os

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_NAME = "google/medgemma-4b-it"
    
    # MedGEMMA Specifics
    IMAGE_SIZE = (224, 224)
    PROMPT = "Classify this brain MRI scan as glioma, meningioma, pituitary, or no tumor."

    @classmethod
    def setup(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)

Config.setup()