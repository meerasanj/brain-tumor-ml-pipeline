import os
from pathlib import Path

class Config:
    # Directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    MODEL_DIR = BASE_DIR / "models"
    
    # Model
    MODEL_NAME = "google/vit-base-patch16-224-in21k"
    IMAGE_SIZE = (128, 128)
    CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
    LEARNING_RATE = 3e-5
    BATCH_SIZE = 4
    EPOCHS = 3
    
    @classmethod
    def setup(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)

Config.setup()