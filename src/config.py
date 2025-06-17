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
    CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
    
    # Training
    BATCH_SIZE = 16
    IMAGE_SIZE = (224, 224)
    LEARNING_RATE = 3e-5
    EPOCHS = 3
    
    # Evaluation
    TEST_FOLDER = BASE_DIR / "new_images"  # Folder for unseen test images
    
    @classmethod
    def setup(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.TEST_FOLDER, exist_ok=True)

Config.setup()