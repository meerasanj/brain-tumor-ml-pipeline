import os
from pathlib import Path

class Config:
    # Project root directory
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Model Configuration
    MODEL_NAME = "google/vit-base-patch16-224-in21k" 
    IMAGE_SIZE = (224, 224)  # ViT requires 224x224
    HUGGINGFACE_TIMEOUT = 30
    
    # Dataset Configuration 
    CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    @classmethod
    def setup(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

# Initialize configuration
Config.setup()