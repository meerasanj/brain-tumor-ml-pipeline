class Config:
    # Model Configuration
    MODEL_NAME = "google/vit-base-patch16-224-in21k" 
    IMAGE_SIZE = (224, 224)  # ViT requires 224x224
    
    # Dataset Configuration 
    CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"