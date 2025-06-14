import torch
import os
import logging
from pathlib import Path
from src.config import Config

class DataHandler:
    @staticmethod
    def download_dataset():
        """Returns path to Training directory, verifying both Training and Testing folders"""
        data_dir = Config.DATA_DIR
        
        # Verify both Training and Testing exist
        training_dir = data_dir / "Training"
        testing_dir = data_dir / "Testing"
        
        if not training_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found at {training_dir}\n"
                "Required structure:\n"
                "data/\n"
                "├── Training/\n"
                "│   ├── glioma/\n"
                "│   ├── meningioma/\n"
                "│   ├── pituitary/\n"
                "│   └── notumor/\n"
                "└── Testing/\n"
                "    ├── glioma/\n"
                "    ├── ... (same classes as Training)"
            )

        # Verify all class folders exist in both directories
        for folder in [training_dir, testing_dir]:
            for cls in Config.CLASSES:
                if not (folder / cls).exists():
                    raise FileNotFoundError(
                        f"Missing class folder: {folder/cls}\n"
                        f"Each of {Config.CLASSES} must exist in both Training/ and Testing/"
                    )

        logging.info(f"Found valid dataset structure at {data_dir}")
        return training_dir

    @staticmethod
    def preprocess_image(image_path: Path) -> torch.Tensor:
        """Preprocesses an image for ViT model"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        try:
            img = Image.open(image_path).convert('RGB')
            return transform(img)
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            raise

    @staticmethod
    def get_testing_dir():
        """Optional: Helper to get testing directory path"""
        return Config.DATA_DIR / "Testing"
