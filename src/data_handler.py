import kagglehub
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.config import Config
import logging
import subprocess
import sys

class DataHandler:
    @staticmethod
    def download_dataset():
        """Downloads dataset and returns path to Training directory"""
        dataset_path = Config.DATA_DIR
        
        # Check if Training folder already exists
        training_dir = dataset_path / "Training"
        if training_dir.exists():
            return training_dir
            
        try:
            logging.info(f"Downloading dataset to {dataset_path}")
            
            # Using Kaggle CLI
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "masoudnickparvar/brain-tumor-mri-dataset",
                "-p", str(dataset_path),
                "--unzip",
                "--force"  # Ensure fresh download
            ], check=True)
            
            # The dataset extracts directly into data/Training and data/Testing
            if not training_dir.exists():
                raise FileNotFoundError(
                    f"Training directory not found at {training_dir}\n"
                    f"Actual contents: {list(dataset_path.glob('*'))}"
                )
                
            return training_dir
            
        except subprocess.CalledProcessError as e:
            logging.error("Download failed. Please ensure:")
            logging.error("1. You've accepted dataset rules at:")
            logging.error("   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
            logging.error("2. Your Kaggle API credentials are valid")
            raise

    @staticmethod
    def preprocess_image(image_path: Path) -> torch.Tensor:
        """Preprocess an image for MedGEMMA"""
        transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path).convert('RGB')
        return transform(img)