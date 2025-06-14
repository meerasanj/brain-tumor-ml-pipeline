import os
import logging
from pathlib import Path
import kagglehub
import subprocess
import torch
from torchvision import transforms
from PIL import Image
from src.config import Config

class DataHandler:
    @staticmethod
    def download_dataset():
        """Downloads and extracts the brain tumor dataset"""
        dataset_path = Config.DATA_DIR / "brain-tumor-mri-dataset"
        
        # Check if dataset already exists
        if (dataset_path / "Training").exists():
            return dataset_path / "Training"
            
        try:
            logging.info(f"Downloading dataset to {dataset_path}")
            
            # Using Kaggle CLI for reliable downloads
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "masoudnickparvar/brain-tumor-mri-dataset",
                "-p", str(Config.DATA_DIR),
                "--unzip",
                "--force"
            ], check=True)
            
            # Verify the extracted folder structure
            training_dir = dataset_path / "Training"
            if not training_dir.exists():
                raise FileNotFoundError(
                    f"Training directory not found at {training_dir}\n"
                    f"Found: {list(dataset_path.glob('*'))}"
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
        """Preprocesses an image for ViT model"""
        transform = transforms.Compose([
            transforms.Resize(256),          # Resize to slightly larger than target
            transforms.CenterCrop(224),      # ViT requires 224x224 input
            transforms.ToTensor(),
            transforms.Normalize(            # MRI-specific normalization
                mean=[0.485, 0.456, 0.406], # Standard ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        try:
            img = Image.open(image_path).convert('RGB')  # Ensure 3 channels
            return transform(img)
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            raise

    @staticmethod
    def get_class_distribution(data_dir: Path) -> dict:
        """Returns count of images per class"""
        return {
            cls: len(list((data_dir / cls).glob("*.jpg"))) 
            for cls in Config.CLASSES
        }