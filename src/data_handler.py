import kagglehub
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from config import Config
import logging

class DataHandler:
    @staticmethod
    def download_dataset():
        """Downloads dataset only if not already present"""
        dataset_path = Config.DATA_DIR / "brain-tumor-mri-dataset"
        if not dataset_path.exists():
            try:
                kagglehub.dataset_download(
                    "masoudnickparvar/brain-tumor-mri-dataset",
                    path=str(Config.DATA_DIR)
                )
                logging.info(f"Dataset downloaded to {dataset_path}")
            except Exception as e:
                logging.error(f"Download failed: {e}")
                raise
        return dataset_path

    @staticmethod
    def preprocess_image(image_path: Path) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path).convert('RGB')
        return transform(img)