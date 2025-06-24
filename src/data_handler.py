import torch
import os
import logging
from pathlib import Path
from PIL import Image
from torchvision import transforms, datasets
from src.config import Config

# DataHandler class handles all data-related operations for the image classification pipeline.
# Provides methods for dataset verification, loading, and preprocessing
class DataHandler:
    @staticmethod
    def verify_dataset_structure():
        data_dir = Config.DATA_DIR
        
        training_dir = data_dir / "Training"
        testing_dir = data_dir / "Testing"
        
        if not training_dir.exists():
            raise FileNotFoundError(f"Training directory not found at {training_dir}")
            
        for folder in [training_dir, testing_dir]:
            for cls in Config.CLASSES:
                if not (folder / cls).exists():
                    raise FileNotFoundError(f"Missing class folder: {folder/cls}")

        logging.info(f"Dataset structure verified at {data_dir}")

    # Method to create and return PyTorch datasets for training and testing
    @staticmethod
    def get_datasets():
        # Applies standard image preprocessing transforms:
        to_tensor = transforms.Compose([
            transforms.Resize(256), # Resize to 256x256 pixels
            transforms.CenterCrop(224), # Center crop to 224x224 pixels (standard input size for many CNNs)
            transforms.ToTensor() # Convert to tensor format
        ])

        # Normalize using ImageNet statistics for transfer learning compatibility
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # PyTorch ImageFolder datasets
        train_dataset = datasets.ImageFolder(
            root=Config.DATA_DIR / "Training",
            transform=transforms.Compose([
                to_tensor,
                normalize
            ])
        )
        
        test_dataset = datasets.ImageFolder(
            root=Config.DATA_DIR / "Testing",
            transform=transforms.Compose([
                to_tensor,
                normalize
            ])
        )
        
        return train_dataset, test_dataset 

    # Method to create PyTorch DataLoaders for efficient batch processing during training/testing
    @staticmethod
    def get_dataloaders(batch_size=32):
        train_set, test_set = DataHandler.get_datasets()
        
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, test_loader

    # Preprocess single image for inference
    @staticmethod
    def preprocess_image(image_path: Path) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        return transform(img)
