from torchvision import transforms
from src.config import Config

# Method to create/return composition of image transformations for testing/validation data
# Ensures that input images are consistently sized and normalized for model's expected format
def get_test_transform():
    return transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(), # PIL Image --> PyTorch Tensor 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # standard ImageNet statistics
    ])
