from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from torchvision import transforms
from src.config import Config
import logging

class MedicalImageClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading {Config.MODEL_NAME} on {self.device}")
        
        self.processor = ViTImageProcessor.from_pretrained(Config.MODEL_NAME)
        self.model = ViTForImageClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=len(Config.CLASSES),
            ignore_mismatched_sizes=True
        ).to(self.device)

    def predict(self, image: torch.Tensor) -> dict:
        """Predict tumor class from preprocessed image tensor"""
        try:
            # Ensure tensor is on correct device
            image = image.to(self.device)
            
            # Convert back to PIL Image for ViT processor
            inverse_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            unnormalized = inverse_normalize(image)
            pil_image = transforms.ToPILImage()(unnormalized.cpu())
            
            # Process through ViT processor
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax().item()
            
            return {
                "class": Config.CLASSES[pred_idx],
                "confidence": float(probs[0][pred_idx]),
                "all_probs": {c: float(p) for c, p in zip(Config.CLASSES, probs[0])}
            }
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise