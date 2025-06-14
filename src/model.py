from transformers import ViTForImageClassification, ViTImageProcessor
import torch
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
            ignore_mismatched_sizes=True  # Allows for custom num_classes
        ).to(self.device)

    def predict(self, image: torch.Tensor) -> dict:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return {
            "class": Config.CLASSES[probs.argmax()],
            "confidence": probs.max().item(),
            "all_probs": {c: float(p) for c, p in zip(Config.CLASSES, probs[0])}
        }