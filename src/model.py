from transformers import AutoProcessor, AutoModelForImageClassification
import torch
from config import Config
import logging

class MedGemmaClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading {Config.MODEL_NAME} on {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModelForImageClassification.from_pretrained(
            Config.MODEL_NAME
        ).to(self.device)

    def predict(self, image: torch.Tensor) -> dict:
        inputs = self.processor(
            images=image,
            text=Config.PROMPT,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return {
            "logits": outputs.logits,
            "probs": torch.nn.functional.softmax(outputs.logits, dim=-1)
        }