from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torch.nn as nn
from torchvision import transforms
from src.config import Config
import logging
from tqdm import tqdm

class MedicalImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading {Config.MODEL_NAME} on {self.device}")
        
        self.processor = ViTImageProcessor.from_pretrained(Config.MODEL_NAME)
        self.model = ViTForImageClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=len(Config.CLASSES),
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.inverse_norm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def _prepare_batch(self, batch_images):
        """Convert normalized batch back to PIL images"""
        unnormalized = self.inverse_norm(batch_images)
        return [transforms.ToPILImage()(img) for img in unnormalized]

    def train(self, train_loader, val_loader, epochs=5):
        """Train the model"""
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                pil_images = self._prepare_batch(images)
                inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            val_loss, val_acc = self.evaluate(val_loader)
            
            logging.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

    def evaluate(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                pil_images = self._prepare_batch(images)
                inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return running_loss / len(test_loader), 100 * correct / total

    def predict(self, image_tensor):
        """Make prediction on single image"""
        self.model.eval()
        with torch.no_grad():
            pil_image = transforms.ToPILImage()(self.inverse_norm(image_tensor))
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax().item()
            
            return {
                "class": Config.CLASSES[pred_idx],
                "confidence": float(probs[0][pred_idx]),
                "all_probs": {c: float(p) for c, p in zip(Config.CLASSES, probs[0])}
            }