from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.config import Config
from src.data_handler import DataHandler
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
        """Train model with progress tracking"""
        self.model.train()
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
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
            
            # Calculate epoch metrics
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            val_loss, val_acc, val_report = self.evaluate(val_loader)
            
            # Store history
            history['train_loss'].append(float(train_loss))
            history['train_acc'].append(float(train_acc))
            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_acc))
            
            logging.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )
        
        return history, val_report

    def evaluate(self, loader):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_preds = []
        all_labels = []
        running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Evaluating"):
                pil_images = self._prepare_batch(images)
                inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
                running_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = running_loss / len(loader)
        val_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Generate classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=Config.CLASSES,
            output_dict=True,
            digits=4
        )
        
        # Save confusion matrix
        self._save_confusion_matrix(all_labels, all_preds)
        
        return float(val_loss), float(val_acc), report

    def _save_confusion_matrix(self, y_true, y_pred):
        """Save confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=Config.CLASSES,
            yticklabels=Config.CLASSES,
            cmap="Blues"
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / "confusion_matrix.png")
        plt.close()

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

    def test_on_new_images(self, image_folder):
        """Test on completely unseen images"""
        dataset = ImageFolder(
            str(image_folder),
            transform=DataHandler.get_test_transform()
        )
        loader = DataLoader(dataset, batch_size=8)
        
        logging.info(f"\nTesting on {len(dataset)} unseen images")
        return self.evaluate(loader)