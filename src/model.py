# Import necessary libraries from Hugging Face, PyTorch, and data science ecosystem
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
import pandas as pd

# MedicalImageClassifier class encapsulates ViT model handling
# Handles initialization, training, and evaluation
class MedicalImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading {Config.MODEL_NAME} on {self.device}")

        # Load the pre-trained ViT image processor and model from Hugging Face
        self.processor = ViTImageProcessor.from_pretrained(Config.MODEL_NAME)
        self.model = ViTForImageClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=len(Config.CLASSES),
            ignore_mismatched_sizes=True
        ).to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Define a transform to reverse normalization from dataloader
        # ViTProcessor expects PIL Images, not normalized tensors
        self.inverse_norm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    # Prepares batch of normalized tensors for the ViT processor by converting them back to PIL Images
    def _prepare_batch(self, batch_images):
        """Convert normalized batch back to PIL images"""
        unnormalized = self.inverse_norm(batch_images)
        return [transforms.ToPILImage()(img) for img in unnormalized]

    # Handles the main training loop for the specified number of epochs
    # Allows for progress tracking 
    def train(self, train_loader, val_loader, epochs=5):
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

                # Update loss and accuracy metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate epoch metrics
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            val_loss, val_acc, val_report, cm = self.evaluate(val_loader)
            
            # Log and store history
            history['train_loss'].append(float(train_loss))
            history['train_acc'].append(float(train_acc))
            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_acc))
            
            logging.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )
            logging.info(f"Confusion Matrix:\n{pd.DataFrame(cm, index=Config.CLASSES, columns=Config.CLASSES)}")
        
        return history, val_report, cm

    # Evaluates the model on a given data loader
    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        running_loss = 0.0

        # Disable gradient calculations
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
        
        # Generate classification report from scikit-learn
        report = classification_report(
            all_labels, all_preds,
            target_names=Config.CLASSES,
            output_dict=True,
            digits=4
        )
        
        # Generate and save confusion matrix
        cm = self._save_confusion_matrix(all_labels, all_preds)
        
        return float(val_loss), float(val_acc), report, cm

    # Helper function to generate and save confusion matrix plots
    def _save_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        
        # Save raw counts confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=Config.CLASSES,
            yticklabels=Config.CLASSES,
            cmap="Blues"
        )
        plt.title("Confusion Matrix (Counts)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / "confusion_matrix_counts.png")
        plt.close()
        
        # Save normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f",
            xticklabels=Config.CLASSES,
            yticklabels=Config.CLASSES,
            cmap="Blues"
        )
        plt.title("Confusion Matrix (Normalized)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / "confusion_matrix_normalized.png")
        plt.close()
        
        return cm

    # Makes a prediction on a single, preprocessed image tensor
    def predict(self, image_tensor):
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

    # Convenience method to evaluate the model on a folder of new images
    def test_on_new_images(self, image_folder):
        dataset = ImageFolder(
            str(image_folder),
            transform=DataHandler.get_test_transform()
        )
        loader = DataLoader(dataset, batch_size=8)
        
        logging.info(f"\nTesting on {len(dataset)} unseen images")
        return self.evaluate(loader)
