from src.data_handler import DataHandler
from src.model import MedGemmaClassifier
from src.metrics import Metrics
from src.utils import setup_logging
from pathlib import Path
import logging

def main():
    setup_logging()
    
    # 1. Data
    data_path = DataHandler.download_dataset()
    
    # 2. Sample prediction (full loop would iterate through dataset)
    sample_img = next((data_path / "glioma").glob("*.jpg"))
    image_tensor = DataHandler.preprocess_image(sample_img)
    
    # 3. Inference
    classifier = MedGemmaClassifier()
    results = classifier.predict(image_tensor)
    
    # 4. Evaluation (mock example)
    y_true = ["glioma"]
    y_pred = [results["probs"].argmax().item()]
    metrics = Metrics.calculate(y_true, y_pred)
    
    logging.info(f"Accuracy: {metrics['accuracy']:.2f}")

if __name__ == "__main__":
    main()