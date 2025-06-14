from src.data_handler import DataHandler
from src.model import MedGemmaClassifier
from src.metrics import Metrics
from src.utils import setup_logging, save_results
import logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Data Preparation
        logger.info("Downloading dataset...")
        data_dir = DataHandler.download_dataset()
        
        # 2. Initialize Model
        logger.info("Loading MedGEMMA...")
        classifier = MedGemmaClassifier()
        
        # 3. Sample Prediction (in practice, iterate through dataset)
        sample_path = next((data_dir / "glioma").glob("*.jpg"))
        image_tensor = DataHandler.preprocess_image(sample_path)
        
        # 4. Inference
        logger.info("Running inference...")
        prediction = classifier.predict(image_tensor)
        predicted_class = prediction["probs"].argmax().item()
        
        # 5. Evaluation
        logger.info("Evaluating...")
        metrics = Metrics.calculate(
            y_true=["glioma"],
            y_pred=[predicted_class]
        )
        
        # 6. Save Results
        save_results(metrics)
        logger.info(f"Predicted class: {predicted_class}")
        logger.info(f"Accuracy: {metrics['accuracy']:.2f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()