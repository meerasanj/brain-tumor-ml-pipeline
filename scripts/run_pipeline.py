import sys
from pathlib import Path
import logging

# Critical path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
from src.config import Config
from src.data_handler import DataHandler
from src.model import MedGemmaClassifier
from src.metrics import Metrics
from src.utils import setup_logging, save_results

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Data Preparation
        logger.info("Downloading dataset...")
        training_dir = DataHandler.download_dataset()
        
        # 2. Verify Training folder structure
        tumor_types = ["glioma", "meningioma", "pituitary", "notumor"]
        valid_structure = all((training_dir / subtype).exists() for subtype in tumor_types)
        
        if not valid_structure:
            found = [d.name for d in training_dir.iterdir() if d.is_dir()]
            raise FileNotFoundError(
                f"Invalid Training folder structure.\n"
                f"Expected: {tumor_types}\n"
                f"Found: {found}"
            )
        
        # 3. Initialize Model
        logger.info("Loading MedGEMMA...")
        classifier = MedGemmaClassifier()
        
        # 4. Sample Prediction
        sample_path = next((training_dir / "glioma").glob("*.jpg"), None)
        if not sample_path:
            raise FileNotFoundError("No sample image found in glioma directory")
            
        logger.info(f"Processing sample: {sample_path.name}")
        image_tensor = DataHandler.preprocess_image(sample_path)
        
        # 5. Inference
        logger.info("Running inference...")
        prediction = classifier.predict(image_tensor)
        predicted_class = prediction["probs"].argmax().item()
        
        # 6. Evaluation
        logger.info("Evaluating...")
        metrics = Metrics.calculate(
            y_true=["glioma"],
            y_pred=[predicted_class]
        )
        
        # 7. Save Results
        save_path = save_results({
            "sample": sample_path.name,
            "true_class": "glioma",
            "predicted_class": predicted_class,
            "metrics": metrics,
            "dataset_path": str(training_dir)
        })
        logger.info(f"Results saved to: {save_path}")
        logger.info(f"Predicted class index: {predicted_class}")
        logger.info(f"Accuracy: {metrics['accuracy']:.2f}")
        
    except Exception as e:
        logger.exception("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()