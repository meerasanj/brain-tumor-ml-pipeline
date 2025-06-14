import sys
import logging
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
from src.config import Config
from src.data_handler import DataHandler
from src.model import MedicalImageClassifier
from src.metrics import Metrics
from src.utils import setup_logging, save_results

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 1. Data Preparation
        logger.info("Using local dataset from /data directory...")
        training_dir = PROJECT_ROOT / "data" / "Training"  # Changed to point to Training folder

        # 2. Verify Training folder structure
        tumor_types = Config.CLASSES
        valid_structure = all((training_dir / subtype).exists() for subtype in tumor_types)

        if not valid_structure:
            found = [d.name for d in training_dir.iterdir() if d.is_dir()]
            raise FileNotFoundError(
                f"Invalid Training folder structure.\n"
                f"Expected subfolders: {tumor_types}\n"
                f"Found in {training_dir}: {found}"
            )

        # 3. Initialize Model
        logger.info(f"Loading {Config.MODEL_NAME}...")
        classifier = MedicalImageClassifier()

        # 4. Sample Prediction
        sample_path = next((training_dir / "glioma").glob("*.jpg"), None)
        if not sample_path:
            raise FileNotFoundError(f"No sample image found in {training_dir/'glioma'}")

        logger.info(f"Processing sample: {sample_path.name}")
        image_tensor = DataHandler.preprocess_image(sample_path)

        # 5. Inference
        logger.info("Running inference...")
        results = classifier.predict(image_tensor)

        # 6. Evaluation
        logger.info("Evaluating...")
        metrics = Metrics.calculate(
            y_true=["glioma"],
            y_pred=[results["class"]]
        )

        # 7. Save Results
        save_path = save_results({
            "sample": sample_path.name,
            "true_class": "glioma",
            "predicted_class": results["class"],
            "confidence": results["confidence"],
            "metrics": metrics,
            "model": Config.MODEL_NAME
        })
        logger.info(f"Results saved to: {save_path}")
        logger.info(f"Predicted: {results['class']} (confidence: {results['confidence']:.2%})")
        logger.info(f"Accuracy: {metrics['accuracy']:.2f}")

    except Exception as e:
        logger.exception("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()