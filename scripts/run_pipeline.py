import sys
import logging
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data_handler import DataHandler
from src.model import MedicalImageClassifier
from src.utils import setup_logging, save_results

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 1. Verify dataset
        DataHandler.verify_dataset_structure()

        # 2. Prepare data
        train_loader, test_loader = DataHandler.get_dataloaders(Config.BATCH_SIZE)

        # 3. Initialize and train model
        classifier = MedicalImageClassifier()
        classifier.train(train_loader, test_loader, epochs=Config.EPOCHS)

        # 4. Evaluate on test set
        logger.info("Running evaluation on test set...")
        test_metrics = classifier.evaluate(test_loader)

        # 5. Save results
        save_path = save_results({
            "sample": sample_path.name,
            "prediction": results,
            "model": Config.MODEL_NAME
        })

        logger.info(f"Results saved to: {save_path}")
        logger.info(f"Predicted: {results['class']} (confidence: {results['confidence']:.2%})")

    except Exception as e:
        logger.exception("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
