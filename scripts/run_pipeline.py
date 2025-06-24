import sys
import logging
from pathlib import Path
import numpy as np

# project root to Python path
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
        # 1. Verify dataset structure
        DataHandler.verify_dataset_structure()
        
        # 2. Prepare data loaders
        train_loader, test_loader = DataHandler.get_dataloaders(Config.BATCH_SIZE)
        
        # 3. Initialize model
        classifier = MedicalImageClassifier()
        
        # 4. Minimal initial evaluation
        initial_loss, initial_acc, _, initial_cm = classifier.evaluate(test_loader)
        logger.info(f"\nInitial Accuracy (before training): {initial_acc:.2f}%")
        logger.info("Initial Confusion Matrix:")
        logger.info(np.array2string(initial_cm, 
                                 formatter={'int': lambda x: f"{x:4d}"},
                                 prefix="    "))
        
        # 5. Train model
        train_history, val_report, confusion_matrix = classifier.train(
            train_loader, 
            test_loader, 
            epochs=Config.EPOCHS
        )
        
        # 6. Sample prediction
        sample_path = next((Config.DATA_DIR / "Training" / "glioma").glob("*.jpg"))
        image_tensor = DataHandler.preprocess_image(sample_path)
        prediction = classifier.predict(image_tensor)
        
        # 7. Prepare results for saving
        results = {
            "model": Config.MODEL_NAME,
            "config": {
                "batch_size": Config.BATCH_SIZE,
                "image_size": Config.IMAGE_SIZE,
                "epochs": Config.EPOCHS,
                "learning_rate": Config.LEARNING_RATE
            },
            "initial_metrics": { 
                "accuracy": initial_acc,
                "confusion_matrix": initial_cm.tolist()
            },
            "training_history": train_history,
            "validation_report": val_report,
            "confusion_matrix": confusion_matrix.tolist(),
            "sample_prediction": prediction,
            "class_labels": Config.CLASSES
        }
        
        # 8. Save results 
        save_path = save_results(results)
        logger.info(f"Results saved to: {save_path}")
        
    except Exception as e:
        logger.exception("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()