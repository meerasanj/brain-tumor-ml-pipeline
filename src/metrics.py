# Imports in order to calculate metrics
from sklearn.metrics import ( 
    classification_report,
    confusion_matrix,
    accuracy_score
)
import numpy as np # For array manipulation
import logging

# Metrics class is a utility class to group metrics-related calculations.
# Can be used without needing to create an instance of the class
class Metrics:
    # Method to calculate a suite of classification metrics
    # y_true (list): A list of the true, ground-truth labels.
    # y_pred (list): A list of the labels predicted by the model.
    @staticmethod
    def calculate(y_true: list, y_pred: list) -> dict:
        # Returns dictionary containing  accuracy score, classification report, and confusion matrix
        return {
            # Proportion of correct predictions
            "accuracy": accuracy_score(y_true, y_pred),
            # Precision, recall, and F1-score for each class
            "classification_report": classification_report( 
                y_true, y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
