from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import numpy as np
import logging

class Metrics:
    @staticmethod
    def calculate(y_true: list, y_pred: list) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }