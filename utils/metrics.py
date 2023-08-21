
from __future__ import annotations

from collections.abc import Callable
import json

import numpy as np
from sklearn.metrics import classification_report


def test_classification(
    targets: np.ndarray, predictions: np.ndarray, labels: list=None, digits: int=6,
    filename: str=None, print_fn: Callable=None
) -> None|dict:
    """Calculate and visualizing scores for classification (mostly using sklearn).

    Args:
        targets (np.ndarray): Ground truth.
        predictions (np.ndarray): Predictions.
        labels (list, optional): List containing names of the classes.. Defaults to None.
        digits (int, optional): digits argument for classification_report. Default: 6.
        return_dict (bool, optional): return classification report as dict? Default: True.
        filename (str, optional): If provided, visualize the confusion matrix via matplotlib. Defaults to None.
        print_fn (Callable, optional): A callable for printing the results. Defaults to print.

    Returns:
        dict: classification report as dict if return_dict is True.
    """
    labels = np.array(labels)

    if print_fn:
        print_fn('TEST')
        # precision, recall, F1, accuracy
        print_fn(
            f'Classification report:\n{classification_report(targets, predictions, target_names=labels, digits=digits)}')

    cls_report_dict = classification_report(targets, predictions, output_dict=True)

    if filename:
        with open(filename, 'w') as fp:
            json.dump(cls_report_dict, fp, indent=2)

    return cls_report_dict
