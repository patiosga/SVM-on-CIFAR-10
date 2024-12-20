from sklearn.metrics import classification_report
import numpy as np
from variables import labels


def accuracy(test_labels, predicted_labels):
    acc = np.mean(predicted_labels == test_labels)
    return acc 


def cls_report(test_labels, predicted_labels):
    # Αναλυτική αναφορά ταξινόμησης (precision, recall, f1-score, support) για κάθε κλάση ξεχωριστά
    report = classification_report(test_labels, predicted_labels, target_names=['dog', 'truck'], zero_division=1)
    return report
