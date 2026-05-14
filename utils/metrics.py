import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score


def accuracy(pred, true):
    """Calculate accuracy for classification"""
    pred_classes = np.argmax(pred, axis=1)
    return accuracy_score(true, pred_classes)


def classification_f1(pred, true):
    """Calculate F1 score for classification"""
    pred_classes = np.argmax(pred, axis=1)
    return f1_score(true, pred_classes, average='weighted')


def classification_recall(pred, true):
    """Calculate recall for classification"""
    pred_classes = np.argmax(pred, axis=1)
    return recall_score(true, pred_classes, average='weighted')


def classification_precision(pred, true):
    """Calculate precision for classification"""
    pred_classes = np.argmax(pred, axis=1)
    return precision_score(true, pred_classes, average='weighted')


def classification_confusion_matrix(pred, true):
    """Calculate confusion matrix for classification"""
    pred_classes = np.argmax(pred, axis=1)
    return confusion_matrix(true, pred_classes)


def metric(pred, true):
    """
    Calculate classification metrics
    
    Args:
        pred: Predictions (logits or probabilities)
        true: Ground truth class labels
    
    Returns:
        accuracy, f1_score, recall, precision, confusion_matrix
    """
    acc = accuracy(pred, true)
    f1 = classification_f1(pred, true)
    recall = classification_recall(pred, true)
    precision = classification_precision(pred, true)
    cm = classification_confusion_matrix(pred, true)
    return acc, f1, recall, precision, cm
