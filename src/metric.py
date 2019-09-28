import warnings

import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning


def get_score(y_pred, y_labels):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UndefinedMetricWarning)
        return fbeta_score(y_labels, y_pred, beta=2, average='samples')
    
    
def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ Return matrix of 0/1 predictions, same shape as probabilities.
    """
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    
    return mask