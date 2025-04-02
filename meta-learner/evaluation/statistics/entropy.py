import numpy as np
from scipy.stats import entropy


def mean_entropy(predictions):
    """
    Computes mean entropy across all task-level prediction distributions.
    predictions: List of 1D numpy arrays or torch tensors (probabilities/scores)
    """
    return (
        np.mean(
            [
                entropy(p if isinstance(p, np.ndarray) else p.numpy())
                for p in predictions
            ]
        )
        if predictions
        else 0.0
    )
