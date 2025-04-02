import numpy as np
from scipy.spatial.distance import cosine


def mean_cosine_similarity(predictions):
    """
    Computes mean pairwise cosine similarity across task-level prediction vectors.
    """
    if len(predictions) < 2:
        return 1.0

    sims = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            a = (
                predictions[i].numpy()
                if hasattr(predictions[i], "numpy")
                else predictions[i]
            )
            b = (
                predictions[j].numpy()
                if hasattr(predictions[j], "numpy")
                else predictions[j]
            )
            sim = 1 - cosine(a, b)
            sims.append(sim)

    return np.mean(sims)
