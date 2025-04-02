from collections import Counter


def top1_diversity(predictions):
    """
    Returns number of unique top-1 model predictions.
    """
    top1 = [
        p.argmax().item() if hasattr(p, "argmax") else p.argmax() for p in predictions
    ]
    return len(set(top1)), Counter(top1).most_common()


def topk_diversity(predictions, k=3):
    """
    Returns number of unique top-k sets (as tuples) in predictions.
    """
    topk = [
        tuple(p.argsort(descending=True)[:k].tolist())
        if hasattr(p, "argsort")
        else tuple(p.argsort()[-k:][::-1])
        for p in predictions
    ]
    return len(set(topk))
