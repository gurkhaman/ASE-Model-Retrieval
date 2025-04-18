import torch
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from collections import Counter


def dcg_at_k(scores, k):
    scores = scores[:k]
    positions = torch.arange(
        2, 2 + len(scores), dtype=torch.float32, device=scores.device
    )
    return torch.sum(scores / torch.log2(positions))


def ndcg(model_list, k=5):
    # Step 1: Extract predicted and true performance scores
    pred_scores = torch.tensor([m.pred_perf for m in model_list], dtype=torch.float32)
    true_scores = torch.tensor([m.real_perf for m in model_list], dtype=torch.float32)

    # Step 2: Rank indices
    _, pred_indices = torch.sort(pred_scores, descending=True)
    _, ideal_indices = torch.sort(true_scores, descending=True)

    # Step 3: Top-k indices
    pred_topk = pred_indices[:k]
    ideal_topk = ideal_indices[:k]

    # Step 4: Union of indices for normalization
    combined_indices = torch.unique(torch.cat([pred_topk, ideal_topk]))
    combined_scores = true_scores[combined_indices]

    min_val = combined_scores.min()
    max_val = combined_scores.max()

    if max_val > min_val:
        normalized = (combined_scores - min_val) / (max_val - min_val)
    else:
        normalized = torch.zeros_like(combined_scores)

    # Step 5: Map normalized scores back to top-k relevance
    index_map = {idx.item(): normalized[i] for i, idx in enumerate(combined_indices)}

    pred_relevance = torch.tensor(
        [index_map[idx.item()] for idx in pred_topk], dtype=torch.float32
    )
    ideal_relevance = torch.tensor(
        [index_map[idx.item()] for idx in ideal_topk], dtype=torch.float32
    )

    # Step 6: Compute DCG and IDCG
    dcg = dcg_at_k(pred_relevance, k)
    idcg = dcg_at_k(ideal_relevance, k)

    return (dcg / idcg).item() if idcg > 0 else 0.0


def mrr(model_list):
    pred_scores = torch.tensor([m.pred_perf for m in model_list])
    true_scores = torch.tensor([m.real_perf for m in model_list])

    pred_order = torch.argsort(pred_scores, descending=True)
    top_true_idx = torch.argmax(true_scores)

    rank = (pred_order == top_true_idx).nonzero(as_tuple=True)[0].item() + 1
    return 1.0 / rank


def map_at_k(model_list, k=3):
    pred_scores = torch.tensor([m.pred_perf for m in model_list])
    true_scores = torch.tensor([m.real_perf for m in model_list])

    _, pred_indices = torch.sort(pred_scores, descending=True)
    _, ideal_indices = torch.sort(true_scores, descending=True)

    pred_topk = pred_indices[:k]
    ideal_topk = set(ideal_indices[:k].tolist())

    hits, precision_sum = 0, 0.0
    for n, idx in enumerate(pred_topk):
        if idx.item() in ideal_topk:
            hits += 1
            precision_sum += hits / (n + 1)

    return precision_sum / max(1, len(ideal_topk))


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
