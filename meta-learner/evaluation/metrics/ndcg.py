import torch


def dcg_at_k(scores, k):
    scores = scores[:k]
    positions = torch.arange(
        2, 2 + len(scores), dtype=torch.float32, device=scores.device
    )
    return torch.sum(scores / torch.log2(positions))


def ndcg(model_list, k=5):
    pred_scores = torch.tensor([m.pred_perf for m in model_list], dtype=torch.float32)
    true_scores = torch.tensor([m.real_perf for m in model_list], dtype=torch.float32)

    _, pred_indices = torch.sort(pred_scores, descending=True)
    _, ideal_indices = torch.sort(true_scores, descending=True)

    pred_relevance = true_scores[pred_indices]
    ideal_relevance = true_scores[ideal_indices]

    max_val = true_scores.max()
    if max_val > 0:
        pred_relevance /= max_val
        ideal_relevance /= max_val

    dcg = dcg_at_k(pred_relevance, k)
    idcg = dcg_at_k(ideal_relevance, k)

    return (dcg / idcg).item() if idcg > 0 else 0.0
