import torch


def dcg_at_k(scores, k):
    scores = scores[:k]
    positions = torch.arange(
        2, 2 + len(scores), dtype=torch.float32, device=scores.device
    )
    return torch.sum(scores / torch.log2(positions))


def NDCG(model_list, k=5, sample_size=1):
    ndcg_scores = []

    for _ in range(sample_size):
        predicted_scores = torch.tensor(
            [m.pred_perf for m in model_list], dtype=torch.float32
        )
        true_scores = torch.tensor(
            [m.real_perf for m in model_list], dtype=torch.float32
        )

        # Sort by predicted performance to get predicted ranking
        _, predicted_indices = torch.sort(predicted_scores, descending=True)
        predicted_relevance = true_scores[predicted_indices]

        # Sort by ground truth to get ideal ranking
        _, ideal_indices = torch.sort(true_scores, descending=True)
        ideal_relevance = true_scores[ideal_indices]

        max_val = true_scores.max()
        if max_val > 0:
            predicted_relevance = predicted_relevance / max_val
            ideal_relevance = ideal_relevance / max_val

        dcg = dcg_at_k(predicted_relevance, k)
        idcg = dcg_at_k(ideal_relevance, k)

        ndcg = (dcg / idcg).item() if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return ndcg_scores
