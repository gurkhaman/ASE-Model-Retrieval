import torch


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
