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


def MRR(model_list, sample_size=1):
    mrr = []

    for _ in range(sample_size):
        preds = torch.tensor([m.pred_perf for m in model_list])
        reals = torch.tensor([m.real_perf for m in model_list])

        predicted_order = torch.argsort(preds, descending=True)
        top_real_idx = torch.argmax(reals)

        rank = (predicted_order == top_real_idx).nonzero(as_tuple=True)[0].item() + 1
        mrr.append(1.0 / rank)

    return mrr


def MAP(model_list, k=3, sample_size=1):
    ap_scores = []

    for _ in range(sample_size):
        SDS_scores = torch.tensor([m.pred_perf for m in model_list])
        real_scores = torch.tensor([m.real_perf for m in model_list])

        _, predicted_indices = torch.sort(SDS_scores, descending=True)
        _, ideal_indices = torch.sort(real_scores, descending=True)

        predicted_topk = predicted_indices[:k]
        ideal_topk = set(ideal_indices[:k].tolist())

        hits = 0
        precision_sum = 0.0

        for n, idx in enumerate(predicted_topk):
            if idx.item() in ideal_topk:
                hits += 1
                precision_sum += hits / (n + 1)

        ap = precision_sum / max(1, len(ideal_topk))
        ap_scores.append(ap)

    return ap_scores
