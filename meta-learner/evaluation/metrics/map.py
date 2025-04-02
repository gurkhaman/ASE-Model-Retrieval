import torch


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
