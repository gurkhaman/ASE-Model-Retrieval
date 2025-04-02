import torch


def mrr(model_list):
    pred_scores = torch.tensor([m.pred_perf for m in model_list])
    true_scores = torch.tensor([m.real_perf for m in model_list])

    pred_order = torch.argsort(pred_scores, descending=True)
    top_true_idx = torch.argmax(true_scores)

    rank = (pred_order == top_true_idx).nonzero(as_tuple=True)[0].item() + 1
    return 1.0 / rank
