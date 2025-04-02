import torch


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
