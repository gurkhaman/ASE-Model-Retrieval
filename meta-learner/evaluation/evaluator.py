import torch
import numpy as np
from collections import defaultdict

from evaluation.metrics import ndcg, mrr, map_at_k
from evaluation.statistics import (
    mean_entropy,
    mean_cosine_similarity,
    top1_diversity,
    topk_diversity,
)


class Model:
    def __init__(self, model_id, pred_perf, real_perf):
        self.model_id = model_id
        self.pred_perf = pred_perf
        self.real_perf = real_perf


class Evaluator:
    def __init__(self, k_ndcg=5, k_map=3):
        self.k_ndcg = k_ndcg
        self.k_map = k_map

        self.prediction_proba = []
        self.ndcg_scores = []
        self.mrr_scores = []
        self.map_scores = []

    def evaluate_task(self, model_list):
        self.ndcg_scores.append(ndcg(model_list, k=self.k_ndcg))
        self.mrr_scores.append(mrr(model_list))
        self.map_scores.append(map_at_k(model_list, k=self.k_map))

        pred_scores = torch.tensor(
            [m.pred_perf for m in model_list], dtype=torch.float32
        )
        self.prediction_proba.append(pred_scores.detach().cpu())

    def evaluate_all(self, all_model_lists):
        for model_list in all_model_lists:
            self.evaluate_task(model_list)

    def summarize(self, include_stats=True):
        summary = {
            f"Mean NDCG@{self.k_ndcg}": np.mean(self.ndcg_scores),
            "Mean MRR": np.mean(self.mrr_scores),
            f"Mean MAP@{self.k_map}": np.mean(self.map_scores),
        }

        if include_stats:
            summary["Mean Entropy"] = mean_entropy(self.prediction_proba)
            summary["Mean Cosine Similarity"] = mean_cosine_similarity(
                self.prediction_proba
            )

            unique_top1, top1_freq = top1_diversity(self.prediction_proba)
            summary["Unique Top-1 Predictions"] = unique_top1
            summary["Top-1 Frequencies"] = dict(top1_freq)

            summary["Unique Top-3 Predictions"] = topk_diversity(
                self.prediction_proba, k=3
            )

        return summary

    def reset(self):
        self.prediction_proba = []
        self.ndcg_scores = []
        self.mrr_scores = []
        self.map_scores = []
