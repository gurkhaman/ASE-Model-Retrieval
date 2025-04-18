from collections import Counter
from metrics import (
    ndcg,
    mrr,
    map_at_k,
    mean_cosine_similarity,
    top1_diversity,
    topk_diversity,
    mean_entropy,
)
import torch
import numpy as np


class Evaluator:
    def __init__(self, k_ndcg, k_map):
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

    def summarize(self, include_stats=True, as_table=False):
        """
        Returns summary as a dict (default) or list-of-dicts (table mode).
        """
        # Per-task rows
        rows = []
        for idx, (ndcg_score, mrr_score, map_score) in enumerate(
            zip(self.ndcg_scores, self.mrr_scores, self.map_scores)
        ):
            rows.append(
                {
                    "Type": "Task",
                    "Task": idx,
                    f"NDCG@{self.k_ndcg}": float(ndcg_score),
                    "MRR": float(mrr_score),
                    f"MAP@{self.k_map}": float(map_score),
                }
            )

        # Mean row
        summary_row = {
            "Type": "Summary",
            "Task": None,
            f"NDCG@{self.k_ndcg}": float(np.mean(self.ndcg_scores)),
            "MRR": float(np.mean(self.mrr_scores)),
            f"MAP@{self.k_map}": float(np.mean(self.map_scores)),
        }

        if include_stats:
            summary_row["Entropy"] = float(mean_entropy(self.prediction_proba))
            summary_row["Cosine Similarity"] = float(
                mean_cosine_similarity(self.prediction_proba)
            )

        rows.append(summary_row)

        if as_table:
            return rows  # For table logging
        else:
            return {
                k: v for k, v in summary_row.items() if k not in {"Type", "Task"}
            }  # Flat dict

    def config(self):
        return {
            "NDCG@k": self.k_ndcg,
            "MAP@k": self.k_map,
        }


def get_top_k_model_counts(evaluation_data, target, k):
    """
    Count how many times each model appears in the top-k for the target metric across all tasks.
    """
    topk_models = []
    for df in evaluation_data:
        if target in df.columns and not df.empty:
            top_models = df.nlargest(k, target)["model_id"].values
            topk_models.extend(top_models)
    counts = Counter(topk_models)
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    return sorted_counts


def compute_model_ranking_stats(evaluation_data, target, topk_levels):
    ranking_stats = {}
    for k in topk_levels:
        ranking_stats[f"top{k}_counts"] = get_top_k_model_counts(
            evaluation_data, target, k
        )
    return ranking_stats
