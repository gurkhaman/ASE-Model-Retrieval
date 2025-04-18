import argparse
import wandb
from utils import load_config
from dataloader import load_data
import torch
from custombaggingsvc import CustomBaggingSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from evaluation import Evaluator, compute_model_ranking_stats
import numpy as np
from customxgboost import CustomXGBoost
import time
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self, model_id, pred_perf, real_perf):
        self.model_id = model_id
        self.pred_perf = pred_perf
        self.real_perf = real_perf

    def __repr__(self):
        return (
            f"Model(model_id='{self.model_id}', "
            f"pred_perf={self.pred_perf:.4f}, "
            f"real_perf={self.real_perf:.4f})"
        )


def load_model(model_name, model_params):
    if model_name == "custom_bagging_svc":
        return CustomBaggingSVC(
            n_estimators=model_params.get("n_estimators", 10),
            random_state=model_params.get("random_state", 42),
        )
    elif model_name == "xgboost":
        return CustomXGBoost(**model_params)
    elif model_name == "randomforest":
        return RandomForestClassifier()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(model_name, model_params, X_train, y_train, X_test, y_test):
    model = load_model(model_name, model_params)
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    y_proba = (
        pipeline.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    )

    return {
        "model": pipeline,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "classes": model.classes_,
    }


def compare_rankings(
    y_proba, real_accuracy_matrix, idx_test, class_indices, model_names, top_k=5
):
    comparisons = []

    for i, task_index in enumerate(idx_test):
        pred_scores = y_proba[i]  # shape: [#models_predicted]
        true_scores = real_accuracy_matrix[task_index][class_indices].clone().numpy()

        pred_rank = np.argsort(pred_scores)[::-1][:top_k]
        true_rank = np.argsort(true_scores)[::-1][:top_k]

        pred_top = [model_names[j] for j in pred_rank]
        true_top = [model_names[j] for j in true_rank]

        comparisons.append(
            {"task_index": task_index, "true_top": true_top, "pred_top": pred_top}
        )

    return comparisons


def evaluate_ranking(
    y_proba, idx_test, real_accuracy_matrix, model_names, class_indices, eval_config
):
    evaluator = Evaluator(
        k_ndcg=eval_config["top-k"]["ndcg"], k_map=eval_config["top-k"]["map"]
    )
    print("Evaluator config:", evaluator.config())

    wandb_table = wandb.Table(
        columns=["task_id", "model_name", "pred_perf", "real_perf"]
    )

    for i, task_index in enumerate(idx_test):
        model_list = [
            Model(
                model_id=model_names[j],
                pred_perf=y_proba[i][j],
                real_perf=real_accuracy_matrix[task_index][class_indices[j]].item(),
            )
            for j in range(len(model_names))
        ]

        # print(f"\nTask {i} (Index {task_index}) models:")
        for m in model_list:
            # print(m)
            wandb_table.add_data(i, m.model_id, m.pred_perf, m.real_perf)

        evaluator.evaluate_task(model_list)

    wandb.log({"Model Evaluations": wandb_table})

    summary = evaluator.summarize(as_table=True)

    columns = list(summary[0].keys())
    table = wandb.Table(columns=columns)
    for row in summary:
        table.add_data(*[row[col] for col in columns])

    wandb.log({"Evaluation Table": table})
    # wandb.log({"Summary Table": summary[-1]})
    print(summary[-1])

    summary = evaluator.summarize(include_stats=True)

    # comparisons = compare_rankings(
    #     y_proba=y_proba,
    #     real_accuracy_matrix=real_accuracy_matrix,
    #     idx_test=idx_test,
    #     class_indices=class_indices,
    #     model_names=model_names,
    #     top_k=eval_config["top-k"]["ndcg"],  # or another top_k value if you prefer
    # )

    # comparison_table = wandb.Table(columns=["task_id", "true_top", "pred_top"])

    # for entry in comparisons:
    #     comparison_table.add_data(
    #         entry["task_index"],
    #         ", ".join(entry["true_top"]),
    #         ", ".join(entry["pred_top"]),
    #     )

    # wandb.log({"Top-K Comparisons": comparison_table})

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    wandb_cfg = config["wandb"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]

    wandb.init(
        project=wandb_cfg.get("project", "meta-learner"),
        name=wandb_cfg.get("run_name", f"{model_cfg['name']}_run"),
        config={"model": model_cfg["name"], **model_cfg.get("params", {})},
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.log({"learner_config": config})
    wandb.log({"device": device})

    learner_data = load_data(
        paths["meta_features_dir"],
        paths["evaluation_results_dir"],
        eval_cfg["exclude"],
        eval_cfg["include_only"],
    )

    model_stats = compute_model_ranking_stats(
        learner_data.evaluation_data, "Top-1 Accuracy", [1]
    )
    print(model_stats)

    all_model_names = learner_data.evaluation_data[0]["model_id"].tolist()

    run_summaries = []
    total_time = 0

    num_runs = wandb_cfg["runs"]
    for run in range(num_runs):
        print(f"\n[RUN {run + 1}/{num_runs}]")

        start_time = time.time()
        results = train_and_evaluate(
            model_name=model_cfg["name"],
            model_params=model_cfg.get("params", {}),
            X_train=learner_data.X_train,
            y_train=learner_data.y_train,
            X_test=learner_data.X_test,
            y_test=learner_data.y_test,
        )

        # Subset of model indices the classifier actually learned
        class_indices = results["classes"]

        summary = evaluate_ranking(
            y_proba=results["y_proba"],
            idx_test=learner_data.idx_test,
            real_accuracy_matrix=learner_data.real_accuracy_matrix,
            model_names=[all_model_names[i] for i in class_indices],
            class_indices=class_indices,
            eval_config=eval_cfg,
        )

        elapsed = time.time() - start_time
        print(f"[⏱️] Total time for training + evaluation: {elapsed:.2f} seconds")
        total_time += elapsed

        wandb.log({f"run_{run + 1}/train_eval_time": elapsed})
        for k, v in summary.items():
            if isinstance(v, (int, float, np.float32, np.float64)):
                wandb.log({f"run_{run + 1}/ranking/{k}": v})

        run_summaries.append(summary)

    metric_keys = run_summaries[0].keys()
    aggregated = {
        k: np.mean([s[k] for s in run_summaries])
        for k in metric_keys
        if isinstance(run_summaries[0][k], (int, float, np.float32, np.float64))
    }

    print("\n[📊] Aggregated Summary (Avg over runs):")
    for k, v in aggregated.items():
        print(f"{k}: {v:.4f}")

    wandb.log({"aggregated_ranking": aggregated})
    wandb.log({"timing/total_train_eval_time": total_time})